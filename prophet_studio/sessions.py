"""Sessions d'agent : un tour = Prophet en flux continu dans un fil, autorisations demandees a l'interface.

La transcription est reconstruite a partir des evenements par le meme reducteur que l'interface (ui/src/lib/
transcript.ts) : elle est sauvegardee a chaque tour et rejouee a la reouverture d'une session.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Callable

from jev_clone.backend_llamacpp import LlamaCppBackend
from jev_clone.engine import SystemOneEngine
from jev_clone.desktop_use import make_desktop_factory
from jev_clone.guard import grant_for
from jev_clone.prophet import Prophet, Workspace, make_browser_factory
from prophet_studio import calibration as s1cal


def new_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]


# ---- reducteur de transcription (miroir de ui/src/lib/transcript.ts) -------------------------------------------------
def _close_thinking(blocks: list, ts: float) -> None:
    if blocks and blocks[-1]["type"] == "thinking" and blocks[-1].get("ms") is None and blocks[-1].get("started"):
        blocks[-1]["ms"] = round((ts - blocks[-1]["started"]) * 1000, 1)


CU_KEEP = 40   # pas du computer use gardes par outil : session sauvegardee bornee


def _cu_clip(v, n: int = 200):
    return (v or "")[:n] or None


def reduce_computer(blocks: list, evt: dict) -> None:
    """Progression par pas de browse / desktop (computer.escalate, .think, .action, .step), rattachee a l'outil en cours du
    meme nom (Prophet execute ses outils un par un) : pas (voie, action, probabilite, raison, actions de Bonsai), escalades
    et etat en direct. Les actions de Bonsai arrivent avant le computer.step de leur pas."""
    b = next((b for b in reversed(blocks) if b["type"] == "tool" and b.get("status") == "running" and b["name"] == evt.get("tool")), None)
    if b is None:
        return
    cu = b.setdefault("cu", {"steps": [], "n": 0, "escalations": 0, "pending": [], "live": None})
    t = evt["type"]
    base = cu["live"] or {"kind": "", "step": cu["n"], "why": None, "turn": None, "action": None, "blocked": False}
    if t == "computer.escalate":
        cu["escalations"] = evt.get("escalations") or cu["escalations"]
        cu["live"] = {"kind": "escalate", "step": evt["step"] if evt.get("step") is not None else cu["n"], "why": _cu_clip(evt.get("why")),
                      "turn": None, "action": None, "blocked": False}
    elif t == "computer.think":
        cu["live"] = {**base, "kind": "think", "turn": evt.get("turn") or 0}
    elif t == "computer.action":
        a = {"action": evt.get("action"), "blocked": bool(evt.get("blocked")), "ok": bool(evt.get("ok"))}
        cu["pending"].append(a)
        cu["live"] = {**base, "kind": "action", "action": a["action"], "blocked": a["blocked"]}
    elif t == "computer.step":
        slow = cu["pending"] or [{"action": x, "blocked": False, "ok": None} for x in evt.get("slow_actions") or []]
        cu["steps"] = (cu["steps"] + [{"step": evt["step"] if evt.get("step") is not None else cu["n"], "path": evt.get("path"),
                                        "action": evt.get("action"), "p": evt.get("p"), "why": _cu_clip(evt.get("why")),
                                        "verify": evt.get("verify"), "error": _cu_clip(evt.get("error")),
                                        "escalations": evt.get("escalations") or 0, "slow": slow}])[-CU_KEEP:]
        cu.update(n=cu["n"] + 1, pending=[], live=None, escalations=evt.get("escalations") or cu["escalations"])


def reduce_event(item: dict, evt: dict) -> None:
    blocks = item.setdefault("blocks", [])
    t = evt.get("type")
    ts = evt.get("ts") or time.time()
    if t in ("thinking.delta", "text.delta"):
        kind = "thinking" if t == "thinking.delta" else "text"
        if kind == "text":
            _close_thinking(blocks, ts)
        # jamais fusionne par-dessus une voie directe ecartee : la voie agent commence un bloc a elle
        if blocks and blocks[-1]["type"] == kind and len(blocks) > (item.get("reroute") or {}).get("at", 0):
            blocks[-1]["text"] += evt.get("text", "")
        else:
            blocks.append({"type": kind, "text": evt.get("text", ""), **({"started": ts} if kind == "thinking" else {})})
    elif t in ("tool.pending", "llm.end"):
        _close_thinking(blocks, ts)
    elif t == "tool.call":
        _close_thinking(blocks, ts)
        blocks.append({"type": "tool", "id": evt["id"], "name": evt["name"], "args": evt.get("args", {}), "status": "running", "started": ts})
    elif t == "tool.result":
        for b in reversed(blocks):
            if b["type"] == "tool" and b["id"] == evt["id"]:
                b.update(status="done" if evt.get("ok") else "error", ok=evt.get("ok"), result=evt.get("result"), ui=evt.get("ui"), ended=ts)
                break
    elif t == "permission.request":
        blocks.append({"type": "permission", "id": evt["id"], "tool": evt.get("tool"), "describe": evt.get("describe"),
                       "preview": evt.get("preview"), "judged": evt.get("judged"), "decision": None})
    elif t == "permission.resolved":
        for b in blocks:
            if b["type"] == "permission" and b["id"] == evt["id"]:
                b["decision"] = "allow" if evt.get("allow") else "deny"
    elif t == "s1.decision":   # calibrated / s1_model : l'etat du classifieur pour CE tour (pas celui du moment de l'affichage)
        item["s1"] = {k: evt.get(k) for k in ("pre", "latency_ms", "budget", "risk_level", "path", "calibrated", "s1_model", "gates")}
    elif t == "s1.tools":
        item.setdefault("s1", {})["tools"] = evt.get("relevance")
    elif t == "s1.reroute":   # la voie directe est ecartee : les blocs deja la sont la premiere reponse, remplacee
        item["reroute"] = {"reason": evt.get("reason"), "verification": evt.get("verification"), "budget": evt.get("budget"), "at": len(blocks)}
        s1 = item.setdefault("s1", {})   # voie et budget reels du tour ; la decision de S1 reste dans rerouted_from
        s1.update(rerouted_from=s1.get("path"), path=evt.get("to") or "agent")
        if evt.get("budget") is not None:
            s1["budget"] = evt["budget"]
    elif t == "turn.end":
        item.update({k: evt.get(k) for k in ("path", "response", "verification", "latency_ms", "stopped_by", "stats")})
        item["status"] = "done"
    elif t == "turn.error":
        item["status"], item["error"] = "error", evt.get("error")
    elif t in ("computer.escalate", "computer.think", "computer.action", "computer.step"):
        reduce_computer(blocks, evt)


class SessionStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def path(self, sid: str) -> Path:
        if not sid.replace("-", "").isalnum():
            raise ValueError("identifiant de session invalide")
        return self.root / f"{sid}.json"

    def create(self, workspace: str, title: str = "") -> dict:
        s = {"id": new_id(), "title": title or "Nouvelle session", "created": time.time(), "updated": time.time(), "workspace": workspace,
             "history": [], "transcript": [], "always_allow": []}
        self.save(s)
        return s

    def load(self, sid: str) -> dict:
        return json.loads(self.path(sid).read_text(encoding="utf-8"))

    def save(self, s: dict) -> None:
        with self._lock:
            s["updated"] = time.time()
            p = self.path(s["id"])
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(s, ensure_ascii=False, default=str), encoding="utf-8")
            tmp.replace(p)

    def list(self) -> list[dict]:
        out = []
        for f in self.root.glob("*.json"):
            try:
                s = json.loads(f.read_text(encoding="utf-8"))
                out.append({"id": s["id"], "title": s["title"], "updated": s["updated"], "workspace": s.get("workspace"),
                            "turns": sum(1 for i in s.get("transcript", []) if i.get("kind") == "user")})
            except Exception:
                continue
        return sorted(out, key=lambda x: -x["updated"])

    def delete(self, sid: str) -> None:
        self.path(sid).unlink(missing_ok=True)

    def set_always_allow(self, sid: str, allow: list[str]) -> None:
        """Ecrit seulement always_allow, relu et reecrit sous le verrou (jamais une copie perimee de la transcription)."""
        with self._lock:
            p = self.path(sid)
            s = json.loads(p.read_text(encoding="utf-8"))
            s["always_allow"] = list(allow)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(s, ensure_ascii=False, default=str), encoding="utf-8")
            tmp.replace(p)


class PendingPermission:
    def __init__(self, session_id: str, tool: str):
        self.id = uuid.uuid4().hex[:10]
        self.session_id, self.tool = session_id, tool
        self.event = threading.Event()
        self.allow = False
        self.remember = False


class AgentService:
    def __init__(self, store: SessionStore, publish: Callable[[dict], None], urls: Callable[[], tuple[str, str]],
                 settings: Callable, ctx: Callable[[], int], calibration: str | Callable[[], str | Path | None] | None = None,
                 runs_dir: Path | None = None, desktop_backend: Callable | None = None):
        self.store, self.publish, self.urls, self.settings, self.ctx = store, publish, urls, settings, ctx
        self.calibration = calibration   # chemin, ou rappel -> fichier du classifieur en marche (None en mode mono)
        self.running: dict[str, dict] = {}           # session_id -> {"turn_id", "cancel": Event, "thread"}
        self.permissions: dict[str, PendingPermission] = {}
        self._engines: tuple | None = None
        self._lock = threading.Lock()
        # evenements de tour : reduits et publies sous ce verrou, dans l'ordre de leur numero (tseq) ; snapshot() y copie le
        # tour en cours, pour qu'une interface rechargee reprenne exactement la ou la copie s'arrete
        self._live = threading.Lock()
        self.runs_dir = runs_dir
        self.desktop_backend = desktop_backend   # None = Windows UI Automation ; la demo passe un bureau simule

    def calibration_path(self) -> str | None:
        """Calibration a appliquer : celle du classifieur en marche, s'il en a une (jamais celle d'un autre modele)."""
        c = self.calibration() if callable(self.calibration) else self.calibration
        return str(c) if c and Path(c).is_file() else None

    def s1_model(self) -> str | None:
        """Classifieur en marche (nom de son fichier de calibration, meme absent) ; None en mode mono ou sans rappel."""
        c = self.calibration() if callable(self.calibration) else None
        return Path(c).stem if c else None

    def invalidate_engines(self) -> None:
        self._engines = None

    @staticmethod
    def _stamp(p) -> tuple | None:
        try:
            st = Path(p).stat()
            return st.st_size, st.st_mtime_ns
        except (OSError, TypeError):
            return None

    def engines(self) -> tuple[SystemOneEngine, LlamaCppBackend]:
        s1_url, s2_url = self.urls()
        cal, model = self.calibration_path(), self.s1_model()
        # autre classifieur, calibration ajoutee / remplacee, GGUF calibre remplace sur place : moteur reconstruit
        key = (s1_url, s2_url, model, cal, self._stamp(cal))
        e = self._engines
        if e is None or e[2] != key or (e[3] is not None and self._stamp(e[3]) != e[4]):
            calibration, _why, w = s1cal.load_for_engine(cal)   # illisible, perimee, generique : lecture brute (dit dans l'etat)
            # nom du classifieur en marche, porte par chaque decision de tour ("" en mode mono : Bonsai repond)
            s1 = SystemOneEngine(LlamaCppBackend(s1_url, max_workers=4, timeout=120), calibration=calibration, model_name=model or "")
            s2 = LlamaCppBackend(s2_url, max_workers=1, timeout=900)
            self._engines = (s1, s2, key, w, self._stamp(w))
        return self._engines[0], self._engines[1]

    # ---- tours -----------------------------------------------------------------------------------------------------
    def submit(self, sid: str, text: str, plan_mode: bool = False, permission_mode: str | None = None, effort: str | None = None) -> str:
        with self._lock:
            if sid in self.running:
                raise RuntimeError("un tour est deja en cours dans cette session")
            session = self.store.load(sid)
            turn_id = uuid.uuid4().hex[:10]
            cancel = threading.Event()
            self.running[sid] = {"turn_id": turn_id, "cancel": cancel, "session": session}   # session vivante : revocation immediate
        st = self.settings()
        pm = permission_mode or st.permission_mode
        eff = effort or st.effort
        user_item = {"kind": "user", "text": text, "ts": time.time(), "turn_id": turn_id, "plan_mode": plan_mode}
        item = {"kind": "assistant", "turn_id": turn_id, "blocks": [], "status": "running", "ts": time.time(), "plan_mode": plan_mode}
        session["transcript"] += [user_item, item]
        if session["title"] == "Nouvelle session":
            session["title"] = (text.strip().splitlines() or ["Session"])[0][:64]
        self.store.save(session)

        def emit(evt: dict) -> None:
            with self._live:
                evt = {**evt, "session_id": sid, "turn_id": turn_id, "ts": round(time.time(), 3), "tseq": item.get("tseq", 0) + 1}
                reduce_event(item, evt)
                item["tseq"] = evt["tseq"]
                self.publish(evt)

        def confirm(describe: str, judged: dict) -> bool:
            tool = judged.get("tool", "action")
            # « toujours » vaut pour un couple (outil, classe de risque jugee par S1) ; sans jugement, pour l'outil seul.
            # Jamais pour un arret obligatoire (dangereux probable, politique, S1 en panne, action vue en partie, code
            # invisible) ni pour un verdict incertain : une autorisation ne couvre que ce que le mode smart aurait laisse
            # passer sans demander (jev_clone.guard.grant_for). Vaut aussi pour les pas du computer use (desktop_step...).
            grant = grant_for(tool, judged)
            if grant is not None and grant in session.get("always_allow", []):
                return True
            pp = PendingPermission(sid, tool)
            self.permissions[pp.id] = pp
            emit({"type": "permission.request", "id": pp.id, "tool": tool, "describe": describe[:4000], "preview": judged.get("preview"),
                  "judged": {**{k: v for k, v in judged.items() if k not in ("preview",)}, "grant": grant}})
            while not pp.event.wait(0.25):
                if cancel.is_set():
                    break
            self.permissions.pop(pp.id, None)
            remembered = pp.allow and pp.remember and grant is not None
            if remembered and grant not in session.setdefault("always_allow", []):
                session["always_allow"].append(grant)
            emit({"type": "permission.resolved", "id": pp.id, "allow": pp.allow, "remember": remembered, "grant": grant if remembered else None})
            return pp.allow

        def run():
            self.publish({"type": "turn.queued", "session_id": sid, "turn_id": turn_id, "text": text})
            try:
                s1, s2 = self.engines()
                ctx = max(4096, self.ctx())
                ws = Workspace(session.get("workspace") or st.workspace)
                # computer use : un pas risque passe par la meme demande d'autorisation (sauf mode « jamais »), un evenement
                # par pas, annulation par Stop, trajectoires journalisees pour re-entrainer la politique rapide (DAgger)
                from jev_clone.computer_use import browser_available
                from jev_clone.desktop_use import desktop_available
                runs = self.runs_dir or self.store.root.parent / "runs"
                max_tokens = max(1024, min(8192, ctx // 3))   # Bonsai des outils browse / desktop : meme generation, meme part de reflexion
                cu = {"confirm": (lambda d, j: True) if pm == "auto" else confirm, "on_event": emit, "should_stop": cancel.is_set,
                      "max_tokens": max_tokens}
                # outil indisponible (sans Playwright / Chromium ; hors Windows pour le bureau) : jamais propose a Bonsai
                browser_ok = st.browser_tool and browser_available()[0]
                desktop_ok = st.desktop_tool and (self.desktop_backend is not None or desktop_available()[0])
                prophet = Prophet(s1, s2, ws, confirm=confirm, max_turns=24, on_event=emit, should_stop=cancel.is_set,
                                  permission_mode=pm, plan_mode=plan_mode, max_context_chars=int(ctx * 3.2 * 0.7),
                                  max_tokens=max_tokens,
                                  browser_factory=make_browser_factory(s1, s2, ledger=runs / "trajectories.jsonl", **cu) if browser_ok else None,
                                  desktop_factory=make_desktop_factory(s1, s2, self.desktop_backend, ledger=runs / "desktop_trajectories.jsonl", **cu)
                                  if desktop_ok else None)
                turn = prophet.handle(text, session["history"], effort=eff)
                session["history"] += [{"role": "user", "content": text}, {"role": "assistant", "content": turn.response}]
            except Exception as e:
                emit({"type": "turn.error", "error": f"{type(e).__name__}: {str(e)[:500]}"})
            finally:
                if item.get("status") == "running":
                    item["status"] = "done"
                try:
                    self.store.save(session)
                finally:
                    with self._lock:
                        self.running.pop(sid, None)
                    self.publish({"type": "turn.finished", "session_id": sid, "turn_id": turn_id})

        th = threading.Thread(target=run, daemon=True, name=f"turn-{turn_id}")
        self.running[sid]["thread"] = th
        th.start()
        return turn_id

    def cancel(self, sid: str) -> bool:
        r = self.running.get(sid)
        if not r:
            return False
        r["cancel"].set()
        for pp in list(self.permissions.values()):
            if pp.session_id == sid:
                pp.event.set()
        return True

    def respond(self, perm_id: str, allow: bool, remember: bool = False) -> bool:
        # retrait atomique : la premiere reponse fait foi (double clic, 2e fenetre, voix + clavier) ; apres Stop, expiree
        pp = self.permissions.pop(perm_id, None)
        if not pp or pp.event.is_set():
            return False
        pp.allow, pp.remember = allow, remember
        pp.event.set()
        return True

    def pending_permissions(self) -> list[dict]:
        return [{"id": p.id, "session_id": p.session_id, "tool": p.tool} for p in self.permissions.values()]

    def snapshot(self, sid: str) -> dict | None:
        """Copie de la session d'un tour en cours (blocs deja recus, demande d'autorisation en attente), None sinon. Le dernier
        element porte tseq : l'interface n'applique ensuite que les evenements plus recents (rechargement, changement de session)."""
        r = self.running.get(sid)
        if r is None:
            return None
        with self._live:
            return {**json.loads(json.dumps(r["session"], ensure_ascii=False, default=str)), "running": True}

    # ---- autorisations memorisees (« toujours pour cet outil ») : visibles et revocables --------------------------------
    def always_allow(self, sid: str) -> list[str]:
        live = (self.running.get(sid) or {}).get("session")
        return list((live if live is not None else self.store.load(sid)).get("always_allow", []))

    def revoke(self, sid: str, grant: str | None = None) -> list[str]:
        """Retire une autorisation (toutes si grant est None). Un tour en cours la perd aussitot (meme objet de session)."""
        with self._lock:
            live = (self.running.get(sid) or {}).get("session")
            s = live if live is not None else self.store.load(sid)
            allow = s.setdefault("always_allow", [])
            allow[:] = [g for g in allow if grant is not None and g != grant]
            self.store.set_always_allow(sid, allow)
            return list(allow)

    def wait_idle(self, sid: str, timeout: float = 30) -> bool:
        t0 = time.time()
        while sid in self.running and time.time() - t0 < timeout:
            time.sleep(0.02)
        return sid not in self.running

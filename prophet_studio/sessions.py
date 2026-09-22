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
from jev_clone.prophet import Prophet, Workspace, make_browser_factory
from jev_clone.readout import Calibration


def new_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]


# ---- reducteur de transcription (miroir de ui/src/lib/transcript.ts) -------------------------------------------------
def _close_thinking(blocks: list, ts: float) -> None:
    if blocks and blocks[-1]["type"] == "thinking" and blocks[-1].get("ms") is None and blocks[-1].get("started"):
        blocks[-1]["ms"] = round((ts - blocks[-1]["started"]) * 1000, 1)


def reduce_event(item: dict, evt: dict) -> None:
    blocks = item.setdefault("blocks", [])
    t = evt.get("type")
    ts = evt.get("ts") or time.time()
    if t in ("thinking.delta", "text.delta"):
        kind = "thinking" if t == "thinking.delta" else "text"
        if kind == "text":
            _close_thinking(blocks, ts)
        if blocks and blocks[-1]["type"] == kind:
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
    elif t == "s1.decision":
        item["s1"] = {k: evt.get(k) for k in ("pre", "latency_ms", "budget", "risk_level", "path")}
    elif t == "s1.tools":
        item.setdefault("s1", {})["tools"] = evt.get("relevance")
    elif t == "turn.end":
        item.update({k: evt.get(k) for k in ("path", "response", "verification", "latency_ms", "stopped_by", "stats")})
        item["status"] = "done"
    elif t == "turn.error":
        item["status"], item["error"] = "error", evt.get("error")


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


class PendingPermission:
    def __init__(self, session_id: str, tool: str):
        self.id = uuid.uuid4().hex[:10]
        self.session_id, self.tool = session_id, tool
        self.event = threading.Event()
        self.allow = False
        self.remember = False


class AgentService:
    def __init__(self, store: SessionStore, publish: Callable[[dict], None], urls: Callable[[], tuple[str, str]],
                 settings: Callable, ctx: Callable[[], int], calibration: str | None = None, runs_dir: Path | None = None):
        self.store, self.publish, self.urls, self.settings, self.ctx = store, publish, urls, settings, ctx
        self.calibration = calibration
        self.running: dict[str, dict] = {}           # session_id -> {"turn_id", "cancel": Event, "thread"}
        self.permissions: dict[str, PendingPermission] = {}
        self._engines: tuple | None = None
        self._lock = threading.Lock()
        self.runs_dir = runs_dir

    def engines(self) -> tuple[SystemOneEngine, LlamaCppBackend]:
        s1_url, s2_url = self.urls()
        if self._engines is None or self._engines[2] != (s1_url, s2_url):
            s1 = SystemOneEngine(LlamaCppBackend(s1_url, max_workers=4, timeout=120), calibration=Calibration.load(self.calibration),
                                 model_name="systemone")
            s2 = LlamaCppBackend(s2_url, max_workers=1, timeout=900)
            self._engines = (s1, s2, (s1_url, s2_url))
        return self._engines[0], self._engines[1]

    # ---- tours -----------------------------------------------------------------------------------------------------
    def submit(self, sid: str, text: str, plan_mode: bool = False, permission_mode: str | None = None, effort: str | None = None) -> str:
        with self._lock:
            if sid in self.running:
                raise RuntimeError("un tour est deja en cours dans cette session")
            session = self.store.load(sid)
            turn_id = uuid.uuid4().hex[:10]
            cancel = threading.Event()
            self.running[sid] = {"turn_id": turn_id, "cancel": cancel}
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
            evt = {**evt, "session_id": sid, "turn_id": turn_id, "ts": round(time.time(), 3)}
            reduce_event(item, evt)
            self.publish(evt)

        def confirm(describe: str, judged: dict) -> bool:
            tool = judged.get("tool", "action")
            if tool in session.get("always_allow", []):
                return True
            pp = PendingPermission(sid, tool)
            self.permissions[pp.id] = pp
            emit({"type": "permission.request", "id": pp.id, "tool": tool, "describe": describe[:4000], "preview": judged.get("preview"),
                  "judged": {k: v for k, v in judged.items() if k not in ("preview",)}})
            while not pp.event.wait(0.25):
                if cancel.is_set():
                    break
            self.permissions.pop(pp.id, None)
            if pp.allow and pp.remember and tool not in session.setdefault("always_allow", []):
                session["always_allow"].append(tool)
            emit({"type": "permission.resolved", "id": pp.id, "allow": pp.allow, "remember": pp.remember})
            return pp.allow

        def run():
            self.publish({"type": "turn.queued", "session_id": sid, "turn_id": turn_id, "text": text})
            try:
                s1, s2 = self.engines()
                ctx = max(4096, self.ctx())
                ws = Workspace(session.get("workspace") or st.workspace)
                prophet = Prophet(s1, s2, ws, confirm=confirm, max_turns=24, on_event=emit, should_stop=cancel.is_set,
                                  permission_mode=pm, plan_mode=plan_mode, max_context_chars=int(ctx * 3.2 * 0.7),
                                  max_tokens=max(1024, min(8192, ctx // 3)),
                                  browser_factory=make_browser_factory(s1, s2) if st.browser_tool else None)
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
        pp = self.permissions.get(perm_id)
        if not pp:
            return False
        pp.allow, pp.remember = allow, remember
        pp.event.set()
        return True

    def pending_permissions(self) -> list[dict]:
        return [{"id": p.id, "session_id": p.session_id, "tool": p.tool} for p in self.permissions.values()]

    def wait_idle(self, sid: str, timeout: float = 30) -> bool:
        t0 = time.time()
        while sid in self.running and time.time() - t0 < timeout:
            time.sleep(0.02)
        return sid not in self.running

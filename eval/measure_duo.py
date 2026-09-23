"""Mesure du duo reel (S1 = classifieur, S2 = modele de raisonnement) sur votre machine, en une commande.

    python -m eval.measure_duo                                   # Prophet Studio en marche (core.json du dossier de donnees)
    python -m eval.measure_duo --studio http://127.0.0.1:7878 --token XYZ
    python -m eval.measure_duo --s1 http://127.0.0.1:8081 --s2 http://127.0.0.1:8080   # deux llama-server, sans Studio

Mesure : latence S1 par type de question a froid / a chaud (p50 / p95), lecture des etiquettes (grammaire et libre), S2
generation / prefill / premier token, budget de reflexion respecte (0 puis 512), un tour Prophet voie directe et un tour voie
agent (fichier cree dans un espace de travail temporaire, permission_mode auto), VRAM avant / pic / apres (nvidia-smi).
Ecrit eval/results/duo-<date>.json et affiche les lignes a coller dans eval/SCOREBOARD.md (section "Duo reel").
Mode Studio : les tours passent par l'API du coeur (comme l'interface), les mesures S1 / S2 vont directement aux llama-server
qu'il supervise. Scripts : scripts/measure_rtx5060.ps1 (Windows), scripts/measure.sh (Linux, macOS).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import random
import sys
import time
import unicodedata
from pathlib import Path

if __package__ in (None, ""):   # lance par chemin (python eval/measure_duo.py) : le depot d'abord
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests  # noqa: E402

from eval import duo_probes as dp  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
TABLE_HEAD = ["| Mesure (duo reel) | Estimation | Mesure | Conditions |", "|---|---|---|---|"]


# ---- Prophet Studio ------------------------------------------------------------------------------------------------------
def default_data_dir() -> Path:
    """Meme regle que prophet_studio.config.data_dir, sans rien creer."""
    if os.environ.get("PROPHET_HOME"):
        return Path(os.environ["PROPHET_HOME"])
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "ProphetStudio"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "ProphetStudio"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "prophet-studio"


def _alive(url: str) -> bool:
    try:
        r = requests.get(f"{url.rstrip('/')}/api/health", timeout=3)
        return r.ok and r.json().get("app") == "prophet-studio"
    except Exception:
        return False


def find_core(data_dir: Path | None) -> dict | None:
    """core.json d'un coeur qui repond : <data>/core.json, puis <data>/demo/core.json (prophet-studio --demo)."""
    root = Path(data_dir) if data_dir else default_data_dir()
    for f in (root / "core.json", root / "demo" / "core.json"):
        try:
            info = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if info.get("url") and _alive(info["url"]):
            return {**info, "core_json": str(f)}
    return None


class StudioApi:
    def __init__(self, url: str, token: str):
        self.url = url.rstrip("/")
        self.s = requests.Session()
        self.s.headers["X-Prophet-Token"] = token

    def call(self, method: str, path: str, body: dict | None = None, timeout: float = 60) -> dict:
        r = self.s.request(method, self.url + path, json=body, timeout=timeout)
        if r.status_code == 401:
            raise SystemExit("jeton refuse par Prophet Studio : --token, ou relancez sans --studio URL pour lire core.json")
        if r.status_code >= 400:
            raise RuntimeError(f"Studio {method} {path} : {r.status_code} {r.text[:300]}")
        return r.json()


def studio_ready(api: StudioApi, start: bool, timeout: float = 900, log=print) -> dict:
    """Etat du coeur ; avec start, demarre les modeles s'ils sont arretes et attend qu'ils soient prets."""
    state = api.call("GET", "/api/state")
    rt = state["runtime"]
    if rt["state"] in ("stopped", "error") and start:
        log("Modeles arretes : demarrage (ecran Modeles de Prophet Studio)...")
        api.call("POST", "/api/runtime/start")
        time.sleep(1.0)
    t0 = time.time()
    while rt["state"] not in ("ready", "degraded"):
        if rt["state"] == "error" or (rt["state"] == "stopped" and not start) or time.time() - t0 > timeout:
            raise SystemExit(f"les modeles ne sont pas prets (etat {rt['state']} : {rt.get('message') or '-'}) : demarrez-les dans "
                             "Prophet Studio (ecran Modeles) ou relancez avec --demarrer")
        time.sleep(1.0)
        state = api.call("GET", "/api/state")
        rt = state["runtime"]
    return state


def studio_summary(state: dict, info: dict) -> dict:
    rt, plan = state["runtime"], state["runtime"].get("plan") or {}
    sp = lambda k: {x: (plan.get(k) or {}).get(x) for x in ("model_id", "device", "ctx", "np", "kv_type", "ngl")} if plan.get(k) else None  # noqa: E731
    return {"url": info["url"], "core_json": info.get("core_json"), "version": state.get("version"), "demo": state.get("demo"),
            "runtime_state": rt["state"], "mono": rt.get("mono"), "plan": plan.get("title"), "priority": plan.get("priority"),
            "s2": sp("s2"), "s1": sp("s1"), "gpu": ((state.get("hardware") or {}).get("primary_gpu") or {}).get("name")}


def run_turn_studio(api: StudioApi, request: str, workspace: Path, agent: bool, effort: str = "auto", timeout: float = 900,
                    keep: bool = False) -> dict:
    """Un tour par l'API du coeur (session neuve, espace temporaire, permission_mode auto), lu dans la transcription sauvegardee."""
    sid = api.call("POST", "/api/sessions", {"workspace": str(workspace), "title": "mesure du duo"})["id"]
    before = dp.list_files(workspace)
    t: dict = {"mode": "studio", "request": request, "effort": effort, "session": sid}
    t0 = time.perf_counter()
    try:
        api.call("POST", f"/api/sessions/{sid}/turn", {"text": request, "permission_mode": "auto", "effort": effort})
        while True:
            s = api.call("GET", f"/api/sessions/{sid}")
            item = next((i for i in reversed(s.get("transcript") or []) if i.get("kind") == "assistant"), None)
            if not s.get("running") and item and item.get("status") != "running":
                break
            if time.perf_counter() - t0 > timeout and "error" not in t:
                api.call("POST", f"/api/sessions/{sid}/cancel")
                t["error"] = f"delai depasse ({timeout:.0f} s) : tour annule"
            elif time.perf_counter() - t0 > timeout + 120:
                break   # l'annulation n'a pas abouti : on rend ce qui existe
            time.sleep(0.2)
    except Exception as e:
        t.update(error=f"{type(e).__name__}: {str(e)[:300]}", wall_ms=round((time.perf_counter() - t0) * 1000, 1))
        return dp.finish_turn(t, None, agent)
    t["wall_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    st, s1 = item.get("stats") or {}, item.get("s1") or {}
    t.update(path=item.get("path"), rerouted=(s1.get("pre") or {}).get("rerouted"), latency_ms=item.get("latency_ms"), ttft_ms=None,
             s1_calls=st.get("s1_calls"), s1_ms=st.get("s1_ms"), s2_calls=st.get("llm_calls"), s2_tokens=st.get("tokens"),
             s2_tok_s=st.get("tok_s"), s2_prompt_ms=st.get("prompt_ms"), ctx_tokens=st.get("ctx_tokens"),
             tools=[b.get("name") for b in item.get("blocks") or [] if b.get("type") == "tool"],
             files_created=sorted(dp.list_files(workspace) - before), verification=item.get("verification"),
             stopped_by=item.get("stopped_by"), response=" ".join((item.get("response") or "").split())[:300],
             error=t.get("error") or item.get("error"))
    if not keep:
        try:
            api.call("DELETE", f"/api/sessions/{sid}")
        except Exception:
            pass
    return dp.finish_turn(t, s1, agent)


# ---- Markdown pour eval/SCOREBOARD.md ------------------------------------------------------------------------------------
def fr(x, nd: int = 0) -> str:
    """Nombre a la francaise (virgule decimale, espace des milliers) ; '-' si absent."""
    if x is None:
        return "-"
    return f"{float(x):,.{nd}f}".replace(",", " ").replace(".", ",")


def conditions(res: dict) -> str:
    sv, st, vr = res.get("servers") or {}, res.get("studio") or {}, res.get("vram") or {}
    s1, s2 = sv.get("s1") or {}, sv.get("s2") or {}
    gpu = vr.get("gpu") or st.get("gpu") or "sans GPU NVIDIA visible"
    parts = [res["started"][:10], gpu + (f" (pilote {vr['driver']})" if vr.get("driver") else ""),
             f"S2 {s2.get('model') or '?'} ctx {s2.get('n_ctx') or '?'}",
             "mode mono (S1 = S2)" if sv.get("mono") else f"S1 {s1.get('model') or '?'} ({s1.get('slots') or '?'} slot(s), ctx {s1.get('n_ctx') or '?'})"]
    if st:
        parts.append(f"Studio {st.get('plan') or ''}".strip() + (" DEMO (faux serveurs)" if st.get("demo") else ""))
    if res.get("label"):
        parts.append(res["label"])
    # tableau en ASCII, comme le reste de SCOREBOARD.md (point median du titre du plan, accents eventuels)
    return unicodedata.normalize("NFKD", ", ".join(parts).replace("\u00b7", "-")).encode("ascii", "ignore").decode()


def markdown_rows(res: dict) -> list[str]:
    lat, rd = res.get("s1_latency") or {}, res.get("s1_readout") or {}
    s2, th, turns, vr = res.get("s2") or {}, res.get("thinking") or {}, res.get("turns") or {}, res.get("vram") or {}
    g = lambda k, series, q="p50": ((lat.get(k) or {}).get(series) or {}).get(q)   # noqa: E731

    def masses(mode: str) -> str:
        return " ".join(fr(((rd.get(k) or {}).get(mode) or {}).get("label_mass"), 2) for k in ("noul", "choice", "score"))

    def turn(k: str) -> str:
        t = turns.get(k) or {}
        if not t or (t.get("error") and t.get("wall_ms") is None):
            return "-" if not t else f"echec : {t['error'][:60]}"
        n = lambda x: "-" if x is None else x   # noqa: E731
        s = f"{fr((t.get('wall_ms') or 0) / 1000, 1)} s, {n(t.get('s1_calls'))} / {fr(t.get('s1_ms'))} ms, {n(t.get('s2_calls'))} (voie {t.get('path') or '?'}"
        if k == "agent":
            s += ", fichier cree" if t.get("files_created") else ", aucun fichier"
        return s + ("" if t.get("ok") else ", invariants NON") + ")"
    th0, th512 = th.get("0") or {}, th.get("512") or {}
    think = "-" if not th else f"{th0.get('reasoning_tokens', '-')} / {th512.get('reasoning_tokens', '-')} ({'oui' if th.get('honored') else 'NON'})"
    vram = "-" if not vr.get("available") else f"{fr(vr['before_mib'])} / {fr(vr['peak_mib'])} / {fr(vr['after_mib'])} (sur {fr(vr['total_mib'])})"
    rows = [("S2 generation (tok/s)", "47-59", fr((s2.get("generation") or {}).get("tok_s"), 1)),
            ("S2 prefill (tok/s, invite de ~1 500 tokens)", "a mesurer", fr((s2.get("prefill") or {}).get("tok_s"))),
            ("S2 premier token (ms) : invite courte / ~1 500 tokens", "a mesurer",
             f"{fr((s2.get('generation') or {}).get('ttft_ms'))} / {fr((s2.get('prefill') or {}).get('ttft_ms'))}"),
            ("S2 reflexion : tokens avec budget 0 / budget 512 (respecte ?)", "0 / <= 512", think),
            ("S1 pre-tour (6 questions) a froid p50 / p95 (ms), etat neuf ~2 k tokens", "a mesurer",
             f"{fr(g('pre_tour', 'cold'))} / {fr(g('pre_tour', 'cold', 'p95'))}"),
            ("S1 pre-tour a chaud p50 / p95 (ms), etat deja lu", "100-350", f"{fr(g('pre_tour', 'warm'))} / {fr(g('pre_tour', 'warm', 'p95'))}"),
            ("S1 a chaud p50 (ms) : noul / choice / score", "a mesurer", " / ".join(fr(g(k, "warm")) for k in ("noul", "choice", "score"))),
            ("S1 selection des outils (un noul par outil) a froid / a chaud p50 (ms)", "a mesurer", f"{fr(g('outils', 'cold'))} / {fr(g('outils', 'warm'))}"),
            ("S1 garde-fou (3 questions) a froid / a chaud p50 (ms)", "a mesurer", f"{fr(g('garde_fou', 'cold'))} / {fr(g('garde_fou', 'warm'))}"),
            ("S1 lecture : masse sur les etiquettes avec grammaire / sans (noul choice score)", "1,00 / a mesurer",
             "-" if not rd else f"{masses('grammar')} / {masses('free')}"),
            ("Tour Prophet, voie directe : duree, appels S1 / ms S1, appels S2", "a mesurer", turn("direct")),
            ("Tour Prophet, voie agent (creation de fichier) : duree, appels S1 / ms S1, appels S2", "a mesurer", turn("agent")),
            ("VRAM utilisee avant / pic / apres (Mio)", "a mesurer", vram)]
    cond = conditions(res)
    return TABLE_HEAD + [f"| {a} | {b} | {c} | {cond if i == 0 else 'idem'} |" for i, (a, b, c) in enumerate(rows)]


# ---- mesure ---------------------------------------------------------------------------------------------------------------
def machine_info() -> dict:
    return {"os": platform.platform(), "machine": platform.machine(), "cpu": platform.processor() or None, "cpu_count": os.cpu_count(),
            "python": platform.python_version()}


def measure(args, log=print) -> tuple[dict, int]:
    now = dt.datetime.now()
    res: dict = {"tool": "eval/measure_duo.py", "version": 1, "started": now.isoformat(timespec="seconds"), "label": args.label,
                 "options": {k: v for k, v in vars(args).items() if k != "token"}, "machine": machine_info(), "errors": {}}
    api = None
    if args.s1 or args.s2:
        if not (args.s1 and args.s2):
            raise SystemExit("--s1 et --s2 vont ensemble (meme URL pour les deux en mode mono)")
        s1_url, s2_url = args.s1.rstrip("/"), args.s2.rstrip("/")
        res["mode"] = "servers"
    else:
        info = find_core(args.data_dir) if args.studio in (None, "auto") else None
        url = info["url"] if info else (None if args.studio in (None, "auto") else args.studio)
        if not url:
            raise SystemExit(f"Prophet Studio ne tourne pas (aucun core.json valide dans {Path(args.data_dir) if args.data_dir else default_data_dir()}) :"
                             " lancez Prophet Studio, ou mesurez deux llama-server avec --s1 URL --s2 URL")
        if info is None and args.studio and not args.token:
            info = find_core(args.data_dir)
        token = args.token or (info or {}).get("token") or os.environ.get("PROPHET_TOKEN")
        if not token:
            raise SystemExit("jeton de Prophet Studio inconnu : --token (ou PROPHET_TOKEN), ou laissez --studio seul pour lire core.json")
        api = StudioApi(url, token)
        state = studio_ready(api, args.demarrer, log=log)
        res["mode"], res["studio"] = "studio", studio_summary(state, {**(info or {}), "url": url})
        s1_url, s2_url = state["runtime"]["s1_url"].rstrip("/"), state["runtime"]["s2_url"].rstrip("/")
    res["servers"] = {"s1": dp.server_info(s1_url), "s2": dp.server_info(s2_url), "mono": s1_url == s2_url}
    for k in ("s1", "s2"):
        if not res["servers"][k].get("health"):
            raise SystemExit(f"{k.upper()} ne repond pas sur {res['servers'][k]['url']} : {res['servers'][k].get('error', '/health en echec')}")
    sv = res["servers"]
    log(f"S1 : {sv['s1'].get('model')} ({s1_url}, {sv['s1'].get('slots')} slot(s), ctx {sv['s1'].get('n_ctx')})"
        + (" -- mode mono : S1 = S2" if sv["mono"] else ""))
    log(f"S2 : {sv['s2'].get('model')} ({s2_url}, ctx {sv['s2'].get('n_ctx')})")
    n_cold, n_warm, n_s2 = (2, 1, 1) if args.rapide else (args.n_froid, args.n_chaud, args.n_s2)
    rng = random.Random(args.seed if args.seed is not None else time.time_ns())
    vram = dp.VramSampler().start()

    def section(name: str, title: str, fn) -> None:
        log(f"-- {title}")
        t0 = time.perf_counter()
        try:
            res[name] = fn()
            bad = {k: v["error"] for k, v in res[name].items() if isinstance(v, dict) and v.get("error")}   # echec d'un type, d'un tour
            if bad:
                res["errors"][name] = bad
        except Exception as e:
            res["errors"][name] = f"{type(e).__name__}: {str(e)[:400]}"
            log(f"  ECHEC : {res['errors'][name]}")
        res.setdefault("durations_s", {})[name] = round(time.perf_counter() - t0, 1)
    from jev_clone.backend_llamacpp import LlamaCppBackend
    from jev_clone.engine import SystemOneEngine
    s1 = SystemOneEngine(LlamaCppBackend(s1_url, max_workers=4, timeout=300))
    s2 = LlamaCppBackend(s2_url, max_workers=1, timeout=900)
    section("s1_latency", f"S1 : latence par type de question ({n_cold} etats neufs, {n_warm} relecture(s) chacun)",
            lambda: dp.s1_latency(s1, rng, n_cold, n_warm, log=log))
    section("s1_readout", "S1 : lecture des etiquettes (grammaire / libre)", lambda: dp.s1_readout(s1, rng))
    for k, v in (res.get("s1_readout") or {}).items():
        log(f"  {k:<6} " + (f"ECHEC {v['error']}" if "error" in v else f"grammaire : masse {v['grammar']['label_mass']} ({v['grammar']['labels_found']} etiquettes) ; "
                                                                        f"libre : masse {v['free']['label_mass']} ({v['free']['labels_found']}, top1 {v['free']['top1']!r})"))
    section("s2", f"S2 : generation et prefill ({n_s2} essai(s))", lambda: dp.s2_speed(s2, rng, n_s2, log=log))
    section("thinking", "S2 : budget de reflexion (0 puis 512)", lambda: dp.s2_thinking(s2, log=log))
    if not args.sans_tours:
        def turns() -> dict:
            out = {}
            for k, request, agent in (("direct", dp.DIRECT_QUESTION, False), ("agent", dp.AGENT_TASK, True)):
                ws = dp.temp_workspace()
                try:
                    out[k] = (run_turn_studio(api, request, ws, agent, timeout=args.timeout_tour, keep=args.garder) if api else
                              dp.run_turn_local(s1_url, s2_url, request, ws, calibration=args.calibration, timeout=args.timeout_tour, agent=agent))
                finally:
                    if not args.garder:
                        dp.remove_tree(ws)
                t = out[k]
                log(f"  tour {k} : voie {t.get('path')}, {fr((t.get('wall_ms') or 0) / 1000, 1)} s, S1 {t.get('s1_calls')} appels / "
                    f"{t.get('s1_ms')} ms, S2 {t.get('s2_calls')} appels, outils {t.get('tools')}, fichiers {t.get('files_created')}"
                    + ("" if t["ok"] else f" -- invariants en echec : {[c for c, ok in t['checks'].items() if not ok]} {t.get('error') or ''}"))
            return out
        section("turns", "Tours Prophet complets (espace temporaire, permission_mode auto)", turns)
    res["vram"] = vram.stop()
    res["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    res["markdown"] = markdown_rows(res)
    out_dir = Path(args.out) if args.out else RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"duo-{now.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    res["path"] = str(path)
    return res, 1 if res["errors"] or any(not t.get("ok") for t in (res.get("turns") or {}).values()) else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.measure_duo", description="Mesure du duo reel (S1 classifieur + S2 raisonnement) : "
                                 "JSON dans eval/results/ et lignes a coller dans eval/SCOREBOARD.md")
    ap.add_argument("--studio", nargs="?", const="auto", default=None, metavar="URL",
                    help="Prophet Studio en marche (defaut : URL et jeton lus dans core.json du dossier de donnees)")
    ap.add_argument("--token", help="jeton du coeur (defaut : core.json, puis PROPHET_TOKEN)")
    ap.add_argument("--data-dir", help="dossier de donnees de Studio (defaut : %%LOCALAPPDATA%%\\ProphetStudio sous Windows)")
    ap.add_argument("--demarrer", action="store_true", help="Studio : demarrer les modeles s'ils sont arretes, puis attendre qu'ils soient prets")
    ap.add_argument("--s1", metavar="URL", help="llama-server du classifieur (sans Studio, avec --s2)")
    ap.add_argument("--s2", metavar="URL", help="llama-server du modele de raisonnement (sans Studio, avec --s1)")
    ap.add_argument("--calibration", help="sans Studio : calibration.json du classifieur pour le tour Prophet")
    ap.add_argument("--out", help="dossier des resultats (defaut : eval/results)")
    ap.add_argument("--label", default="", help="texte libre ajoute aux conditions (pilote, reglages...)")
    ap.add_argument("--rapide", action="store_true", help="moins de repetitions (2 etats neufs, 1 essai S2) : ~2 min")
    ap.add_argument("--n-froid", type=int, default=4, help="etats neufs par type de question S1 (defaut 4)")
    ap.add_argument("--n-chaud", type=int, default=2, help="relectures a chaud de chaque etat (defaut 2)")
    ap.add_argument("--n-s2", type=int, default=3, help="essais S2 generation et prefill (defaut 3)")
    ap.add_argument("--sans-tours", action="store_true", help="ne pas lancer les tours Prophet complets")
    ap.add_argument("--timeout-tour", type=float, default=900.0, help="duree maximale d'un tour Prophet, en secondes (defaut 900)")
    ap.add_argument("--garder", action="store_true", help="garder les espaces de travail temporaires et les sessions de mesure")
    ap.add_argument("--seed", type=int, help="graine des etats aleatoires (defaut : horloge)")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")   # console Windows (cp1252) : jamais d'UnicodeEncodeError
    res, code = measure(args, log=lambda m: print(m, flush=True))
    print("\nLignes a coller dans eval/SCOREBOARD.md (section \"Duo reel\") :\n")
    print("\n".join(res["markdown"]))
    bad = {k: [c for c, ok in t["checks"].items() if not ok] for k, t in (res.get("turns") or {}).items() if not t.get("ok")}
    if bad:
        print(f"\nATTENTION invariants de tour en echec : {bad}")
    if res["errors"]:
        print(f"\nATTENTION sections en echec : {', '.join(res['errors'])}")
    print(f"\nResultats : {res['path']}")
    return code


if __name__ == "__main__":
    sys.exit(main())

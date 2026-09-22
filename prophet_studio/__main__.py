"""prophet-studio : lance le coeur (API + interface) et ouvre l'application dans le navigateur.

    prophet-studio                    # interface sur http://127.0.0.1:7878 (ouverte automatiquement)
    prophet-studio --demo             # sans modele : faux serveurs, pour decouvrir l'interface
    prophet-studio --no-browser --port 0 --token XYZ --parent-pid 1234   # utilise par l'application de bureau
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

import requests


def _free_port(preferred: int) -> int:
    for p in ([preferred] if preferred else []) + [0]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return s.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("aucun port libre")


def _existing_instance(info_file: Path) -> dict | None:
    try:
        info = json.loads(info_file.read_text(encoding="utf-8"))
        r = requests.get(f"{info['url']}/api/health", timeout=1.5)
        if r.ok and r.json().get("app") == "prophet-studio":
            return info
    except Exception:
        pass
    return None


def _watch_parent(pid: int, on_exit) -> None:
    def loop():
        try:
            import psutil
        except Exception:
            return
        while True:
            if not psutil.pid_exists(pid):
                on_exit()
                os._exit(0)
            time.sleep(2)
    threading.Thread(target=loop, daemon=True).start()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="prophet-studio", description="Prophet Studio : Bonsai 2 27B + classifieur type Jev, en local")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PROPHET_PORT", "7878")))
    ap.add_argument("--token", default=os.environ.get("PROPHET_TOKEN"))
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--demo", action="store_true", help="faux modeles (decouverte de l'interface, captures)")
    ap.add_argument("--dev", action="store_true", help="autorise le serveur Vite (localhost:5173)")
    ap.add_argument("--parent-pid", type=int, default=0, help="s'arreter quand ce processus disparait (application de bureau)")
    args = ap.parse_args(argv)

    from prophet_studio.config import Paths, data_dir
    root = Path(args.data_dir) if args.data_dir else (data_dir() / "demo" if args.demo else data_dir())
    paths = Paths(root)
    if sys.stdout is None or sys.stderr is None:   # pythonw (raccourci Windows sans console) : journal sur disque
        log = open(paths.logs / "core.log", "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stdout or log
        sys.stderr = sys.stderr or log

    if not args.parent_pid:
        other = _existing_instance(paths.core_info)
        if other:
            print(f"Prophet Studio tourne deja : {other['url']}")
            if not args.no_browser:
                webbrowser.open(other["url"])
            return

    port = _free_port(args.port)
    token = args.token or secrets.token_urlsafe(24)
    url = f"http://127.0.0.1:{port}"
    paths.core_info.write_text(json.dumps({"url": url, "port": port, "token": token, "pid": os.getpid()}), encoding="utf-8")
    if os.name != "nt":
        os.chmod(paths.core_info, 0o600)

    from prophet_studio.server import Studio, build_app
    studio = Studio(paths, token, port, demo=args.demo, dev=args.dev)
    app = build_app(studio)
    if args.parent_pid:
        _watch_parent(args.parent_pid, studio.runtime.stop)

    import atexit
    atexit.register(studio.runtime.stop)
    atexit.register(lambda: paths.core_info.unlink(missing_ok=True))

    print(f"Prophet Studio {'(demo) ' if args.demo else ''}-> {url}", flush=True)
    print(f"PROPHET_READY {json.dumps({'url': url, 'port': port})}", flush=True)
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning", access_log=False)


if __name__ == "__main__":
    sys.exit(main())

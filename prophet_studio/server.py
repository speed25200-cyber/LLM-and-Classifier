"""API de Prophet Studio : REST + WebSocket sur 127.0.0.1, protegee par jeton.

Securite : le coeur peut executer des commandes ; il n'ecoute que sur la boucle locale et exige :
  * un en-tete Host local (parade au DNS rebinding) ;
  * une Origin absente ou autorisee (l'interface elle-meme, l'application Tauri) ;
  * le jeton de lancement (en-tete X-Prophet-Token, Authorization: Bearer, ou ?token= pour les WebSocket).
Le jeton est injecte dans la page servie (lisible seulement par la meme origine) et transmis par Tauri.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse

from prophet_studio import __version__
from prophet_studio import hardware as hwmod
from prophet_studio.catalog import MODELS, VOICE, catalog_dict
from prophet_studio.config import Paths, SettingsStore
from prophet_studio.downloads import Downloader
from prophet_studio.events import Bus
from prophet_studio.installer import Installer
from prophet_studio.planner import make_plan
from prophet_studio.sessions import AgentService, SessionStore
from prophet_studio.supervisor import Runtime
from prophet_studio.voice import VoiceService, VoiceUnavailable, pcm16_to_float, route_utterance

WEB_DIR = Path(__file__).parent / "web"
LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]", "::1", "tauri.localhost"}


class Studio:
    """Etat de l'application : un seul objet, construit au demarrage du coeur."""

    def __init__(self, paths: Paths, token: str, port: int, demo: bool = False, dev: bool = False):
        self.paths, self.token, self.port, self.demo, self.dev = paths, token, port, demo, dev
        self.settings = SettingsStore(paths)
        self.bus = Bus()
        self.hw = hwmod.detect()
        self.downloader = Downloader(self.bus.publish)
        self.installer = Installer(paths, self.settings, self.downloader, self.bus.publish)
        self.runtime = Runtime(paths.logs, self.bus.publish, self.installer.model_path, self.installer.mmproj_path,
                               self.server_cmd, self.installer.installed_models)
        self.sessions = SessionStore(paths.sessions)
        from prophet_studio.calibration import active_s1_model, calibration_file   # un fichier par classifieur, aucun en mono
        self.agent = AgentService(self.sessions, self.bus.publish, urls=lambda: (self.runtime.s1_url, self.runtime.s2.url),
                                  settings=self.settings.get, ctx=lambda: self.runtime.plan.s2.ctx if self.runtime.plan else 8192,
                                  calibration=lambda: calibration_file(paths.runs, active_s1_model(self.runtime)),
                                  desktop_backend=self._demo_desktop if demo else None)
        self.voice = VoiceService(self.installer.voice_files, self.settings.get, threads=max(1, min(4, self.hw.cpu_cores // 2)))
        self.monitor = hwmod.GpuMonitor()
        self.runtime.vram_probe = lambda: (self.monitor.sample() or {}).get("vram_used_mib")
        self.runtime.on_measure = self._record_vram
        self._load_vram_calibration()
        self.last_bench: dict | None = None
        self.started = time.time()
        if demo:
            self._prepare_demo()
        self.installer.on_change = self._autostart_after_install   # apres la preparation : pas de double demarrage

    @staticmethod
    def _demo_desktop():
        from jev_clone.desktop_use import SimulatedDesktop
        return SimulatedDesktop()

    # ---- demo : faux serveurs lances par le vrai superviseur ------------------------------------------------------------
    def _prepare_demo(self) -> None:
        from prophet_studio.config import Settings
        if self.settings.get().workspace == Settings().workspace:   # la demo n'ecrit jamais dans le vrai ~/Prophet
            self.settings.update({"workspace": str(self.paths.root / "workspace")})
        # les modeles que le planificateur choisit pour CETTE machine (Bonsai 8B sans GPU...) : la demo demarre partout
        ids = ["bonsai2-27b-ptq1", "ternary-1.7b"]
        try:
            p = self.plan()
            ids += [sp.model_id for sp in (p.s2, p.s1) if sp is not None and sp.model_id not in ids and sp.model_id in MODELS]
        except Exception:
            pass
        for mid in ids:
            f = self.paths.models / mid / f"{mid}-demo.gguf"
            f.parent.mkdir(parents=True, exist_ok=True)
            f.touch()
            self.installer.registry["models"][mid] = {"main": str(f), "role": MODELS[mid].role, "installed_at": time.time(), "demo": True}
        self.installer.registry["runtime"] = {"tag": "demo", "backend": "demo", "server": sys.executable, "cuda": None, "version": "faux llama-server"}
        self.installer._save()

    # ---- calibration VRAM : mesures reelles -> planificateur -------------------------------------------------------
    def _load_vram_calibration(self) -> None:
        from prophet_studio import planner
        f = self.paths.runs / "vram_calibration.json"
        if f.exists():
            try:
                planner.MEASURED.update({k: float(v["overhead_mib"]) for k, v in json.loads(f.read_text(encoding="utf-8")).items()})
            except Exception:
                pass

    def _record_vram(self, role: str, sp, used_mib: float) -> None:
        from prophet_studio import planner
        from prophet_studio.catalog import MODELS
        m = MODELS.get(sp.model_id)
        if m is None:
            return
        ctx = sp.ctx
        mm = m.mmproj_gib * 1024 if role == "s2" and sp.mmproj == "gpu" else 0.0
        over = planner.calibrated_overhead(m.id, m.kv_kib_f16, m.weights_gib, ctx, sp.kv_type, mm, used_mib)
        planner.MEASURED[m.id] = over
        f = self.paths.runs / "vram_calibration.json"
        data = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        data[m.id] = {"overhead_mib": round(over, 1), "used_mib": round(used_mib, 1), "ctx": ctx, "kv_type": sp.kv_type,
                      "gpu": self.hw.gpu.name if self.hw.gpu else "", "ts": time.time()}
        f.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.bus.publish({"type": "runtime.calibrated", "model": m.id, "used_mib": round(used_mib), "overhead_mib": round(over)})

    def _autostart_after_install(self) -> None:
        """Des que le runtime et Bonsai sont la, les modeles demarrent seuls : installer = pouvoir utiliser."""
        if (self.settings.get().autostart_models and self.runtime.state in ("stopped", "error")
                and not self.recommended_missing_core() and not self.downloader_busy()):
            self.start_runtime()

    def downloader_busy(self) -> bool:
        return any(j.status in ("queued", "running", "verifying", "extracting") and j.group in ("runtime", self.plan().s2.model_id)
                   for j in self.downloader.jobs.values())

    def server_cmd(self) -> list[str] | None:
        if self.demo:
            return [sys.executable, "-m", "prophet_studio.demo.fake_llama", "--tps", os.environ.get("FAKE_LLAMA_TPS", "55")]
        b = self.installer.server_binary()
        return [str(b)] if b else None

    # ---- plan ---------------------------------------------------------------------------------------------------------
    def plan(self, priority: str | None = None, s2: str | None = None, s1: str | None = None, ctx: int | None = None):
        st = self.settings.get()
        other = None
        g = self.hw.gpu
        if g is not None and self.runtime.state not in ("stopped", "error"):
            sample = self.monitor.sample(self.runtime.pids()) or {}
            if sample:
                other = max(0, sample["vram_used_mib"] - sample.get("vram_ours_mib", 0))
        return make_plan(self.hw, priority or st.priority, s2 or st.s2_model, s1 or st.s1_model, ctx if ctx is not None else st.ctx_override,
                         other_used_mib=other)

    def start_runtime(self) -> None:
        st = self.settings.get()
        plan = self.plan()
        threading.Thread(target=self.runtime.start, args=(plan, (st.s2_port, st.s1_port)), daemon=True, name="runtime-start").start()

    def recommended(self) -> list[dict]:
        ids = self.installer.recommended(self.plan())
        out = []
        for i in ids:
            if i == "runtime":
                out.append({"id": "runtime", "label": "Runtime llama.cpp (fork PrismML)", "size_gb": 0.25})
            elif i in MODELS:
                m = MODELS[i]
                out.append({"id": i, "label": m.label, "size_gb": m.size_gb + (m.mmproj_gib * 1.07 if m.mmproj_pattern else 0), "role": m.role})
            elif i in VOICE:
                out.append({"id": i, "label": VOICE[i].label, "size_gb": VOICE[i].size_mb / 1000, "role": "voice"})
        return out

    def recommended_missing_core(self) -> bool:
        """Vrai s'il manque le runtime ou le modele System Two (sans classifieur -> mode mono, ce n'est pas bloquant)."""
        return self.server_cmd() is None or self.plan().s2.model_id not in self.installer.installed_models()

    def state(self) -> dict:
        plan = self.plan()
        from jev_clone.desktop_use import desktop_available
        return {"version": __version__, "demo": self.demo, "hardware": self.hw.to_dict(), "settings": self.settings.get().model_dump(),
                "desktop": dict(zip(("available", "reason"), (True, "bureau simule (demo)") if self.demo else desktop_available())),
                "plan": plan.to_dict(), "runtime": self.runtime.public(), "installed": self.installer.status(),
                "downloads": self.downloader.snapshot(), "voice": self.voice.status(), "sessions": self.sessions.list(),
                "permissions": self.agent.pending_permissions(), "running": list(self.agent.running), "recommended": self.recommended(),
                "catalog": catalog_dict(), "bench": self.last_bench, "data_dir": str(self.paths.root)}

    def metrics(self) -> dict:
        m = self.monitor.sample(self.runtime.pids())
        if m is None and self.demo and self.hw.gpu:
            g, p = self.hw.gpu, self.runtime.plan
            used = g.vram_used_mib + (sum(v for k, v in p.budget.items() if k not in ("total", "other", "free", "reserve")) if p and self.runtime.state == "ready" else 0)
            busy = bool(self.agent.running)
            m = {"util": 93 if busy else 3, "vram_used_mib": round(used), "vram_total_mib": g.vram_total_mib, "temp_c": 61 if busy else 44,
                 "power_w": 118.0 if busy else 14.0, "vram_ours_mib": round(used - g.vram_used_mib), "simulated": True}
        return {"type": "metrics", "gpu": m, "runtime": self.runtime.state, "running": list(self.agent.running)}

    # ---- banc de mesure -------------------------------------------------------------------------------------------------
    def bench(self) -> dict:
        s1, s2 = self.agent.engines()
        req = {"state": "Customer: my payouts failed three times this week and nobody answered my emails.",
               "questions": {"team": {"type": "choice", "instructions": "Which team should handle this?", "criteria": ["payments", "account", "other"]},
                             "escalate": {"type": "noul", "instructions": "Should this be escalated?"},
                             "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["can wait", "today", "blocked now"]}}}
        s1.answer(req)
        lat = []
        for _ in range(10):
            t0 = time.perf_counter(); s1.answer(req); lat.append((time.perf_counter() - t0) * 1000)
        lat.sort()
        t0 = time.perf_counter()
        r = s2.chat([{"role": "user", "content": "Ecris un paragraphe de 120 mots sur l'art du bonsai."}], max_tokens=160, thinking_budget=0, temperature=0.7)
        tm = r.get("timings") or {}
        out = {"ts": time.time(), "s1_p50_ms": round(lat[len(lat) // 2], 1), "s1_p95_ms": round(lat[-1], 1),
               "s2_tok_s": round(float(tm.get("predicted_per_second") or 0), 1), "s2_prefill_tok_s": round(float(tm.get("prompt_per_second") or 0), 1),
               "s2_total_s": round(time.perf_counter() - t0, 2), "plan": self.runtime.plan.title if self.runtime.plan else None,
               "gpu": self.hw.gpu.name if self.hw.gpu else "CPU", "demo": self.demo}
        self.last_bench = out
        with open(self.paths.runs / "bench.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
        return out


def build_app(studio: Studio) -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(_app):
        loop = asyncio.get_running_loop()
        studio.bus.bind(loop)
        metrics = loop.create_task(_metrics_loop())
        st = studio.settings.get()
        if st.autostart_models and not studio.recommended_missing_core():
            studio.start_runtime()
        if st.voice.enabled:
            studio.voice.preload()
        try:
            yield
        finally:
            metrics.cancel()
            studio.runtime.stop()

    app = FastAPI(title="Prophet Studio", version=__version__, docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    origins = {f"http://127.0.0.1:{studio.port}", f"http://localhost:{studio.port}", "tauri://localhost", "http://tauri.localhost",
               "https://tauri.localhost"}
    if studio.dev:
        origins |= {"http://localhost:5173", "http://127.0.0.1:5173"}

    def token_ok(tok: str | None) -> bool:
        return bool(tok) and hmac.compare_digest(tok, studio.token)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        host = (request.headers.get("host") or "").rsplit(":", 1)[0] if not (request.headers.get("host") or "").startswith("[") else "[::1]"
        if host not in LOCAL_HOSTS:
            return JSONResponse({"detail": "hote refuse"}, status_code=403)
        origin = request.headers.get("origin")
        if origin and origin not in origins:
            return JSONResponse({"detail": "origine refusee"}, status_code=403)
        cors = {"Access-Control-Allow-Origin": origin, "Vary": "Origin", "Access-Control-Allow-Headers": "content-type, x-prophet-token, authorization",
                "Access-Control-Allow-Methods": "GET, POST, PUT, PATCH, DELETE, OPTIONS", "Access-Control-Max-Age": "600"} if origin else {}
        if request.method == "OPTIONS":
            return Response(status_code=204, headers=cors)
        path = request.url.path
        if (path.startswith("/api/") and path != "/api/health") or path.startswith("/v1/"):
            auth = request.headers.get("authorization", "")
            tok = request.headers.get("x-prophet-token") or (auth[7:] if auth.lower().startswith("bearer ") else None) or request.query_params.get("token")
            if not token_ok(tok):
                return JSONResponse({"detail": "jeton manquant ou invalide"}, status_code=401, headers=cors)
        resp = await call_next(request)
        for k, v in cors.items():
            resp.headers[k] = v
        return resp

    async def _metrics_loop():
        while True:
            await asyncio.sleep(1.5)
            if studio.bus.clients:
                try:
                    studio.bus.publish(await asyncio.to_thread(studio.metrics))
                except Exception:
                    pass

    # ---- etat, reglages, materiel ------------------------------------------------------------------------------------------
    @app.get("/api/health")
    def health():
        return {"ok": True, "version": __version__, "app": "prophet-studio"}

    @app.get("/api/state")
    def state():
        return studio.state()

    @app.post("/api/hardware/refresh")
    def refresh_hw():
        studio.hw = hwmod.detect()
        return studio.hw.to_dict()

    @app.get("/api/plan")
    def plan(priority: str | None = None, s2: str | None = None, s1: str | None = None, ctx: int | None = None):
        return studio.plan(priority, s2, s1, ctx).to_dict()

    @app.put("/api/settings")
    async def put_settings(request: Request):
        patch = await request.json()
        try:
            s = studio.settings.update(patch)
        except Exception as e:
            raise HTTPException(422, str(e)[:300])
        studio.bus.publish({"type": "settings.changed", "settings": s.model_dump(), "plan": studio.plan().to_dict()})
        return s.model_dump()

    # ---- installation ----------------------------------------------------------------------------------------------------
    @app.post("/api/install")
    async def install(request: Request):
        body = await request.json()
        items = body.get("items") or [r["id"] for r in studio.recommended()]
        jobs, errors = [], {}
        for it in items:
            try:
                if it == "runtime":
                    jobs += await asyncio.to_thread(studio.installer.install_runtime, studio.hw)
                elif it in MODELS:
                    jobs += await asyncio.to_thread(studio.installer.install_model, it, body.get("mmproj", True))
                elif it in VOICE:
                    jobs += await asyncio.to_thread(studio.installer.install_voice, it)
                else:
                    errors[it] = "inconnu"
            except Exception as e:
                errors[it] = str(e)[:300]
        return {"jobs": jobs, "errors": errors}

    @app.get("/api/runtime/asset")
    def runtime_asset():
        sel = studio.installer.plan_runtime(studio.hw)
        return {**sel, "main": sel["main"]["name"], "extra": [a["name"] for a in sel["extra"]]}

    @app.post("/api/downloads/cancel")
    async def cancel_dl(request: Request):
        b = await request.json()
        return {"cancelled": studio.downloader.cancel(b.get("id"), b.get("group"))}

    @app.delete("/api/installed/{item_id}")
    def remove(item_id: str):
        if not studio.installer.remove(item_id):
            raise HTTPException(404, "introuvable")
        return {"ok": True}

    @app.post("/api/import")
    async def import_gguf(request: Request):
        b = await request.json()
        try:
            return {"id": studio.installer.import_gguf(b["path"], b.get("role", "s1"), b.get("label", ""))}
        except Exception as e:
            raise HTTPException(400, str(e)[:300])

    # ---- calibration du classifieur (System One) : un fichier par modele, applique au seul S1 en marche ----------------------
    def _calibration_changed() -> None:
        studio.agent.invalidate_engines()
        studio.bus.publish({"type": "install.changed", "installed": studio.installer.status()})

    @app.post("/api/calibrate")
    async def calibrate_s1(request: Request):
        """Lit les graines etiquetees livrees avec le S1 en marche (T = 1), ajuste temperature + seuils, ecrit son fichier."""
        from prophet_studio import calibration as s1cal
        b = await request.json() if int(request.headers.get("content-length") or 0) else {}
        b = b if isinstance(b, dict) else {}
        mid = s1cal.active_s1_model(studio.runtime)
        if studio.runtime.state not in ("ready", "degraded") or mid is None:
            raise HTTPException(409, "aucun classifieur (System One) en marche a calibrer : modeles arretes ou mode mono")
        target = float(b.get("target_precision") or 0.95)
        if not 0.5 <= target < 1.0:
            raise HTTPException(422, "target_precision doit etre entre 0,5 et 1")
        try:
            cal = await asyncio.to_thread(s1cal.run, studio.runtime.s1_url, mid, studio.bus.publish, target)
        except s1cal.Busy as e:
            raise HTTPException(409, str(e))
        except Exception as e:
            raise HTTPException(502, f"calibration impossible : {type(e).__name__}: {str(e)[:300]}")
        if s1cal.active_s1_model(studio.runtime) != mid:
            raise HTTPException(409, "le classifieur a change pendant la calibration : relancez-la")
        out = s1cal.store(studio.paths.runs, mid, cal)
        studio.bus.publish({"type": "s1.calibration", "status": "done", "model": mid, "done": out["n"], "total": out["n"]})
        _calibration_changed()
        return out

    @app.post("/api/calibration/import")
    async def import_calibration(request: Request):
        """calibration.json (notebook, autre machine) -> fichier du modele choisi (par defaut le classifieur en marche)."""
        from prophet_studio import calibration as s1cal
        b = await request.json()
        if not isinstance(b, dict):
            raise HTTPException(422, "objet {path | data, model_id} attendu")
        mid = b.get("model_id") or s1cal.active_s1_model(studio.runtime)
        if not isinstance(mid, str) or (mid not in MODELS and mid not in studio.installer.registry["models"]):
            raise HTTPException(404, f"modele inconnu : {mid}")
        try:
            out = s1cal.import_calibration(studio.paths.runs, mid, b.get("data"), b.get("path"), bool(b.get("force")))
        except Exception as e:
            raise HTTPException(400, str(e)[:300])
        _calibration_changed()
        return out

    # ---- runtime ------------------------------------------------------------------------------------------------------------
    @app.post("/api/runtime/start")
    def rt_start():
        studio.start_runtime()
        return {"ok": True}

    @app.post("/api/runtime/stop")
    def rt_stop():
        studio.runtime.stop()
        return {"ok": True}

    @app.get("/api/runtime/logs/{name}")
    def rt_logs(name: str, n: int = 200):
        if name not in ("s1", "s2"):
            raise HTTPException(404)
        return {"lines": studio.runtime.logs(name, n)}

    @app.post("/api/bench")
    async def bench():
        if studio.runtime.state not in ("ready", "degraded"):
            raise HTTPException(409, "les modeles ne sont pas demarres")
        try:
            return await asyncio.to_thread(studio.bench)
        except Exception as e:
            raise HTTPException(500, str(e)[:300])

    # ---- sessions et agent -------------------------------------------------------------------------------------------------
    @app.get("/api/sessions")
    def sessions():
        return studio.sessions.list()

    @app.post("/api/sessions")
    async def new_session(request: Request):
        b = await request.json() if int(request.headers.get("content-length") or 0) else {}
        ws = b.get("workspace") or studio.settings.get().workspace
        Path(ws).expanduser().mkdir(parents=True, exist_ok=True)
        return studio.sessions.create(str(Path(ws).expanduser()), b.get("title", ""))

    def _load(sid: str) -> dict:
        try:
            return studio.sessions.load(sid)
        except (FileNotFoundError, ValueError):
            raise HTTPException(404, "session introuvable")

    @app.get("/api/sessions/{sid}")
    def get_session(sid: str):
        s = _load(sid)
        s["running"] = sid in studio.agent.running
        return s

    @app.patch("/api/sessions/{sid}")
    async def patch_session(sid: str, request: Request):
        s, b = _load(sid), await request.json()
        for k in ("title", "workspace"):
            if k in b:
                s[k] = str(b[k])[:300]
        studio.sessions.save(s)
        return {k: s[k] for k in ("id", "title", "workspace")}

    @app.delete("/api/sessions/{sid}")
    def del_session(sid: str):
        studio.agent.cancel(sid)
        studio.sessions.delete(sid)
        return {"ok": True}

    @app.post("/api/sessions/{sid}/turn")
    async def turn(sid: str, request: Request):
        _load(sid)
        b = await request.json()
        text = str(b.get("text", "")).strip()
        if not text:
            raise HTTPException(422, "message vide")
        if studio.runtime.state not in ("ready", "degraded"):
            raise HTTPException(409, "les modeles ne sont pas demarres")
        try:
            tid = studio.agent.submit(sid, text, bool(b.get("plan_mode")), b.get("permission_mode"), b.get("effort"))
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        return {"turn_id": tid}

    @app.post("/api/sessions/{sid}/cancel")
    def cancel(sid: str):
        return {"cancelled": studio.agent.cancel(sid)}

    @app.post("/api/permissions/{pid}")
    async def permission(pid: str, request: Request):
        b = await request.json()
        if not studio.agent.respond(pid, bool(b.get("allow")), bool(b.get("remember"))):
            raise HTTPException(404, "demande expiree")
        return {"ok": True}

    # autorisations memorisees d'une session ("outil:classe de risque", ou "outil" pour une action non jugee) : liste et revocation
    @app.get("/api/sessions/{sid}/always_allow")
    def always_allow(sid: str):
        _load(sid)
        return {"always_allow": studio.agent.always_allow(sid)}

    @app.delete("/api/sessions/{sid}/always_allow")
    def revoke_always_allow(sid: str, grant: str | None = None):
        _load(sid)
        return {"always_allow": studio.agent.revoke(sid, grant)}   # sans grant : tout revoquer

    def _ws_root(session: str | None) -> Path:
        root = studio.settings.get().workspace
        if session:
            root = _load(session).get("workspace") or root
        return Path(root).expanduser().resolve()

    @app.get("/api/files")
    def files(session: str | None = None, max_entries: int = 2000):
        from jev_clone.prophet import Workspace
        root = _ws_root(session)
        if not root.exists():
            return {"root": str(root), "entries": []}
        return {"root": str(root), "entries": Workspace(root).listing(".", max_entries)["entries"]}

    @app.get("/api/file")
    def file(path: str, session: str | None = None):
        from jev_clone.prophet import Workspace
        try:
            r = Workspace(_ws_root(session)).read(path, 400_000)
        except PermissionError as e:
            raise HTTPException(403, str(e))
        if not r.get("ok"):
            raise HTTPException(404, r.get("error"))
        return r

    @app.post("/api/open")
    async def open_path(request: Request):
        from jev_clone.prophet import Workspace
        b = await request.json()
        try:
            p = Workspace(_ws_root(b.get("session"))).resolve(b.get("path", "."))
        except PermissionError as e:
            raise HTTPException(403, str(e))
        if not p.exists():
            raise HTTPException(404)
        if sys.platform == "win32":
            os.startfile(str(p))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(p)])
        else:
            subprocess.Popen(["xdg-open", str(p)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return {"ok": True}

    # ---- voix ---------------------------------------------------------------------------------------------------------------
    def s1_or_none():
        if studio.runtime.state in ("ready", "degraded"):
            return studio.agent.engines()[0]
        return None

    @app.get("/api/voice/status")
    def voice_status():
        return studio.voice.status()

    @app.post("/api/voice/route")
    async def voice_route(request: Request):
        b = await request.json()
        v = studio.settings.get().voice
        r = await asyncio.to_thread(route_utterance, str(b.get("text", "")), s1_or_none(), bool(b.get("require_wake")), v.wake_word, v.command_threshold)
        return r.to_dict()

    @app.post("/api/voice/transcribe")
    async def transcribe(request: Request, sample_rate: int = 16000, route: int = 1, require_wake: int = 0):
        data = await request.body()
        if len(data) < 3200:
            raise HTTPException(422, "audio trop court")
        try:
            out = await asyncio.to_thread(studio.voice.transcribe, pcm16_to_float(data), sample_rate)
        except VoiceUnavailable as e:
            raise HTTPException(503, str(e))
        if route and out["text"]:
            v = studio.settings.get().voice
            out["route"] = (await asyncio.to_thread(route_utterance, out["text"], s1_or_none(), bool(require_wake), v.wake_word, v.command_threshold)).to_dict()
        studio.bus.publish({"type": "voice.transcript", **out})
        return out

    @app.post("/api/voice/tts")
    async def tts(request: Request):
        b = await request.json()
        try:
            wav = await asyncio.to_thread(studio.voice.synthesize, str(b.get("text", ""))[:2000], b.get("voice"), b.get("speed"))
        except VoiceUnavailable as e:
            raise HTTPException(503, str(e))
        return Response(wav, media_type="audio/wav", headers={"Cache-Control": "no-store"})

    @app.websocket("/api/voice/stream")
    async def voice_stream(ws: WebSocket):
        if not token_ok(ws.query_params.get("token")) or (ws.headers.get("origin") and ws.headers.get("origin") not in origins):
            await ws.close(code=4401); return
        await ws.accept()
        try:
            vad = studio.voice.vad_session()
        except VoiceUnavailable as e:
            await ws.send_json({"type": "error", "error": str(e)}); await ws.close(); return
        v = studio.settings.get().voice
        speaking = False
        try:
            while True:
                data = await ws.receive_bytes()
                segs = await asyncio.to_thread(vad.feed, pcm16_to_float(data))
                if vad.speaking != speaking:
                    speaking = vad.speaking
                    await ws.send_json({"type": "vad", "speaking": speaking})
                for seg in segs:
                    if len(seg) < 16000 * 0.3:
                        continue
                    out = await asyncio.to_thread(studio.voice.transcribe, seg, 16000)
                    if out["text"]:
                        out["route"] = (await asyncio.to_thread(route_utterance, out["text"], s1_or_none(), v.mode == "handsfree", v.wake_word,
                                                                v.command_threshold)).to_dict()
                    await ws.send_json({"type": "segment", **out})
        except (WebSocketDisconnect, RuntimeError):
            pass

    # ---- evenements ---------------------------------------------------------------------------------------------------------
    @app.websocket("/api/events")
    async def events(ws: WebSocket):
        if not token_ok(ws.query_params.get("token")) or (ws.headers.get("origin") and ws.headers.get("origin") not in origins):
            await ws.close(code=4401); return
        await ws.accept()
        q = studio.bus.subscribe()
        try:
            await ws.send_json({"type": "hello", "state": await asyncio.to_thread(studio.state)})
            while True:
                evt = await q.get()
                await ws.send_text(json.dumps(evt, ensure_ascii=False, default=str))
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            studio.bus.unsubscribe(q)

    # ---- API compatibles (outils tiers) ---------------------------------------------------------------------------------------
    @app.get("/v1/models")
    def v1_models():
        return {"object": "list", "data": [{"id": "bonsai", "object": "model", "owned_by": "local"}, {"id": "systemone", "object": "model", "owned_by": "local"}]}

    @app.post("/v1/chat/completions")
    async def v1_chat(request: Request):
        body = await request.body()
        url = f"{studio.runtime.s2.url}/v1/chat/completions"
        try:
            stream = json.loads(body or b"{}").get("stream", False)
        except json.JSONDecodeError:
            raise HTTPException(400, "JSON invalide")
        if not stream:
            r = await asyncio.to_thread(requests.post, url, data=body, headers={"Content-Type": "application/json"}, timeout=900)
            return Response(r.content, status_code=r.status_code, media_type="application/json")

        def gen():
            with requests.post(url, data=body, headers={"Content-Type": "application/json"}, stream=True, timeout=900) as r:
                for chunk in r.iter_content(chunk_size=None):
                    yield chunk
        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/v1/systemone")
    async def v1_systemone(request: Request):
        from jev_clone.schema import SystemOneRequest
        payload = await request.json()
        try:
            req = SystemOneRequest.model_validate(payload)
        except Exception as e:
            raise HTTPException(422, str(e)[:500])
        s1 = s1_or_none()
        if s1 is None:
            raise HTTPException(409, "le classifieur n'est pas demarre")
        return (await asyncio.to_thread(s1.answer, req)).model_dump()

    # ---- interface ------------------------------------------------------------------------------------------------------------
    boot = "<script>window.__PROPHET__=" + json.dumps({"token": studio.token, "api": "", "demo": studio.demo}) + "</script>"

    @app.get("/{path:path}")
    def spa(path: str):
        if path.startswith(("api/", "v1/")):
            raise HTTPException(404)
        f = (WEB_DIR / path).resolve()
        if path and f.is_file() and WEB_DIR.resolve() in f.parents:
            headers = {"Cache-Control": "public, max-age=31536000, immutable"} if "/assets/" in f.as_posix() else {}
            return FileResponse(f, headers=headers)
        index = WEB_DIR / "index.html"
        if not index.exists():
            return HTMLResponse("<h1>Prophet Studio</h1><p>Interface non construite : <code>cd ui && npm install && npm run build</code></p>", status_code=503)
        html = index.read_text(encoding="utf-8").replace("<!--PROPHET_BOOT-->", boot)
        return HTMLResponse(html, headers={"Cache-Control": "no-store"})

    return app


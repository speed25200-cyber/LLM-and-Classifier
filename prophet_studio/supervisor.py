"""Superviseur des deux llama-server (System Two = Bonsai, System One = classifieur).

  * argv construit depuis le plan (memes drapeaux que scripts/start_*.sh) ;
  * processus sans fenetre de console sous Windows, journal sur disque + tampon memoire pour l'interface ;
  * etat : stopped -> starting -> loading -> ready | crashed | oom ;
  * OOM au chargement : le plan descend d'un cran (planner.degrade) et on relance, jusqu'a ce que ca tienne ;
  * classifieur absent ou en echec : repli automatique en mode "mono" (Bonsai repond aussi aux questions System One).
"""

from __future__ import annotations

import collections
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import requests

from prophet_studio.planner import Plan, ServerPlan, degrade

OOM_MARKERS = ("out of memory", "cudamalloc failed", "failed to allocate", "unable to allocate", "cuda error: out of memory",
               "not enough memory", "ggml_backend_cuda_buffer_type_alloc_buffer", "erroroutofdevicememory")


def build_args(sp: ServerPlan, model: Path, port: int, mmproj: Path | None = None, lora: Path | None = None,
               family: str = "bonsai2") -> list[str]:
    a = ["-m", str(model), "--host", "127.0.0.1", "--port", str(port), "-ngl", str(sp.ngl), "-fa", "on", "-c", str(sp.ctx),
         "-np", str(sp.np), "--jinja", "--cache-ram", "2048" if sp.role == "s2" else "1024", "--ctx-checkpoints", "8", "--metrics"]
    if sp.kv_type != "f16":
        a += ["--cache-type-k", sp.kv_type, "--cache-type-v", sp.kv_type]
    if sp.threads:
        a += ["-t", str(sp.threads)]
    if sp.role == "s2":
        a += ["--temp", "1.0", "--top-p", "0.95", "--top-k", "20"] if family == "bonsai2" else ["--temp", "0.7", "--top-p", "0.95", "--top-k", "20", "--min-p", "0"]
        a += ["--reasoning-budget", str(sp.reasoning_budget), "--alias", "bonsai"]
        if sp.mmproj != "off" and mmproj is not None:
            a += ["--mmproj", str(mmproj)] + (["--no-mmproj-offload"] if sp.mmproj == "cpu" else [])
        else:
            a += ["--no-mmproj"]
        if lora is not None:
            a += ["--lora-scaled", f"{lora}:1.0"]
    else:
        a += ["-b", "2048", "-ub", "512", "--reasoning-budget", "0", "--no-mmproj", "--alias", "systemone"]
    return a


class ServerProcess:
    def __init__(self, name: str, log_dir: Path, emit: Callable[[dict], None]):
        self.name, self.log_dir, self.emit = name, log_dir, emit
        self.proc: subprocess.Popen | None = None
        self.state = "stopped"
        self.lines: collections.deque[str] = collections.deque(maxlen=400)
        self.port = 0
        self.argv: list[str] = []
        self.started_at = 0.0
        self.ready_at = 0.0
        self.error = ""
        self._stopping = False

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def public(self) -> dict:
        return {"name": self.name, "state": self.state, "port": self.port, "pid": self.proc.pid if self.proc else None,
                "load_s": round(self.ready_at - self.started_at, 1) if self.ready_at else None, "error": self.error,
                "argv": self.argv}

    def _set(self, state: str, error: str = "") -> None:
        self.state, self.error = state, error
        self.emit({"type": "runtime.server", "server": self.public()})

    def start(self, binary: list[str], args: list[str], port: int, env: dict | None = None) -> None:
        self.stop()
        self.port, self.argv = port, [*binary, *args]
        self.lines.clear()
        e = dict(os.environ, **(env or {}))
        if sys.platform.startswith("linux") and binary and Path(binary[0]).exists():
            e["LD_LIBRARY_PATH"] = f"{Path(binary[0]).parent}:{e.get('LD_LIBRARY_PATH', '')}"
        kw: dict = {}
        if sys.platform == "win32":
            kw["creationflags"] = 0x08000000 | 0x00000200   # CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
        else:
            kw["start_new_session"] = True
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log = open(self.log_dir / f"{self.name}.log", "w", encoding="utf-8", errors="replace")
        self._stopping = False
        self.started_at, self.ready_at = time.time(), 0.0
        self.proc = subprocess.Popen(self.argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=e,
                                     text=True, encoding="utf-8", errors="replace", bufsize=1, **kw)
        self._set("starting")

        def pump(p=self.proc, f=log):
            for line in p.stdout:  # type: ignore[union-attr]
                line = line.rstrip()
                self.lines.append(line)
                f.write(line + "\n"); f.flush()
                if "loading model" in line.lower() and self.state == "starting":
                    self._set("loading")
            f.close()
        threading.Thread(target=pump, daemon=True).start()

    def oom(self) -> bool:
        return any(m in l.lower() for l in self.lines for m in OOM_MARKERS)

    def wait_ready(self, timeout: float = 600.0, poll: float = 0.25) -> str:
        """ready | oom | crashed | timeout"""
        t0 = time.time()
        s = requests.Session()
        while time.time() - t0 < timeout:
            if self.proc is None:
                return "crashed"
            rc = self.proc.poll()
            if rc is not None:
                time.sleep(0.1)  # laisser la pompe lire les dernieres lignes
                st = "oom" if self.oom() else "crashed"
                tail = "\n".join(list(self.lines)[-6:])
                self._set(st, tail[-600:] or f"code de sortie {rc}")
                return st
            try:
                r = s.get(f"{self.url}/health", timeout=2)
                if r.status_code == 200:
                    self.ready_at = time.time()
                    self._set("ready")
                    return "ready"
                if r.status_code == 503 and self.state == "starting":
                    self._set("loading")
            except requests.RequestException:
                pass
            time.sleep(poll)
        self._set("crashed", "delai de demarrage depasse")
        return "timeout"

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        p, self.proc = self.proc, None
        self._stopping = True
        if p is not None and p.poll() is None:
            try:
                p.terminate()
                p.wait(timeout=8)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        if self.state != "stopped":
            self._set("stopped")


class Runtime:
    """Lance, surveille et arrete les deux serveurs selon un plan, avec l'echelle anti-OOM."""

    def __init__(self, log_dir: Path, emit: Callable[[dict], None], resolve_model: Callable[[str], Path | None],
                 resolve_mmproj: Callable[[str], Path | None], server_cmd: Callable[[], list[str] | None],
                 installed: Callable[[], set[str]]):
        self.emit = emit
        self.s2 = ServerProcess("s2", log_dir, emit)
        self.s1 = ServerProcess("s1", log_dir, emit)
        self.resolve_model, self.resolve_mmproj, self.server_cmd, self.installed = resolve_model, resolve_mmproj, server_cmd, installed
        self.plan: Plan | None = None
        self.mono = False
        self.state = "stopped"        # stopped | starting | ready | degraded | error
        self.message = ""
        self._lock = threading.Lock()
        self.lora: Path | None = None
        # sonde de VRAM utilisee (Mio, None sans GPU) et rappel de calibration : la difference avant / apres le
        # demarrage d'un serveur donne sa consommation reelle (NVML ne donne pas la memoire par processus sous
        # Windows WDDM, la difference globale si)
        self.vram_probe: Callable[[], float | None] | None = None
        self.on_measure: Callable[[str, ServerPlan, float], None] | None = None

    def public(self) -> dict:
        return {"state": self.state, "message": self.message, "mono": self.mono, "plan": self.plan.to_dict() if self.plan else None,
                "servers": {"s2": self.s2.public(), "s1": self.s1.public()}, "s1_url": self.s1_url, "s2_url": self.s2.url}

    @property
    def s1_url(self) -> str:
        return self.s2.url if self.mono else self.s1.url

    def _status(self, state: str, message: str = "") -> None:
        self.state, self.message = state, message
        self.emit({"type": "runtime.status", "runtime": self.public()})

    def pids(self) -> set[int]:
        return {p.proc.pid for p in (self.s1, self.s2) if p.proc is not None}

    def start(self, plan: Plan, ports: tuple[int, int], max_rungs: int = 16) -> bool:
        with self._lock:
            cmd = self.server_cmd()
            if not cmd:
                self._status("error", "runtime llama.cpp absent : installez-le depuis l'ecran Modeles")
                return False
            self.stop()
            self._status("starting", "demarrage de Bonsai (System Two)")
            p = plan
            for _ in range(max_rungs):
                model = self.resolve_model(p.s2.model_id)
                if model is None:
                    self._status("error", f"modele {p.s2.model_id} non installe")
                    return False
                mm = self.resolve_mmproj(p.s2.model_id) if p.s2.mmproj != "off" else None
                fam = "bonsai2" if p.s2.model_id.startswith("bonsai2") else "bonsai"
                before = self._probe()
                self.s2.start(cmd, build_args(p.s2, model, ports[0], mm, self.lora, fam), ports[0])
                r = self.s2.wait_ready()
                if r == "ready":
                    self._measure("s2", p.s2, before)
                    break
                if r == "oom":
                    nxt = degrade(p, self.installed())
                    if nxt is None:
                        self._status("error", "memoire insuffisante meme au plus petit reglage")
                        return False
                    p = nxt
                    self.emit({"type": "runtime.degraded", "plan": p.to_dict(), "note": p.notes[-1]})
                    continue
                self._status("error", f"Bonsai n'a pas demarre : {self.s2.error[-300:]}")
                return False
            else:
                return False
            # System One
            self.mono = False
            if p.s1 is not None:
                s1m = self.resolve_model(p.s1.model_id)
                if s1m is None:
                    self.mono = True
                else:
                    self._status("starting", "demarrage du classifieur (System One)")
                    for _ in range(3):
                        before = self._probe()
                        self.s1.start(cmd, build_args(p.s1, s1m, ports[1]), ports[1])
                        r = self.s1.wait_ready(timeout=180)
                        if r == "ready":
                            if p.s1.device == "gpu":
                                self._measure("s1", p.s1, before)
                            break
                        if r == "oom" and p.s1.device == "gpu":
                            p = degrade(p, self.installed()) or p  # premier cran utile : classifieur sur CPU
                            continue
                        self.mono = True
                        break
            else:
                self.mono = True
            self.plan = p
            if self.mono:
                self._status("degraded" if p.s1 is not None else "ready",
                             "classifieur indisponible : Bonsai repond aussi aux decisions System One (mode mono)")
            else:
                self._status("ready", "")
            return True

    def _probe(self) -> float | None:
        try:
            return self.vram_probe() if self.vram_probe else None
        except Exception:
            return None

    def _measure(self, role: str, sp: ServerPlan, before: float | None) -> None:
        if before is None or self.on_measure is None or sp.device != "gpu":
            return
        time.sleep(1.0)   # laisser les tampons de calcul s'allouer
        after = self._probe()
        if after is not None and after > before:
            try:
                self.on_measure(role, sp, after - before)
            except Exception:
                pass

    def stop(self) -> None:
        self.s1.stop()
        self.s2.stop()
        if self.state != "stopped":
            self._status("stopped")

    def healthy(self) -> bool:
        return self.s2.alive() and (self.mono or self.s1.alive())

    def logs(self, name: str, n: int = 200) -> list[str]:
        p = self.s2 if name == "s2" else self.s1
        return list(p.lines)[-n:]

    def plan_dict(self) -> dict | None:
        return asdict(self.plan) if self.plan else None

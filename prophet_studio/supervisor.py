"""Superviseur des deux llama-server (System Two = Bonsai, System One = classifieur).

  * argv construit depuis le plan (memes drapeaux que scripts/start_*.sh) ;
  * processus sans fenetre de console sous Windows, journal sur disque + tampon memoire pour l'interface ;
  * etat : stopped -> starting -> loading -> ready | crashed | oom ;
  * OOM au chargement : le plan descend d'un cran (planner.degrade) et on relance, jusqu'a ce que ca tienne ;
    OOM du classifieur sur GPU : il passe directement sur CPU (planner.s1_on_cpu), Bonsai n'est pas touche ;
  * classifieur absent ou en echec : repli automatique en mode "mono" (Bonsai repond aussi aux questions System One) ;
    s'il est installe ensuite, il demarre seul et le duo revient (attach_s1), sans relancer Bonsai ;
  * watchdog apres le demarrage : un serveur qui s'arrete tout seul est detecte (poll du processus). System One est
    relance une fois (Bonsai repond en attendant), puis mode mono ; Bonsai est relance une fois, sinon etat error.
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

from prophet_studio.planner import Plan, ServerPlan, degrade, mono_plan, s1_on_cpu

MONO_MSG = "classifieur indisponible : Bonsai repond aussi aux decisions System One (mode mono)"
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
    if sp.np > 1:
        # plusieurs slots : un seul tampon KV partage, chaque slot dispose du contexte entier (au lieu de ctx / np) ; un slot
        # inactif garde son cache (avec -kvu, --cache-idle-slots le viderait a chaque nouvelle requete). Fork prism-b10683.
        a += ["-kvu", "--no-cache-idle-slots"]
    if sp.device == "cpu":
        a += ["--device", "none"]   # aucun contexte ni tampon de calcul sur le GPU : voir server_env()
    return a


def server_env(sp: ServerPlan) -> dict[str, str]:
    """Serveur prevu sur CPU : aucun GPU visible, sinon le binaire CUDA y cree quand meme un contexte et un tampon de calcul
    que le planificateur ne budgete pas (a cote de Bonsai 2 27B sur 8 Go, il ne reste que ~140 Mio). "-1" (index invalide =
    aucun peripherique) plutot qu'une valeur vide, qui peut etre lue comme absente (Windows)."""
    return {"CUDA_VISIBLE_DEVICES": "-1", "HIP_VISIBLE_DEVICES": "-1"} if sp.device == "cpu" else {}


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

    def exit_code(self) -> int | None:
        """Code de sortie si le processus s'est arrete tout seul (None s'il tourne, ou apres stop())."""
        p = self.proc
        return p.poll() if p is not None else None

    def mark_exited(self, rc: int) -> str:
        time.sleep(0.1)  # laisser la pompe lire les dernieres lignes
        st = "oom" if self.oom() else "crashed"
        tail = "\n".join(list(self.lines)[-6:])
        self._set(st, tail[-600:] or f"code de sortie {rc}")
        return st

    def wait_ready(self, timeout: float = 600.0, poll: float = 0.25, abort: Callable[[], bool] | None = None) -> str:
        """ready | oom | crashed | timeout | stopped (abort() vrai : arret demande pendant l'attente)"""
        t0 = time.time()
        s = requests.Session()
        while time.time() - t0 < timeout:
            if abort is not None and abort():
                return "stopped"
            if self.proc is None:
                return "crashed"
            rc = self.proc.poll()
            if rc is not None:
                return self.mark_exited(rc)
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
        # watchdog : un Event par demarrage reussi (stop() le leve) ; _guard serialise ses changements d'etat avec stop()
        # (aucun serveur n'est relance apres un arret) ; _s1_busy : un seul (re)demarrage de System One a la fois
        self.watch_interval = 2.0
        self.restarts = {"s1": 0, "s2": 0}
        self._watch: threading.Event | None = None
        self._guard = threading.Lock()
        self._s1_busy = threading.Lock()
        self._cmd: list[str] = []
        self._ports = (0, 0)
        self._s2_args: list[str] = []
        self.lora: Path | None = None
        # sonde de VRAM utilisee (Mio, None sans GPU) et rappel de calibration : la difference avant / apres le
        # demarrage d'un serveur donne sa consommation reelle (NVML ne donne pas la memoire par processus sous
        # Windows WDDM, la difference globale si)
        self.vram_probe: Callable[[], float | None] | None = None
        self.on_measure: Callable[[str, ServerPlan, float], None] | None = None

    def public(self) -> dict:
        return {"state": self.state, "message": self.message, "mono": self.mono, "plan": self.plan.to_dict() if self.plan else None,
                "servers": {"s2": self.s2.public(), "s1": self.s1.public()}, "s1_url": self.s1_url, "s2_url": self.s2.url,
                "restarts": dict(self.restarts)}

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
            self._cmd, self._ports, self.restarts = cmd, ports, {"s1": 0, "s2": 0}
            self._status("starting", "demarrage de Bonsai (System Two)")
            p = plan
            # classifieur absent des le depart : le mode mono est connu avant Bonsai, qui recoit alors un 2e slot si possible
            s1m = self.resolve_model(p.s1.model_id) if p.s1 is not None else None
            if s1m is None:
                p = mono_plan(p)
            for _ in range(max_rungs):
                model = self.resolve_model(p.s2.model_id)
                if model is None:
                    self._status("error", f"modele {p.s2.model_id} non installe")
                    return False
                mm = self.resolve_mmproj(p.s2.model_id) if p.s2.mmproj != "off" else None
                fam = "bonsai2" if p.s2.model_id.startswith("bonsai2") else "bonsai"
                before = self._probe()
                args = build_args(p.s2, model, ports[0], mm, self.lora, fam)
                self.s2.start(cmd, args, ports[0], server_env(p.s2))
                r = self.s2.wait_ready()
                if r == "ready":
                    self._s2_args = args
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
            self.mono = s1m is None
            if s1m is not None:
                self._status("starting", "demarrage du classifieur (System One)")
                q = self._start_s1(p, s1m, None)
                p, self.mono = (q, False) if q is not None else (p, True)
            self.plan = p
            if self.mono:
                self._status("degraded" if p.s1 is not None else "ready", MONO_MSG)
            else:
                self._status("ready", "")
            self._watch = ev = threading.Event()
            threading.Thread(target=self._watchdog, args=(ev,), daemon=True, name="runtime-watchdog").start()
            return True

    def _spawn(self, sp: ServerProcess, args: list[str], port: int, env: dict, ev: threading.Event | None) -> bool:
        with self._guard:   # un stop() concurrent gagne toujours : aucun serveur n'est lance apres un arret
            if ev is not None and ev.is_set():
                return False
            sp.start(self._cmd, args, port, env)
            return True

    def _start_s1(self, p: Plan, s1m: Path, ev: threading.Event | None) -> Plan | None:
        """Demarre System One ; OOM sur GPU -> directement sur CPU. Renvoie le plan retenu, None en cas d'echec (mode mono)."""
        for _ in range(2):
            before = self._probe()
            if not self._spawn(self.s1, build_args(p.s1, s1m, self._ports[1]), self._ports[1], server_env(p.s1), ev):
                return None
            r = self.s1.wait_ready(timeout=180, abort=ev.is_set if ev is not None else None)
            if r == "ready":
                if p.s1.device == "gpu":
                    self._measure("s1", p.s1, before)
                return p
            if r == "oom" and p.s1.device == "gpu":
                p = s1_on_cpu(p)
                self.emit({"type": "runtime.degraded", "plan": p.to_dict(), "note": p.notes[-1]})
                continue
            return None
        return None

    # ---- watchdog : arret inattendu d'un serveur apres le demarrage ---------------------------------------------------
    def _watchdog(self, ev: threading.Event) -> None:
        while not ev.wait(self.watch_interval):
            if self.s2.exit_code() is not None:
                if not self._s2_died(ev):
                    return
            elif not self.mono and self.s1.exit_code() is not None:
                self._s1_died(ev)

    def _s1_died(self, ev: threading.Event) -> None:
        """System One est relance une fois (Bonsai repond en attendant : mode mono), puis reste en mode mono."""
        if not self._s1_busy.acquire(blocking=False):
            return
        try:
            with self._guard:
                if ev.is_set():
                    return
                self.s1.mark_exited(self.s1.exit_code() or 0)
                self.s1.proc, self.mono = None, True
                again = self.restarts["s1"] < 1
                self.restarts["s1"] += int(again)
                self._status("degraded", "classifieur arrete de facon inattendue : redemarrage, Bonsai repond en attendant (mode mono)"
                             if again else f"{MONO_MSG} ; arret : {self.s1.error[-200:]}")
            if again:
                self._restore_s1(ev, "classifieur redemarre apres un arret inattendu")
        finally:
            self._s1_busy.release()

    def _s2_died(self, ev: threading.Event) -> bool:
        """Bonsai est relance une fois avec les memes arguments ; sinon etat error. Faux : le watchdog s'arrete."""
        with self._guard:
            if ev.is_set():
                return False
            self.s2.mark_exited(self.s2.exit_code() or 0)
            self.s2.proc = None
            if self.restarts["s2"] >= 1:
                self._status("error", f"Bonsai (System Two) s'est arrete : {self.s2.error[-300:]}")
                return False
            self.restarts["s2"] += 1
            self._status("error", "Bonsai (System Two) s'est arrete de facon inattendue : redemarrage automatique")
        ok = (self._spawn(self.s2, self._s2_args, self._ports[0], server_env(self.plan.s2), ev)
              and self.s2.wait_ready(abort=ev.is_set) == "ready")
        with self._guard:
            if ev.is_set():
                return False
            if not ok:
                self._status("error", f"Bonsai n'a pas pu redemarrer : {self.s2.error[-300:]}")
                return False
            mono = self.mono and self.plan.s1 is not None
            self._status("degraded" if mono else "ready", "Bonsai redemarre apres un arret inattendu" + (f" ; {MONO_MSG}" if mono else ""))
            return True

    def _restore_s1(self, ev: threading.Event, ok_msg: str) -> bool:
        p = self.plan
        s1m = self.resolve_model(p.s1.model_id) if p is not None and p.s1 is not None else None
        q = self._start_s1(p, s1m, ev) if s1m is not None else None
        with self._guard:
            if ev.is_set():
                return False
            if q is None:
                self._status("degraded", MONO_MSG)
                return False
            self.plan, self.mono = q, False
            self._status("ready", ok_msg)
            return True

    def can_attach_s1(self) -> bool:
        p = self.plan
        return (self._watch is not None and self.mono and self.state in ("ready", "degraded") and p is not None
                and p.s1 is not None and self.resolve_model(p.s1.model_id) is not None)

    def attach_s1(self) -> bool:
        """Classifieur installe apres coup alors que Bonsai tourne seul (mode mono) : System One demarre sans relancer
        Bonsai et le duo revient."""
        ev = self._watch
        if ev is None or not self.can_attach_s1() or not self._s1_busy.acquire(blocking=False):
            return False
        try:
            with self._guard:
                if ev.is_set() or not self.mono:
                    return False
                self._status(self.state, "demarrage du classifieur (System One)")
            return self._restore_s1(ev, "classifieur demarre : duo Bonsai + System One")
        finally:
            self._s1_busy.release()

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
        with self._guard:
            if self._watch is not None:
                self._watch.set()
                self._watch = None
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

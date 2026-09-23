"""Superviseur des deux llama-server (System Two = Bonsai, System One = classifieur).

  * argv construit depuis le plan (memes drapeaux que scripts/start_*.sh) ; plusieurs slots = KV non unifie, -c = contexte
    par slot x slots (planner.kv_ctx) : chaque slot a son contexte entier, garanti ;
  * processus sans fenetre de console sous Windows (Job Object de la coquille) ; sous Unix, meme groupe de processus que le
    coeur : l'arret force du groupe emporte aussi les llama-server ;
  * journal sur disque (<nom>.log, le lancement precedent en <nom>.prev.log) + tampon memoire pour l'interface ;
  * etat : stopped -> starting -> loading -> ready | crashed | oom ; last_error garde la cause du dernier arret anormal ;
  * OOM au chargement : le plan descend d'un cran (planner.degrade) et on relance, jusqu'a ce que ca tienne ;
    OOM du classifieur sur GPU : il passe directement sur CPU (planner.s1_on_cpu), Bonsai n'est pas touche ;
  * delai de demarrage depasse : le processus est arrete (il ne garde ni RAM ni VRAM) ;
  * classifieur absent ou en echec : repli automatique en mode "mono" (Bonsai repond aussi aux questions System One) ;
    s'il est installe ensuite, il demarre seul et le duo revient (attach_s1), sans relancer Bonsai ; un classifieur
    abandonne (echec, 2e arret) ne revient que si son fichier change ;
  * stop() gagne toujours, meme pendant le premier demarrage : plus rien n'est lance ni publie ensuite, et un demarrage
    deja demande (en file derriere un autre) n'a plus lieu (numero de generation) ;
  * watchdog apres le demarrage : un serveur qui s'arrete tout seul est detecte (poll du processus). System One est
    relance une fois dans son propre fil (Bonsai repond en attendant, toujours surveille), puis mode mono ; Bonsai est
    relance une fois (etat transitoire "starting", restarting = "s2"), sinon etat error. Le retour du classifieur ne publie
    jamais "ready" sans Bonsai. Une erreur imprevue n'arrete jamais la surveillance.
"""

from __future__ import annotations

import collections
import contextlib
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

import requests

from prophet_studio.planner import Plan, ServerPlan, attach_plan, degrade, kv_ctx, mono_plan, s1_on_cpu

MONO_MSG = "classifieur indisponible : Bonsai repond aussi aux decisions System One (mode mono)"
OOM_MARKERS = ("out of memory", "cudamalloc failed", "failed to allocate", "unable to allocate", "cuda error: out of memory",
               "not enough memory", "ggml_backend_cuda_buffer_type_alloc_buffer", "erroroutofdevicememory")
ERR_LINE = re.compile(r"GGML_ASSERT|error|failed|out of memory|exception|what\(\)|segmentation fault", re.I)


def build_args(sp: ServerPlan, model: Path, port: int, mmproj: Path | None = None, lora: Path | None = None,
               family: str = "bonsai2") -> list[str]:
    a = ["-m", str(model), "--host", "127.0.0.1", "--port", str(port), "-ngl", str(sp.ngl), "-fa", "on", "-c", str(kv_ctx(sp)),
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
        # plusieurs slots, KV non unifie (pas de -kvu) : -c = contexte par slot x slots, chaque slot a son contexte entier et
        # garanti. Avec -kvu, les slots se partagent un seul tampon : des lectures paralleles d'un gros etat (ou une lecture S1
        # pendant une reponse de Bonsai) le remplissent, et le serveur renvoie "Context size has been exceeded" a TOUTES les
        # requetes en cours. Un slot inactif garde son cache sans copie en RAM a chaque nouvelle requete. Fork prism-b10683.
        a += ["--no-cache-idle-slots"]
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
        # dernier arret anormal (etat, code, cause lisible, fin du journal) : survit aux relances, jamais remis a zero
        self.last_error: dict | None = None
        self._stopping = False

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def public(self) -> dict:
        return {"name": self.name, "state": self.state, "port": self.port, "pid": self.proc.pid if self.proc else None,
                "load_s": round(self.ready_at - self.started_at, 1) if self.ready_at else None, "error": self.error,
                "argv": self.argv, "last_error": self.last_error}

    def _set(self, state: str, error: str = "") -> None:
        self.state, self.error = state, error
        self.emit({"type": "runtime.server", "server": self.public()})

    def _crash(self, state: str, error: str, rc: int | None = None) -> None:
        """Etat d'echec + last_error. Cause : derniere ligne d'erreur de la fin du journal et le code de sortie (arret du
        processus), ou le message lui-meme (lancement impossible, delai depasse)."""
        lines = list(self.lines)
        if rc is None:
            reason = error
        else:
            code = f"signal {-rc}" if rc < 0 else f"code de sortie {rc}"
            hit = next((l.strip() for l in reversed(lines[-12:]) if ERR_LINE.search(l)), "")
            reason = f"{hit[:160]} ({code})" if hit else code
        self.last_error = {"at": time.time(), "state": state, "rc": rc, "reason": reason, "tail": lines[-20:]}
        self._set(state, error)

    def reason(self) -> str:
        """Cause lisible du dernier echec (pour les messages d'etat), sinon la fin du journal."""
        return (self.last_error or {}).get("reason") or self.error[-200:]

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
        # Unix : pas de nouvelle session, le serveur reste dans le groupe de processus du coeur. La coquille de bureau tue ce
        # groupe (SIGKILL sur -pgid) si le coeur ne s'arrete pas : les llama-server partent avec lui au lieu de garder la VRAM.
        self.log_dir.mkdir(parents=True, exist_ok=True)
        lp = self.log_dir / f"{self.name}.log"
        with contextlib.suppress(OSError):   # journal du lancement precedent garde (cause d'un arret avant une relance)
            if lp.stat().st_size:
                os.replace(lp, self.log_dir / f"{self.name}.prev.log")
        log = open(lp, "w", encoding="utf-8", errors="replace")
        self._stopping = False
        self.started_at, self.ready_at = time.time(), 0.0
        try:
            self.proc = subprocess.Popen(self.argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=e,
                                         text=True, encoding="utf-8", errors="replace", bufsize=1, **kw)
        except OSError as ex:   # binaire absent ou non executable (runtime en reinstallation...) : un echec de lancement ordinaire
            log.write(f"lancement impossible : {ex}\n")
            log.close()
            self.proc = None
            self._crash("crashed", f"lancement impossible : {ex}"[:600])
            return
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
        self._crash(st, tail[-600:] or f"code de sortie {rc}", rc)
        return st

    def wait_ready(self, timeout: float = 600.0, poll: float = 0.25, abort: Callable[[], bool] | None = None) -> str:
        """ready | oom | crashed | timeout | stopped (abort() vrai : arret demande pendant l'attente)"""
        t0 = time.time()
        s = requests.Session()
        while time.time() - t0 < timeout:
            if abort is not None and abort():
                return "stopped"
            p = self.proc   # lu une seule fois : un stop() concurrent le remet a None
            if p is None:
                return "stopped" if abort is not None and abort() else "crashed"
            rc = p.poll()
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
        # delai depasse : le processus est arrete ici (il garderait sinon RAM / VRAM / CPU alors que le runtime y a renonce) ;
        # proc reste en place, comme apres un arret : l'etat "crashed" et sa cause restent visibles
        p = self.proc
        if p is not None and p.poll() is None:
            try:
                p.terminate()
                p.wait(timeout=8)
            except Exception:
                with contextlib.suppress(Exception):
                    p.kill()
        self._crash("crashed", "delai de demarrage depasse")
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
        # un Event par demarrage, cree avant Bonsai (stop() le leve) : le demarrage, le watchdog et attach_s1 s'y arretent ;
        # _guard serialise leurs changements d'etat avec stop() (aucun serveur n'est lance ni aucun etat publie apres un
        # arret) ; _s1_busy : un seul (re)demarrage de System One a la fois
        self.watch_interval = 2.0
        self.s1_timeout, self.s2_timeout = 180.0, 600.0
        self.restarts = {"s1": 0, "s2": 0}
        self.restarting: str | None = None           # "s1" | "s2" : relance automatique en cours (etat transitoire)
        self._s1_failed: tuple | None = None         # empreinte du fichier du classifieur abandonne (voir can_attach_s1)
        self._watch: threading.Event | None = None
        self._guard = threading.Lock()
        self._s1_busy = threading.Lock()
        self._gen = 0                                # +1 a chaque stop() : un demarrage en file d'une generation passee n'a pas lieu
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
                "restarts": dict(self.restarts), "restarting": self.restarting}

    @property
    def s1_url(self) -> str:
        return self.s2.url if self.mono else self.s1.url

    def _status(self, state: str, message: str = "") -> None:
        self.state, self.message = state, message
        self.emit({"type": "runtime.status", "runtime": self.public()})

    def _status_if(self, ev: threading.Event, state: str, message: str = "") -> bool:
        """Publie l'etat sauf si un stop() est passe entre-temps (il a deja publie "stopped")."""
        with self._guard:
            if ev.is_set():
                return False
            self._status(state, message)
            return True

    def _fail(self, ev: threading.Event, message: str) -> bool:
        self._status_if(ev, "error", message)
        return False

    def pids(self) -> set[int]:
        return {p.proc.pid for p in (self.s1, self.s2) if p.proc is not None}

    def generation(self) -> int:
        """Numero d'arret : un demarrage demande avant le dernier stop() n'a plus lieu (voir start(gen=...)). Lu sans verrou
        (un entier) : une requete ne reste jamais bloquee derriere un arret en cours ; start() le compare sous _guard."""
        return self._gen

    def note(self, message: str) -> None:
        """Message du moteur sans changer d'etat (ex. demarrage automatique suspendu)."""
        with self._guard:
            self._status(self.state, message)

    def start(self, plan: Plan, ports: tuple[int, int], max_rungs: int = 16, gen: int | None = None) -> bool:
        """gen : generation() lue a la demande. Un demarrage en file derriere un autre (verrou) ne part pas si un Stop est
        passe entre-temps : stop() gagne aussi contre les demarrages deja demandes."""
        with self._lock:
            cmd = self.server_cmd()
            with self._guard:
                if gen is not None and gen != self._gen:
                    return False
                if not cmd:
                    self._status("error", "runtime llama.cpp absent : installez-le depuis l'ecran Modeles")
                    return False
                self._halt_locked()
                self._watch = ev = threading.Event()
                self._cmd, self._ports, self.restarts, self.restarting, self._s1_failed = cmd, ports, {"s1": 0, "s2": 0}, None, None
                self._status("starting", "demarrage de Bonsai (System Two)")
            p = plan
            # classifieur absent des le depart : le mode mono est connu avant Bonsai, qui recoit alors un 2e slot si possible
            s1m = self.resolve_model(p.s1.model_id) if p.s1 is not None else None
            if s1m is None:
                p = mono_plan(p)
            for _ in range(max_rungs):
                model = self.resolve_model(p.s2.model_id)
                if model is None:
                    return self._fail(ev, f"modele {p.s2.model_id} non installe")
                mm = self.resolve_mmproj(p.s2.model_id) if p.s2.mmproj != "off" else None
                fam = "bonsai2" if p.s2.model_id.startswith("bonsai2") else "bonsai"
                before = self._probe()
                args = build_args(p.s2, model, ports[0], mm, self.lora, fam)
                if not self._spawn(self.s2, args, ports[0], server_env(p.s2), ev):
                    return False
                r = self.s2.wait_ready(timeout=self.s2_timeout, abort=ev.is_set)
                if ev.is_set():          # arret demande : stop() a deja tout arrete et publie "stopped"
                    return False
                if r == "ready":
                    self._s2_args = args
                    self._measure("s2", p.s2, before, ev)
                    break
                if r == "oom":
                    nxt = degrade(p, self.installed())
                    if nxt is None:
                        return self._fail(ev, "memoire insuffisante meme au plus petit reglage")
                    p = nxt
                    self.emit({"type": "runtime.degraded", "plan": p.to_dict(), "note": p.notes[-1]})
                    continue
                return self._fail(ev, f"Bonsai n'a pas demarre : {self.s2.reason()[-300:]}")
            else:
                return self._fail(ev, "memoire insuffisante : echelle anti-OOM epuisee")
            # System One
            q = None
            if s1m is not None:
                if not self._status_if(ev, "starting", "demarrage du classifieur (System One)"):
                    return False
                q = self._start_s1(p, s1m, ev)
            with self._guard:
                if ev.is_set():
                    return False
                self.plan, self.mono = (q or p), q is None
                if self.mono:
                    if s1m is not None:          # classifieur present mais en echec : il ne revient que si son fichier change
                        self._s1_failed = self._s1_sig()
                    self._status("degraded" if p.s1 is not None else "ready",
                                 MONO_MSG + (f" ; classifieur en echec : {self.s1.reason()}" if s1m is not None else ""))
                else:
                    self._status("ready", "")
                threading.Thread(target=self._watchdog, args=(ev,), daemon=True, name="runtime-watchdog").start()
            return True

    def _spawn(self, sp: ServerProcess, args: list[str], port: int, env: dict, ev: threading.Event) -> bool:
        with self._guard:   # un stop() concurrent gagne toujours : aucun serveur n'est lance apres un arret
            if ev.is_set():
                return False
            sp.start(self._cmd, args, port, env)
            return True

    def _start_s1(self, p: Plan, s1m: Path, ev: threading.Event) -> Plan | None:
        """Demarre System One ; OOM sur GPU -> directement sur CPU. Renvoie le plan retenu, None en cas d'echec (mode mono)."""
        for _ in range(2):
            before = self._probe()
            if not self._spawn(self.s1, build_args(p.s1, s1m, self._ports[1]), self._ports[1], server_env(p.s1), ev):
                return None
            r = self.s1.wait_ready(timeout=self.s1_timeout, abort=ev.is_set)
            if r == "ready":
                if p.s1.device == "gpu":
                    self._measure("s1", p.s1, before, ev)
                return p
            if r == "oom" and p.s1.device == "gpu" and not ev.is_set():
                p = s1_on_cpu(p)
                self.emit({"type": "runtime.degraded", "plan": p.to_dict(), "note": p.notes[-1]})
                continue
            return None
        return None

    def _s1_sig(self) -> tuple | None:
        """Empreinte (chemin, taille, date) du fichier du classifieur du plan : apres un abandon, seul un fichier nouveau ou
        change (reinstallation, import) le fait revenir ; une installation sans rapport ne relance pas un classifieur qui plante."""
        p = self.plan
        m = self.resolve_model(p.s1.model_id) if p is not None and p.s1 is not None else None
        try:
            st = m.stat() if m is not None else None
        except OSError:
            return None
        return (str(m), st.st_size, st.st_mtime_ns) if st is not None else None

    # ---- watchdog : arret inattendu d'un serveur apres le demarrage ---------------------------------------------------
    def _watchdog(self, ev: threading.Event) -> None:
        while not ev.wait(self.watch_interval):
            try:
                if self.s2.exit_code() is not None:
                    if not self._s2_died(ev):
                        return
                elif not self.mono and self.s1.exit_code() is not None:
                    self._s1_died(ev)
            except Exception as e:   # filet : une erreur imprevue (relance...) n'arrete jamais la surveillance en silence
                with self._guard:
                    if ev.is_set():
                        return
                    self.restarting = None
                    st = "error" if not self.s2.alive() else ("degraded" if self.mono else self.state)
                    self._status(st, f"surveillance : {type(e).__name__}: {e}"[:300])

    def _s1_died(self, ev: threading.Event) -> None:
        """System One est relance une fois (Bonsai repond en attendant : mode mono), puis reste en mode mono. La relance
        (jusqu'a s1_timeout) tourne dans son propre fil : le watchdog continue de surveiller Bonsai pendant ce temps."""
        if not self._s1_busy.acquire(blocking=False):
            return
        handed = False
        try:
            with self._guard:
                if ev.is_set():
                    return
                self.s1.mark_exited(self.s1.exit_code() or 0)
                self.s1.proc, self.mono = None, True
                again = self.restarts["s1"] < 1
                self.restarts["s1"] += int(again)
                cause = self.s1.reason()
                if again:
                    self.restarting = "s1"
                    self._status("degraded", f"classifieur arrete de facon inattendue ({cause}) : redemarrage, "
                                             "Bonsai repond en attendant (mode mono)")
                else:
                    self._s1_failed = self._s1_sig()
                    self._status("degraded", f"{MONO_MSG} ; 2e arret inattendu du classifieur : {cause}")
            if again:
                threading.Thread(target=self._relaunch_s1, args=(ev, cause), daemon=True, name="runtime-s1").start()
                handed = True
        finally:
            if not handed:
                self._s1_busy.release()

    def _relaunch_s1(self, ev: threading.Event, cause: str) -> None:
        try:
            self._restore_s1(ev, f"classifieur redemarre apres un arret inattendu ({cause})")
        except Exception as e:   # meme filet que le watchdog : une erreur imprevue ne laisse pas la relance affichee en silence
            with self._guard:
                if not ev.is_set() and self.restarting == "s1":
                    self.restarting = None
                    if self.s2.alive():
                        self._status("degraded", f"surveillance : {type(e).__name__}: {e}"[:300])
        finally:
            self._s1_busy.release()

    def _s2_died(self, ev: threading.Event) -> bool:
        """Bonsai est relance une fois avec les memes arguments ; sinon etat error. Faux : le watchdog s'arrete."""
        with self._guard:
            if ev.is_set():
                return False
            self.s2.mark_exited(self.s2.exit_code() or 0)
            self.s2.proc = None
            cause = self.s2.reason()
            if self.restarts["s2"] >= 1:
                self._status("error", f"Bonsai (System Two) s'est arrete a nouveau : {cause}")
                return False
            self.restarts["s2"] += 1
            # etat transitoire, pas "error" : l'interface affiche une relance, une fin d'installation ne relance pas tout
            self.restarting = "s2"
            self._status("starting", f"Bonsai (System Two) s'est arrete de facon inattendue ({cause}) : redemarrage automatique")
        ok = (self._spawn(self.s2, self._s2_args, self._ports[0], server_env(self.plan.s2), ev)
              and self.s2.wait_ready(timeout=self.s2_timeout, abort=ev.is_set) == "ready")
        with self._guard:
            if ev.is_set():
                return False
            self.restarting = None
            if not ok:
                self._status("error", f"Bonsai n'a pas pu redemarrer : {self.s2.reason()}")
                return False
            mono = self.mono and self.plan.s1 is not None
            self._status("degraded" if mono else "ready",
                         f"Bonsai redemarre apres un arret inattendu ({cause})" + (f" ; {MONO_MSG}" if mono else ""))
            return True

    def _restore_s1(self, ev: threading.Event, ok_msg: str) -> bool:
        p = self.plan
        s1m = self.resolve_model(p.s1.model_id) if p is not None and p.s1 is not None else None
        q = self._start_s1(attach_plan(p), s1m, ev) if s1m is not None else None
        with self._guard:
            if ev.is_set():
                return False
            if self.restarting == "s1":          # seulement la sienne : une relance de Bonsai en cours garde "s2"
                self.restarting = None
            # Bonsai en relance (_s2_died publiera l'etat final, il lit mono et plan) ou arrete sans que le watchdog l'ait
            # encore vu (il le verra, ou l'etat reste "error") : rien n'est publie ici, jamais "ready" sans Bonsai
            say = self.restarting is None and self.s2.alive()
            if q is None:
                if s1m is not None:              # echec de la relance : abandon jusqu'a un nouveau fichier
                    self._s1_failed = self._s1_sig()
                if say:
                    self._status("degraded", MONO_MSG + (f" ; relance du classifieur impossible : {self.s1.reason()}" if s1m is not None else ""))
                return False
            self.plan, self.mono, self._s1_failed = q, False, None
            if say:
                self._status("ready", ok_msg)
            return True

    def can_attach_s1(self) -> bool:
        """Mode mono, classifieur installe, et soit il etait absent, soit son fichier a change depuis son abandon (le
        watchdog ne le relance qu'une fois : une installation sans rapport ne doit pas contourner ce plafond)."""
        p = self.plan
        return (self._watch is not None and self.mono and self.restarting is None and self.state in ("ready", "degraded")
                and p is not None and p.s1 is not None and self.resolve_model(p.s1.model_id) is not None
                and (self._s1_failed is None or self._s1_sig() != self._s1_failed))

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

    def _measure(self, role: str, sp: ServerPlan, before: float | None, ev: threading.Event) -> None:
        if before is None or self.on_measure is None or sp.device != "gpu":
            return
        if ev.wait(1.0):   # laisser les tampons de calcul s'allouer ; un stop() pendant l'attente l'interrompt
            return
        after = self._probe()
        if after is not None and after > before:
            try:
                # KV reellement alloue (-c = contexte par slot x slots) : le surcout deduit n'inclut pas les slots
                self.on_measure(role, replace(sp, ctx=kv_ctx(sp), np=1), after - before)
            except Exception:
                pass

    def stop(self) -> None:
        """Arret demande (bouton Stop, sortie de l'application) : annule aussi les demarrages deja demandes."""
        with self._guard:
            self._gen += 1
            self._halt_locked()

    def _halt_locked(self) -> None:   # _guard tenu ; le nettoyage d'un demarrage passe ici sans changer de generation
        if self._watch is not None:
            self._watch.set()
            self._watch = None
        self.restarting = None
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

"""Revue R3 du runtime : plan CPU hors RAM jamais demarre automatiquement, Stop avant la premiere ligne SSE = annule (voie
directe comprise, Bonsai n'est pas sollicite), Stop honore pendant l'attente d'un slot occupe (avant les en-tetes), branches
d'un etat neuf lues sur un seul slot (S1 sur GPU, -np >= 2), retour du classifieur jamais "ready" sans Bonsai, demarrage en
file annule par Stop, Stop de l'utilisateur respecte par les fins d'installation, scripts : -c = contexte par slot x slots."""

import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import psutil
import pytest
from fastapi.testclient import TestClient

from jev_clone.backend_llamacpp import LlamaCppBackend
from jev_clone.engine import SystemOneEngine
from jev_clone.prophet import PROPHET_TURN
from prophet_studio import catalog
from prophet_studio import sessions as ss
from prophet_studio.config import Paths, Settings
from prophet_studio.hardware import GPU, HardwareInfo, enrich
from prophet_studio.planner import make_plan
from prophet_studio.server import Studio, build_app
from prophet_studio.supervisor import Runtime
from tests.test_fix_r1_runtime import WRAP
from tests.test_prophet import s1 as scripted_s1

TOKEN = "tok-r3rt"
H = {"X-Prophet-Token": TOKEN}
ROOT = Path(__file__).resolve().parents[1]
MSG = [{"role": "user", "content": "Say hi."}]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def until(cond, timeout: float = 30.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.05)
    return bool(cond())


def alive(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def hw(name="NVIDIA GeForce RTX 5060", total=8151, used=650):
    g = enrich(GPU(0, name, "nvidia", total, used, total - used, "580.88", "13.0", display_active=True))
    return HardwareInfo("Windows", "11", "amd64", "AMD Ryzen 7 7700", 8, 16, 32, 22, True, True, [g])


def studio(tmp_path: Path, monkeypatch, gpu: str = "NVIDIA GeForce RTX 5060:8151:650", demo: bool = True, **settings) -> Studio:
    monkeypatch.setenv("PROPHET_FAKE_GPU", gpu)
    monkeypatch.setenv("FAKE_LLAMA_TPS", "0")
    paths = Paths(tmp_path / "home")
    paths.settings.write_text(json.dumps({"workspace": str(tmp_path / "ws"), "s2_port": free_port(), "s1_port": free_port(),
                                          "voice": {"enabled": False}, **settings}))
    return Studio(paths, TOKEN, 7878, demo=demo)


class Chat:
    """Faux Bonsai (/v1/chat/completions diffuse) : compte les POST ; `hold` = secondes sans en-tetes (tache en file derriere
    un slot occupe : le fork ne repond qu'au lancement de la tache) ; `silent` = en-tetes puis silence ; sinon un flux court."""

    def __init__(self, hold: float = 0.0, silent: float = 0.0):
        self.hold, self.silent, self.posts, self.dropped = hold, silent, 0, threading.Event()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                outer.posts += 1
                hold, outer.hold = outer.hold, 0.0                      # seule la 1re requete attend un slot
                t0 = time.time()
                while time.time() - t0 < hold:                          # le client coupe : la tache en file est abandonnee
                    self.connection.settimeout(0.05)
                    try:
                        if self.connection.recv(1) == b"":
                            outer.dropped.set(); return
                    except (TimeoutError, socket.timeout):
                        pass
                    except OSError:
                        outer.dropped.set(); return
                self.connection.settimeout(None)
                self.send_response(200); self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
                if outer.silent:
                    time.sleep(outer.silent); return
                for line in ('data: {"choices":[{"index":0,"delta":{"content":"hi"}}]}',
                             'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}', "data: [DONE]"):
                    b = (line + "\n\n").encode()
                    self.wfile.write(f"{len(b):x}\r\n".encode() + b + b"\r\n")
                self.wfile.write(b"0\r\n\r\n")

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self):
        self.httpd.shutdown()


# ---- plan CPU hors RAM : jamais demarre automatiquement (lancement, fin d'installation), seulement sur accord explicite -------
def test_autostart_never_starts_a_cpu_plan_that_does_not_fit_in_ram(tmp_path, monkeypatch, big_gguf):
    st = studio(tmp_path, monkeypatch, gpu="NVIDIA GeForce GT 710:2048:14", onboarding_done=True)   # autostart : defaut
    assert st.settings.get().autostart_models
    huge = big_gguf(tmp_path / "huge-brain.gguf", int(st.hw.ram_total_gib * 4 * 2**30))   # 4 x la RAM (taille simulee)
    started: list = []
    st.runtime.start = lambda plan, ports, **kw: started.append(plan.s2.model_id) or True
    ids = []
    try:
        ids.append(st.installer.import_gguf(str(huge), "s2"))
        st.settings.update({"s2_model": ids[0]})
        p = st.plan()
        assert p.backend == "cpu" and p.rung == 0 and not p.fits and p.s2.model_id == ids[0]
        started.clear()
        with TestClient(build_app(st), base_url="http://127.0.0.1:7878") as c:   # lancement de l'application
            time.sleep(0.3)
            assert started == [] and "RAM insuffisante" in st.runtime.message
            st.runtime.state = "error"                                  # ex. un chargement precedent en echec
            other = tmp_path / "other.gguf"; other.write_bytes(b"GGUF" + b"\0" * 100)
            oid = c.post("/api/import", headers=H, json={"path": str(other), "role": "s1"}).json()["id"]
            ids.append(oid)
            c.delete(f"/api/installed/{oid}", headers=H)                # import puis suppression sans rapport
            time.sleep(0.3)
            assert started == []                                        # avant : deux demarrages de plus
            assert c.post("/api/runtime/start", headers=H).status_code == 200   # « Lancer quand meme » : accord explicite
            assert until(lambda: started == [ids[0]], 5)
        # dechargement partiel GPU : un manque de VRAM seul (fits False) n'empeche pas le demarrage automatique,
        # un manque de RAM (ram_short) si, comme pour un plan CPU
        gpu_vram_only = replace(p, backend="cuda", ram_short=False)
        st.plan = lambda *a, **k: gpu_vram_only
        assert st.may_autostart()
        gpu_ram_short = replace(p, backend="cuda", ram_short=True)
        st.plan = lambda *a, **k: gpu_ram_short
        assert not st.may_autostart()
    finally:
        for k in ids:
            catalog.CUSTOM.pop(k, None)


# ---- Stop avant la premiere ligne SSE : annule, pas une reponse vide terminee ----------------------------------------------------
class _Silent:
    headers: dict = {}

    def iter_lines(self, **kw):
        return iter(())                                                 # connexion coupee par le veilleur : fin sans erreur

    def close(self):
        pass


def test_stop_before_the_first_sse_line_is_reported_cancelled():
    r = LlamaCppBackend._read_stream(_Silent(), lambda d: None, lambda: True)
    assert r["cancelled"] is True and r["choices"][0]["finish_reason"] == "cancelled"   # avant : cancelled False, finish None
    done = LlamaCppBackend._read_stream(_Silent(), lambda d: None, lambda: False)
    assert done["cancelled"] is False                                   # sans Stop : inchange
    srv = Chat(silent=5)
    try:                                                                # Stop deja demande : Bonsai n'est pas sollicite
        out = LlamaCppBackend(srv.url, timeout=30).chat(MSG, on_delta=lambda d: None, should_stop=lambda: True)
        assert out["cancelled"] and out["choices"][0]["finish_reason"] == "cancelled" and srv.posts == 0
    finally:
        srv.close()


def test_direct_turn_stopped_during_the_s1_read_ends_cancelled_without_asking_bonsai(tmp_path):
    srv = Chat(silent=5)
    events: list[dict] = []
    svc = ss.AgentService(ss.SessionStore(tmp_path / "sessions"), events.append, urls=lambda: ("", ""),
                          settings=lambda: Settings(workspace=str(tmp_path / "ws")), ctx=lambda: 8192)
    sid = svc.store.create(str(tmp_path / "ws"))["id"]
    eng = scripted_s1(direct=0.95)                                      # S1 : voie directe
    first = eng.answer

    def answer(req):                                                    # Stop pendant la lecture de l'etat par S1
        eng.answer = first
        r = first(req)
        svc.cancel(sid)
        return r
    eng.answer = answer
    svc.engines = lambda: (eng, LlamaCppBackend(srv.url, max_workers=1, timeout=30))
    try:
        svc.submit(sid, "Capitale de l'Australie ?")
        assert svc.wait_idle(sid, 20)
        end = [e for e in events if e.get("type") == "turn.end"]
        assert end and end[0]["path"] == "direct" and end[0]["stopped_by"] == "cancelled"   # avant : "final", reponse vide
        assert srv.posts == 0                                           # avant : Bonsai recevait la requete malgre le Stop
        s = svc.store.load(sid)
        assert s["transcript"][-1]["stopped_by"] == "cancelled"
        assert not [m for m in s["history"] if m["role"] == "assistant" and not m["content"]]   # pas de reponse vide en contexte
    finally:
        srv.close()


# ---- Stop pendant l'attente d'un slot occupe (2e session) : honore avant les en-tetes --------------------------------------------
def test_stop_is_honored_while_the_request_waits_for_a_busy_slot():
    srv = Chat(hold=20)
    be = LlamaCppBackend(srv.url, max_workers=1, timeout=60)
    stop = threading.Event()
    try:
        threading.Timer(0.3, stop.set).start()
        t0 = time.time()
        out = be.chat(MSG, on_delta=lambda d: None, should_stop=stop.is_set)
        assert time.time() - t0 < 2 and out["cancelled"] and out["choices"][0]["finish_reason"] == "cancelled"   # avant : 20 s
        assert srv.dropped.wait(2)                                      # connexion coupee : la tache en file est abandonnee
        deltas: list = []                                               # la session suivante n'herite d'aucune connexion cassee
        ok = be.chat(MSG, on_delta=deltas.append, should_stop=lambda: False)
        assert not ok["cancelled"] and ok["choices"][0]["message"]["content"] == "hi" and ok["choices"][0]["finish_reason"] == "stop"
    finally:
        srv.close()


# ---- S1 sur GPU, -np 4 : un etat neuf est lu sur un seul slot, les branches suivantes passent sur son cache ---------------------
class Slots:
    """Faux llama-server a N slots, KV non unifie : une requete va sur le slot libre au plus long prefixe commun (comme le
    fork) ; prompt_n = ce que ce slot doit lire (4 caracteres par token). Compte les tokens lus et la concurrence."""

    def __init__(self, n: int = 4):
        self.cache, self.busy = [""] * n, [False] * n
        self.prefill, self.cur, self.max = 0, 0, 0
        cv = threading.Condition()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _json(self, obj):
                b = json.dumps(obj).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

            def do_GET(self):
                self._json({"model_path": "fake.gguf", "total_slots": n})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
                prompt = body["prompt"]

                def lcp(a: str) -> int:
                    return len(os.path.commonprefix([a, prompt]))
                with cv:
                    while all(outer.busy):
                        cv.wait()
                    free = [i for i in range(n) if not outer.busy[i]]
                    i = max(free, key=lambda k: lcp(outer.cache[k]))
                    hit = lcp(outer.cache[i])
                    outer.busy[i], outer.cur = True, outer.cur + 1
                    outer.max = max(outer.max, outer.cur)
                    todo = (len(prompt) - hit) // 4
                    outer.prefill += todo
                time.sleep(0.002 + todo * 2e-5)                        # prefill : ~50 k tokens/s
                with cv:
                    outer.cache[i], outer.busy[i], outer.cur = prompt, False, outer.cur - 1
                    cv.notify_all()
                labels = [x.strip('" ') for x in body["grammar"].split("::=", 1)[1].split("|")]
                self._json({"completion_probabilities": [{"top_probs": [{"id": k, "token": lab, "prob": 1 / len(labels)}
                                                                        for k, lab in enumerate(labels)]}],
                            "timings": {"prompt_n": todo, "cache_n": hit // 4}})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self):
        self.httpd.shutdown()


def test_gpu_s1_reads_a_fresh_prophet_state_once_not_once_per_slot():
    state = Studio._bench_state()                                       # etat Prophet neuf (~2 k tokens, nonce en tete)
    srv = Slots(4)
    try:
        eng = SystemOneEngine(LlamaCppBackend(srv.url, max_workers=4), model_name="t")
        seen: list = []
        be = eng.backend
        orig = be.score_branches
        be.score_branches = lambda prefix, branches: seen.append((len(prefix), [len(b.text) for b in branches])) or orig(prefix, branches)
        eng.answer({"state": state, "questions": PROPHET_TURN})
        (pre, lens), = seen
        ideal = (pre + sum(lens)) // 4                                  # l'etat une fois + chaque branche
        assert srv.prefill <= ideal * 1.05, (srv.prefill, ideal)        # avant : l'etat relu sur 3 slots de plus (~3 x)
        assert srv.max == 1                                             # branches en sequence sur le slot qui a l'etat
        small = {"state": "Customer: my payouts failed.", "questions": {
            q: {"type": "noul", "instructions": f"Question {q}: is this about {q}?"} for q in ("refund", "fraud", "login", "fees")}}
        srv.max = 0
        eng.answer(small)                                               # petit etat : le parallele reste rentable
        assert srv.max >= 2
    finally:
        srv.close()


# ---- retour du classifieur : jamais "ready" publie sans Bonsai ----------------------------------------------------------------
@pytest.fixture
def ctl(tmp_path, monkeypatch):
    d = tmp_path / "ctl"
    d.mkdir()
    (tmp_path / "wrap.py").write_text(WRAP)
    monkeypatch.setenv("R1_CTL", str(d))
    monkeypatch.setenv("FAKE_LLAMA_TPS", "0")
    return d


def wrapped_runtime(tmp_path: Path, have_s1: list) -> tuple[Runtime, list[dict]]:
    s1f, s2f = tmp_path / "s1.gguf", tmp_path / "s2.gguf"
    s1f.touch(); s2f.touch()
    events: list[dict] = []
    rt = Runtime(tmp_path / "logs", events.append, lambda mid: (s1f if have_s1[0] else None) if mid.startswith("ternary") else s2f,
                 lambda mid: None, lambda: [sys.executable, str(tmp_path / "wrap.py")], lambda: set())
    rt.watch_interval = 0.2
    return rt, events


def ready_without_bonsai(events: list[dict], dead: int) -> list[dict]:
    """Etats "ready" / "degraded" publies alors que Bonsai n'est pas pret (processus tue, ou relance en cours)."""
    return [e["runtime"] for e in events if e["type"] == "runtime.status" and e["runtime"]["state"] in ("ready", "degraded")
            and (e["runtime"]["servers"]["s2"]["pid"] == dead or e["runtime"]["servers"]["s2"]["state"] != "ready")]


def test_bonsai_dying_during_the_classifier_restart_is_seen_at_once_and_never_masked(tmp_path, ctl):
    rt, events = wrapped_runtime(tmp_path, [True])
    try:
        assert rt.start(make_plan(hw()), (free_port(), free_port())) and rt.state == "ready"
        (ctl / "slow_s1").write_text("4")                              # relance du classifieur lente
        rt.s1.proc.kill()
        assert until(lambda: rt.restarting == "s1" and rt.s1.proc is not None, 10), rt.public()
        n0, dead = len(events), rt.s2.proc.pid
        (ctl / "slow_s2").write_text("1")
        rt.s2.proc.kill(); rt.s2.proc.wait()
        # le watchdog surveille toujours Bonsai pendant la relance du classifieur (avant : bloque jusqu'a son retour, 4 s)
        assert until(lambda: rt.restarting == "s2" and rt.state == "starting", 2), rt.public()
        assert until(lambda: rt.state == "ready" and not rt.mono and rt.s1.alive() and rt.s2.alive() and rt.restarting is None, 30), rt.public()
        assert not ready_without_bonsai(events[n0:], dead)
        assert rt.restarts == {"s1": 1, "s2": 1} and rt.s2.proc.pid != dead
    finally:
        rt.stop()


def test_attached_classifier_does_not_hide_a_bonsai_restart(tmp_path, ctl):
    have = [False]
    rt, events = wrapped_runtime(tmp_path, have)                        # classifieur absent : mode mono
    try:
        assert rt.start(make_plan(hw("NVIDIA GeForce RTX 5060 Ti", 16311, 700)), (free_port(), free_port())) and rt.mono
        have[0] = True
        (ctl / "slow_s1").write_text("2")
        assert rt.can_attach_s1()
        th = threading.Thread(target=rt.attach_s1, daemon=True)
        th.start()
        assert until(lambda: rt.s1.proc is not None, 10)
        n0, dead = len(events), rt.s2.proc.pid
        (ctl / "slow_s2").write_text("4")                              # relance de Bonsai plus lente que l'arrivee du classifieur
        rt.s2.proc.kill(); rt.s2.proc.wait()
        th.join(30)
        assert until(lambda: rt.state == "ready" and not rt.mono and rt.s2.alive() and rt.restarting is None, 30), rt.public()
        assert not ready_without_bonsai(events[n0:], dead)              # avant : "ready ... duo" pendant le chargement de Bonsai
        assert rt.s1.alive() and rt.restarts["s2"] == 1
    finally:
        rt.stop()


# ---- demarrage en file derriere un autre : annule par Stop ------------------------------------------------------------------------
def test_a_start_queued_behind_a_running_start_does_not_run_after_stop(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_LLAMA_LOAD_S", "1.5")
    st = studio(tmp_path, monkeypatch)
    rt = st.runtime
    try:
        st.start_runtime()
        st.start_runtime()                                              # « Appliquer et redemarrer » pendant le demarrage
        assert until(lambda: rt.state == "starting", 10)
        time.sleep(0.3)
        rt.stop()
        pids = set()
        t0 = time.time()
        while time.time() - t0 < 6:                                     # avant : le 2e demarrage relancait tout (ready)
            pids |= rt.pids()
            assert rt.state == "stopped", rt.public()
            time.sleep(0.1)
        assert not rt.pids() and not any(alive(p) for p in pids)
        st.start_runtime()                                              # un nouveau demarrage apres le Stop fonctionne
        assert until(lambda: rt.state == "ready", 40), rt.public()
    finally:
        rt.stop()


# ---- Stop de l'utilisateur : une fin d'installation ne relance pas les modeles avant un Demarrer --------------------------------
def test_user_stop_is_remembered_until_the_next_explicit_start(tmp_path, monkeypatch):
    st = studio(tmp_path, monkeypatch)                                  # autostart_models : defaut
    rt = st.runtime
    with TestClient(build_app(st), base_url="http://127.0.0.1:7878") as c:
        try:
            assert until(lambda: rt.state == "ready", 40), rt.public()
            assert c.post("/api/runtime/stop", headers=H).status_code == 200
            st.installer._save()                                        # import, suppression, telechargement fini...
            time.sleep(1.5)
            assert rt.state == "stopped" and not rt.pids()              # avant : ready 6 s plus tard
            assert c.post("/api/runtime/start", headers=H).status_code == 200
            assert until(lambda: rt.state == "ready", 40), rt.public()
            rt.stop()                                                   # arret sans l'utilisateur (sortie, erreur...)
            st.installer._save()
            assert until(lambda: rt.state == "ready", 40), rt.public()  # l'installation redemarre comme avant
        finally:
            rt.stop()


# ---- scripts : -c = contexte par slot x slots, comme le planificateur ----------------------------------------------------------
@pytest.mark.skipif(sys.platform == "win32" or not shutil.which("sh"), reason="scripts POSIX")
@pytest.mark.parametrize("profile", sorted(p.name for p in (ROOT / "scripts" / "profiles").glob("*.env")))
def test_launch_scripts_give_every_slot_its_whole_context(tmp_path, profile):
    stub = tmp_path / "bin" / "cpu" / "llama-server"
    stub.parent.mkdir(parents=True)
    stub.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n")
    stub.chmod(0o755)
    gguf = tmp_path / "m.gguf"; gguf.touch()
    lines = (ROOT / "scripts" / "profiles" / profile).read_text().splitlines()
    prof = tmp_path / profile                                           # adaptateur LoRA du profil : un fichier present
    prof.write_text("\n".join(f"BONSAI_LORA={gguf}" if l.startswith("BONSAI_LORA=") else l for l in lines) + "\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("JEV_", "BONSAI_"))}
    env.update(PROFILE=str(prof), BIN_DIR=str(tmp_path / "bin"), MODELS=str(tmp_path / "models"), JEV_GGUF=str(gguf), BONSAI_GGUF=str(gguf))
    vals = {k: v.split("#")[0].strip() for k, v in (l.split("=", 1) for l in lines if "=" in l and not l.startswith("#"))}
    for script, pre in (("start_bonsai.sh", "BONSAI"), ("start_jev_clone.sh", "JEV")):
        out = subprocess.run(["sh", str(ROOT / "scripts" / script)], env=env, capture_output=True, text=True, timeout=30)
        assert out.returncode == 0, out.stderr
        argv = out.stdout.splitlines()
        if "-m" not in argv:                                            # JEV_MODE=mono : rien a lancer
            assert pre == "JEV" and vals.get("JEV_MODE") == "mono"
            continue
        ctx, np_ = int(vals[f"{pre}_CTX"]), int(vals[f"{pre}_NP"])
        assert argv[argv.index("-np") + 1] == str(np_)
        assert argv[argv.index("-c") + 1] == str(ctx * np_), (script, argv)   # avant : -c CTX, soit CTX / NP par slot
        assert ctx >= 4096, (profile, pre, ctx)                         # un etat Prophet (~2-3 k tokens) tient dans chaque slot
        if pre == "JEV":
            assert argv[argv.index("--cache-type-k") + 1] == "q8_0"     # KV du Studio (S1_KV) : moitie de f16

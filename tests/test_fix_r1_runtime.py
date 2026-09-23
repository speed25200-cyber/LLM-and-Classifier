"""Revue R1 du runtime : stop() pendant le premier demarrage, contexte garanti par slot (sans -kvu), relance de Bonsai
publiee comme transitoire, cause d'un arret conservee, watchdog robuste, delai depasse = processus arrete, attach_s1 apres
abandon, core.json controle sur tous les chemins, banc froid / chaud, llama-server dans le groupe de processus du coeur."""

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest
import requests

from jev_clone.backend_llamacpp import LlamaCppBackend
from jev_clone.engine import SystemOneEngine
from prophet_studio.catalog import get_model, kv_mib
from prophet_studio.config import Paths
from prophet_studio.demo.fake_llama import FakeLlama
from prophet_studio.hardware import GPU, HardwareInfo, enrich
from prophet_studio.planner import S1_CTX, SLOT_STATE_MIB, attach_plan, kv_ctx, make_plan, mono_plan
from prophet_studio.server import Studio, release_core_info
from prophet_studio.supervisor import MONO_MSG, Runtime, build_args

TOKEN = "t0k3n-r1rt"
H = {"X-Prophet-Token": TOKEN}
RTX5060 = "NVIDIA GeForce RTX 5060:8151:650:display"
ROOT = Path(__file__).resolve().parents[1]
REQ = {"state": "Customer: my payouts failed three times this week and nobody answered.",
       "questions": {"team": {"type": "choice", "instructions": "Which team?", "criteria": ["payments", "account", "other"]},
                     "escalate": {"type": "noul", "instructions": "Should this be escalated?"},
                     "refund": {"type": "noul", "instructions": "Is a refund requested?"},
                     "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["can wait", "today", "now"]}}}

# faux llama-server pilote par des fichiers de controle (R1_CTL) : chargement lent, arret (GGML_ASSERT) apres le demarrage
WRAP = r'''import os, runpy, sys, threading, time
from pathlib import Path
a = sys.argv[1:]
role = "s1" if a[a.index("--alias") + 1] == "systemone" else "s2"
ctl = Path(os.environ["R1_CTL"])
slow = ctl / f"slow_{role}"
if slow.exists():
    time.sleep(float(slow.read_text()))
crash = ctl / f"crash_{role}"
if crash.exists():
    def later(d=float(crash.read_text() or 1)):
        time.sleep(d)
        crash.unlink()                  # un seul arret : la relance tourne normalement
        print("load_tensors: done", flush=True)
        print("/src/ggml.c:1234: GGML_ASSERT(n_tokens <= n_batch) failed", flush=True)
        os._exit(1)
    threading.Thread(target=later, daemon=True).start()
sys.argv = ["fake_llama", *a]
runpy.run_module("prophet_studio.demo.fake_llama", run_name="__main__")
'''


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


def hw(name="NVIDIA GeForce RTX 5060", total=8151, used=650):
    g = enrich(GPU(0, name, "nvidia", total, used, total - used, "580.88", "13.0", display_active=True))
    return HardwareInfo("Windows", "11", "amd64", "AMD Ryzen 7 7700", 8, 16, 32, 22, True, True, [g])


def arg(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def alive(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def server_pids(events: list[dict]) -> set[int]:
    return {e["server"]["pid"] for e in events if e["type"] == "runtime.server" and e["server"]["pid"]}


@pytest.fixture
def ctl(tmp_path, monkeypatch):
    d = tmp_path / "ctl"
    d.mkdir()
    (tmp_path / "wrap.py").write_text(WRAP)
    monkeypatch.setenv("R1_CTL", str(d))
    monkeypatch.setenv("FAKE_LLAMA_TPS", "0")
    return d


def runtime(tmp_path: Path) -> tuple[Runtime, list[dict], Path]:
    s1f, s2f = tmp_path / "s1.gguf", tmp_path / "s2.gguf"
    s1f.touch(); s2f.touch()
    events: list[dict] = []
    rt = Runtime(tmp_path / "logs", events.append, lambda mid: s1f if mid.startswith("ternary") else s2f, lambda mid: None,
                 lambda: [sys.executable, str(tmp_path / "wrap.py")], lambda: set())
    rt.watch_interval = 0.2
    return rt, events, s1f


def start_bg(rt: Runtime, plan=None) -> tuple[threading.Thread, list]:
    res: list = []
    th = threading.Thread(target=lambda: res.append(rt.start(plan or make_plan(hw()), (free_port(), free_port()))), daemon=True)
    th.start()
    return th, res


def studio(tmp_path: Path, monkeypatch, demo: bool = True) -> Studio:
    monkeypatch.setenv("PROPHET_FAKE_GPU", RTX5060)
    monkeypatch.setenv("FAKE_LLAMA_TPS", "0")
    paths = Paths(tmp_path / "home")
    paths.settings.write_text(json.dumps({"workspace": str(tmp_path / "ws"), "s2_port": free_port(), "s1_port": free_port(),
                                          "voice": {"enabled": False}}))
    return Studio(paths, TOKEN, 7878, demo=demo)


# ---- stop() pendant le premier demarrage : rien ne ressuscite, aucun S1 orphelin -------------------------------------------
def test_stop_while_s1_loads_is_honored(tmp_path, ctl):
    rt, events, _ = runtime(tmp_path)
    (ctl / "slow_s1").write_text("4")
    th, res = start_bg(rt)
    try:
        assert until(lambda: rt.s1.proc is not None, 30)
        time.sleep(0.3)
        rt.stop()
        th.join(30)
        time.sleep(0.5)
        assert res == [False] and rt.state == "stopped" and not rt.mono, rt.public()
        assert rt._watch is None and not rt.can_attach_s1() and not rt.attach_s1()
        assert not rt.s1.alive() and not rt.s2.alive() and not any(alive(p) for p in server_pids(events))
        assert [e["runtime"]["state"] for e in events if e["type"] == "runtime.status"][-1] == "stopped"
    finally:
        rt.stop()


def test_stop_right_after_bonsai_is_ready_spawns_no_classifier(tmp_path, ctl):
    rt, events, _ = runtime(tmp_path)
    rt.vram_probe, rt.on_measure = (lambda: 1000.0), (lambda *a: None)   # GPU : mesure de VRAM (1 s) apres Bonsai
    th, res = start_bg(rt)
    try:
        assert until(lambda: rt.s2.state == "ready", 30)
        rt.stop()
        th.join(30)
        time.sleep(0.5)
        assert res == [False] and rt.state == "stopped" and rt.plan is None
        assert not [e for e in events if e["type"] == "runtime.server" and e["server"]["name"] == "s1" and e["server"]["state"] == "starting"]
        assert not any(alive(p) for p in server_pids(events))
    finally:
        rt.stop()


def test_stop_while_bonsai_loads_is_not_reported_as_an_error(tmp_path, ctl):
    rt, events, _ = runtime(tmp_path)
    (ctl / "slow_s2").write_text("4")
    th, res = start_bg(rt)
    try:
        assert until(lambda: rt.s2.proc is not None, 30)
        time.sleep(0.3)
        rt.stop()
        th.join(30)
        assert res == [False] and rt.state == "stopped", rt.message
        assert not [e for e in events if e["type"] == "runtime.status" and e["runtime"]["state"] == "error"]
    finally:
        rt.stop()


# ---- contexte garanti par slot : KV non unifie, -c = contexte par slot x slots, budget coherent -------------------------------
def test_parallel_slots_get_their_whole_context_in_args_and_budget():
    ti = make_plan(hw("NVIDIA GeForce RTX 5060 Ti", 16311, 700))
    a = build_args(ti.s1, Path("s1.gguf"), 8081)
    assert ti.s1.np == 4 and "-kvu" not in a and int(arg(a, "-c")) == S1_CTX * 4 == kv_ctx(ti.s1)
    assert ti.budget["s1_kv"] == round(kv_mib(get_model(ti.s1.model_id), S1_CTX * 4, "q8_0"))   # 4 x 8 k budgetes
    v = make_plan(hw(), "vitesse")                         # 8 Go : le contexte de Bonsai d'abord, puis les slots qui tiennent
    assert v.s1.device == "gpu" and v.s1.np == 2 and v.s2.ctx == 16384 and v.budget["free"] > 0
    assert arg(build_args(v.s1, Path("s1.gguf"), 8081), "-c") == str(2 * S1_CTX)
    assert any("2 slots de 8 k" in n for n in v.notes)
    m = mono_plan(ti)                                      # mode mono : 2e slot avec le contexte entier de la conversation
    a2 = build_args(m.s2, Path("s2.gguf"), 8080)
    assert m.s2.np == 2 and m.s2.ctx == ti.s2.ctx and "-kvu" not in a2 and int(arg(a2, "-c")) == 2 * ti.s2.ctx
    s2m = get_model(ti.s2.model_id)
    assert m.budget["s2_kv"] == round(ti.budget["s2_kv"] + kv_mib(s2m, ti.s2.ctx, ti.s2.kv_type))
    assert m.budget["free"] == round(ti.budget["free"] - kv_mib(s2m, ti.s2.ctx, ti.s2.kv_type) - SLOT_STATE_MIB)
    big = make_plan(hw("NVIDIA GeForce RTX 5060 Ti", 16311, 700), "contexte")   # 128 k : un 2e slot entier ne tient pas
    assert mono_plan(big).s2.np == 1                       # un seul slot : les lectures S1 attendent, rien n'echoue
    # le 2e slot a pris la VRAM du classifieur : s'il est installe apres coup, il demarre sur CPU
    tight = mono_plan(make_plan(hw(used=1200)))
    assert tight.s2.np == 2 and not any(k.startswith("s1_") for k in tight.budget)
    ap = attach_plan(tight)
    assert ap.s1.device == "cpu" and ap.s1.np == 1 and ap.s2 == tight.s2
    assert attach_plan(m) is m                             # VRAM libre suffisante : classifieur sur GPU comme prevu


def kv_server(slots: int = 4):
    """KV partage (-kvu) trop petit pour deux lectures : toute requete concurrente recoit l'erreur du fork."""
    base = FakeLlama().handler()
    st = {"cur": 0, "fail": 0}
    lock = threading.Lock()

    class Handler(base):
        def do_GET(self):
            if self.path.startswith("/props"):
                return self._json(200, {"model_path": "fake.gguf", "total_slots": slots, "default_generation_settings": {"n_ctx": 8192}})
            return super().do_GET()

        def do_POST(self):
            if not self.path.startswith("/completion"):
                return super().do_POST()
            with lock:
                st["cur"] += 1
                over = st["cur"] > 1
            try:
                time.sleep(0.05)
                if over:
                    self.rfile.read(int(self.headers.get("Content-Length") or 0))
                    st["fail"] += 1
                    return self._json(500, {"error": {"code": 500, "message": "Context size has been exceeded.", "type": "server_error"}})
                return super().do_POST()
            finally:
                with lock:
                    st["cur"] -= 1

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, st


def test_backend_retries_branches_one_by_one_when_the_shared_kv_is_full():
    srv, st = kv_server()
    ref, _ = kv_server(slots=1)
    try:
        eng = SystemOneEngine(LlamaCppBackend(f"http://127.0.0.1:{srv.server_address[1]}", max_workers=4), model_name="t")
        r = eng.answer(REQ)
        assert st["fail"] > 0                              # les lectures paralleles ont bien deborde
        one = SystemOneEngine(LlamaCppBackend(f"http://127.0.0.1:{ref.server_address[1]}", max_workers=4), model_name="t").answer(REQ)
        assert {k: v.model_dump() for k, v in r.answers.items()} == {k: v.model_dump() for k, v in one.answers.items()}
    finally:
        srv.shutdown(); ref.shutdown()


# ---- relance de Bonsai : etat transitoire, jamais "error" ; une fin d'installation ne relance rien ----------------------------
def test_bonsai_auto_restart_is_transient_not_an_error(tmp_path, monkeypatch):
    st = studio(tmp_path, monkeypatch)
    rt = st.runtime
    rt.watch_interval = 0.2
    events: list[dict] = []
    emit = rt.emit
    rt.emit = lambda e: (events.append(e), emit(e))
    rt.s1.emit = rt.s2.emit = rt.emit
    try:
        st.start_runtime()
        assert until(lambda: rt.state in ("ready", "degraded", "error"), 40) and rt.state == "ready", rt.message
        s1_pid = rt.s1.proc.pid
        monkeypatch.setenv("FAKE_LLAMA_LOAD_S", "2")      # la relance de Bonsai prend du temps
        rt.s2.proc.kill()
        assert until(lambda: rt.restarting == "s2", 15), rt.public()
        pub = rt.public()
        assert pub["state"] == "starting" and pub["restarting"] == "s2" and "redemarrage automatique" in pub["message"]
        st._autostart_after_install()                      # une installation qui se termine pendant la relance
        assert until(lambda: rt.state == "ready" and rt.restarting is None, 40), rt.public()
        assert rt.s1.proc.pid == s1_pid and rt.restarts == {"s1": 0, "s2": 1} and "redemarre" in rt.message
        assert not [e for e in events if e["type"] == "runtime.status" and e["runtime"]["state"] == "error"]
    finally:
        rt.stop()


# ---- la cause d'un arret survit a la relance automatique ---------------------------------------------------------------------
def test_crash_cause_survives_the_automatic_restart(tmp_path, ctl):
    rt, events, _ = runtime(tmp_path)
    (ctl / "crash_s1").write_text("1.0")
    (ctl / "crash_s2").write_text("1.0")
    try:
        assert rt.start(make_plan(hw()), (free_port(), free_port())) and rt.state == "ready"
        assert until(lambda: rt.restarts == {"s1": 1, "s2": 1} and rt.state == "ready" and rt.restarting is None
                     and rt.s1.alive() and rt.s2.alive(), 40), rt.public()
        for name in ("s1", "s2"):
            srv = rt.public()["servers"][name]
            assert "GGML_ASSERT" in srv["last_error"]["reason"] and "(code de sortie 1)" in srv["last_error"]["reason"], srv
            assert any("GGML_ASSERT" in l for l in srv["last_error"]["tail"])
            assert "GGML_ASSERT" in (tmp_path / "logs" / f"{name}.prev.log").read_text()   # journal du lancement arrete
            assert "GGML_ASSERT" not in (tmp_path / "logs" / f"{name}.log").read_text()
        msgs = [e["runtime"]["message"] for e in events if e["type"] == "runtime.status"]
        assert any("redemarrage" in m and "GGML_ASSERT" in m for m in msgs)
        assert any(m.startswith("classifieur redemarre") and "GGML_ASSERT" in m for m in msgs)
        assert any(m.startswith("Bonsai redemarre") and "GGML_ASSERT" in m for m in msgs)
        assert not any("server is listening" in m for m in msgs)   # une cause lisible, pas la fin brute du journal
    finally:
        rt.stop()


# ---- watchdog : une relance qui leve (binaire disparu, erreur imprevue) ne l'arrete pas -------------------------------------
def test_watchdog_survives_a_failed_spawn_and_still_sees_bonsai_die(tmp_path, ctl):
    rt, events, _ = runtime(tmp_path)
    try:
        assert rt.start(make_plan(hw()), (free_port(), free_port())) and rt.state == "ready"
        rt._cmd = [str(tmp_path / "bin" / "llama-server")]  # runtime en cours de reinstallation : binaire absent
        rt.s1.proc.kill()
        assert until(lambda: rt.mono and rt.state == "degraded" and rt.restarting is None, 20), rt.public()
        assert rt.message.startswith(MONO_MSG) and "lancement impossible" in rt.message
        assert rt.s1.state == "crashed" and "lancement impossible" in rt.s1.last_error["reason"]
        rt.s2.proc.kill()                                  # plus tard, Bonsai s'arrete : toujours detecte
        assert until(lambda: rt.state == "error", 20) and "Bonsai" in rt.message, rt.public()
    finally:
        rt.stop()


def test_watchdog_survives_an_unexpected_exception(tmp_path, ctl):
    rt, events, _ = runtime(tmp_path)
    try:
        assert rt.start(make_plan(hw()), (free_port(), free_port())) and rt.state == "ready"

        def boom(*a):
            raise RuntimeError("boom")
        rt._restore_s1 = boom
        rt.s1.proc.kill()
        assert until(lambda: "surveillance" in rt.message and "boom" in rt.message, 20), rt.public()
        assert rt.state == "degraded" and rt.mono and rt.restarting is None
        pid = rt.s2.proc.pid
        rt.s2.proc.kill()
        assert until(lambda: rt.restarts["s2"] == 1 and rt.state == "degraded" and rt.s2.alive() and rt.s2.proc.pid != pid, 30), rt.public()
    finally:
        rt.stop()


# ---- delai de demarrage depasse : le processus est arrete ---------------------------------------------------------------------
def test_start_timeout_kills_the_spawned_process(tmp_path, ctl):
    rt, events, _ = runtime(tmp_path)
    rt.s1_timeout = 2.0
    (ctl / "slow_s1").write_text("60")
    try:
        assert rt.start(make_plan(hw()), (free_port(), free_port()))
        assert rt.mono and rt.state == "degraded" and rt.s1.state == "crashed" and rt.s1.error == "delai de demarrage depasse"
        assert rt.s1.proc is not None and rt.s1.proc.poll() is not None and not alive(rt.s1.proc.pid)
        assert "delai de demarrage depasse" in rt.message
        rt.stop()
        rt.s2_timeout = 2.0                                # meme chose pour Bonsai : demarrage en echec, processus arrete
        (ctl / "slow_s2").write_text("60")
        assert not rt.start(make_plan(hw()), (free_port(), free_port()))
        assert rt.state == "error" and "delai" in rt.message and rt.s2.proc is not None and not alive(rt.s2.proc.pid)
    finally:
        rt.stop()


# ---- attach_s1 respecte l'abandon du watchdog ; un fichier change le fait revenir ---------------------------------------------
def test_attach_respects_the_watchdog_give_up_until_the_classifier_file_changes(tmp_path, monkeypatch):
    st = studio(tmp_path, monkeypatch)
    rt = st.runtime
    rt.watch_interval = 0.2
    try:
        st.start_runtime()
        assert until(lambda: rt.state in ("ready", "degraded", "error"), 40) and rt.state == "ready", rt.message
        rt.s1.proc.kill()
        assert until(lambda: rt.restarts["s1"] == 1 and rt.state == "ready" and rt.s1.alive(), 30), rt.public()
        rt.s1.proc.kill()                                  # 2e arret : le watchdog abandonne
        assert until(lambda: rt.mono and rt.state == "degraded", 30), rt.public()
        assert not rt.can_attach_s1()
        st.installer._save()                               # installation sans rapport (voix, autre modele...)
        time.sleep(1.5)
        assert rt.mono and rt.s1.proc is None and rt.restarts["s1"] == 1
        f = st.installer.model_path(rt.plan.s1.model_id)
        t = time.time() + 5
        os.utime(f, (t, t))                                # classifieur reinstalle / remplace
        assert rt.can_attach_s1()
        st.installer._save()
        assert until(lambda: not rt.mono and rt.state == "ready" and rt.s1.alive(), 30), rt.public()
    finally:
        rt.stop()


# ---- core.json : controle du pid sur tous les chemins de sortie -------------------------------------------------------------
def test_release_core_info_checks_the_pid(tmp_path):
    paths = Paths(tmp_path / "h")
    paths.core_info.write_text(json.dumps({"pid": os.getpid() + 1}))
    release_core_info(paths)
    assert paths.core_info.exists()
    paths.core_info.write_text("{")
    release_core_info(paths)
    assert paths.core_info.exists()
    paths.core_info.write_text(json.dumps({"pid": os.getpid()}))
    release_core_info(paths)
    assert not paths.core_info.exists()
    release_core_info(paths)                               # deja absent : rien


def launch_core(data: Path, *extra: str) -> tuple[subprocess.Popen, dict]:
    paths = Paths(data)
    paths.settings.write_text(json.dumps({"s2_port": free_port(), "s1_port": free_port(), "voice": {"enabled": False},
                                          "workspace": str(data / "ws")}))
    env = dict(os.environ, PROPHET_FAKE_GPU=RTX5060, FAKE_LLAMA_TPS="0")
    p = subprocess.Popen([sys.executable, "-m", "prophet_studio", "--no-browser", "--port", "0", "--data-dir", str(data),
                          "--token", TOKEN, *extra], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    for line in p.stdout:
        if line.startswith("PROPHET_READY"):
            return p, json.loads(line.split(" ", 1)[1])
    raise AssertionError("le coeur n'a pas demarre")


@pytest.mark.skipif(sys.platform == "win32", reason="SIGINT vers un processus : Unix")
def test_ctrl_c_keeps_another_instances_core_json_and_stops_the_models(tmp_path):
    data = tmp_path / "home"
    p, info = launch_core(data, "--demo")
    pids: set[int] = set()
    try:
        url = info["url"]

        def servers():
            r = requests.get(f"{url}/api/state", headers=H, timeout=5).json()["runtime"]
            return r if r["state"] in ("ready", "degraded") else None
        assert until(servers, 40)
        pids = {s["pid"] for s in servers()["servers"].values() if s["pid"]}
        assert len(pids) == 2 and all(alive(x) for x in pids)
        core = data / "core.json"
        other = {"url": "http://127.0.0.1:65000", "port": 65000, "token": "other", "pid": p.pid + 99999}
        core.write_text(json.dumps(other))                 # une autre instance (--parent-pid) a reecrit core.json
        p.send_signal(signal.SIGINT)
        p.communicate(timeout=40)
        assert json.loads(core.read_text())["pid"] == other["pid"]
        assert until(lambda: not any(alive(x) for x in pids), 10)
    finally:
        if p.poll() is None:
            p.kill()
        for x in pids:
            if alive(x):
                os.kill(x, signal.SIGKILL)


# ---- Unix : les llama-server meurent avec le groupe de processus du coeur (SIGKILL de la coquille) --------------------------
CORE = r'''import json, socket, sys, time
from pathlib import Path
from prophet_studio.hardware import GPU, HardwareInfo, enrich
from prophet_studio.planner import make_plan
from prophet_studio.supervisor import Runtime
def port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); return s.getsockname()[1]
m = Path(sys.argv[1]) / "m.gguf"; m.touch()
g = enrich(GPU(0, "NVIDIA GeForce RTX 5060", "nvidia", 8151, 650, 7501, "580.88", "13.0", display_active=True))
hw = HardwareInfo("Windows", "11", "amd64", "x", 8, 16, 32, 22, True, True, [g])
rt = Runtime(Path(sys.argv[1]) / "logs", lambda e: None, lambda mid: m, lambda mid: None,
             lambda: [sys.executable, "-m", "prophet_studio.demo.fake_llama"], lambda: set())
assert rt.start(make_plan(hw), (port(), port()))
print("PIDS " + json.dumps(sorted(rt.pids())), flush=True)
time.sleep(120)
'''


@pytest.mark.skipif(sys.platform == "win32", reason="groupes de processus Unix (Windows : Job Object de la coquille)")
def test_llama_servers_die_with_the_core_process_group(tmp_path):
    p = subprocess.Popen([sys.executable, "-c", CORE, str(tmp_path)], cwd=ROOT, stdout=subprocess.PIPE, text=True,
                         start_new_session=True)       # comme la coquille de bureau : le coeur mene son groupe
    pids: list[int] = []
    try:
        for line in p.stdout:
            if line.startswith("PIDS "):
                pids = json.loads(line[5:])
                break
        assert len(pids) == 2 and all(alive(x) for x in pids)
        os.killpg(p.pid, signal.SIGKILL)                   # repli de la coquille : SIGKILL sur tout le groupe
        p.wait(10)
        assert until(lambda: not any(alive(x) for x in pids), 10), [x for x in pids if alive(x)]
    finally:
        if p.poll() is None:
            p.kill()
        for x in pids:
            if alive(x):
                os.kill(x, signal.SIGKILL)


# ---- banc : serie a froid (etat Prophet neuf) a cote de la serie en cache, etiquetees ---------------------------------------
def test_bench_measures_fresh_prophet_states_next_to_the_cached_one(tmp_path, monkeypatch):
    from jev_clone.prophet import PROPHET_TURN
    st = studio(tmp_path, monkeypatch, demo=False)
    seen: list[str] = []

    class S1:
        def answer(self, req):
            key = json.dumps(req["state"], sort_keys=True)
            cold = key not in seen
            seen.append(key)
            time.sleep(0.03 if cold else 0.002)
            return SimpleNamespace(usage=SimpleNamespace(question_tokens=2100 if cold else 40), questions=req["questions"])

    class S2:
        def chat(self, *a, **k):
            return {"timings": {"predicted_per_second": 51.0, "prompt_per_second": 900.0}}
    asked: list = []
    s1 = S1()
    orig = s1.answer
    s1.answer = lambda req: (asked.append(req["questions"]), orig(req))[1]
    st.agent.engines = lambda: (s1, S2())
    out = st.bench()
    assert set(out["s1_series"]) == {"s1_p50_ms", "s1_cold", "s1_warm"}
    assert out["s1_cold_p50_ms"] > out["s1_warm_p50_ms"] and out["s1_cold_p95_ms"] >= out["s1_cold_p50_ms"]
    assert out["s1_cold_prefill_tokens"] == 2100 and out["s1_warm_prefill_tokens"] == 40
    assert "s1_p50_ms" in out and "s1_p95_ms" in out and out["s2_tok_s"] == 51.0   # champs existants conserves
    prophet = [json.loads(k) for k in seen if k.startswith("{")]
    assert len(prophet) == 6 and len({json.dumps(s, sort_keys=True) for s in prophet}) == 3   # 3 etats neufs, relus une fois
    heads = {s["request"][:34] for s in prophet}
    assert len(heads) == 3                                 # nonce en tete : aucun prefixe commun a reutiliser
    assert all(len(s["workspace_files"]) == 60 and len(s["recent_turns"]) == 4 and len(s["request"]) <= 2500 for s in prophet)
    assert sum(q == PROPHET_TURN for q in asked) == 6
    assert json.loads((st.paths.runs / "bench.jsonl").read_text().splitlines()[-1])["s1_cold_p50_ms"] == out["s1_cold_p50_ms"]

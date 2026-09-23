"""Robustesse du runtime System One / System Two : watchdog, isolation GPU du classifieur sur CPU, schema de requetes
S1 (slots), debordement de contexte (corps d'erreur), mode mono (2e slot, retour au duo), core.json au SIGTERM."""

import json
import os
import socket
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
import requests
from fastapi.testclient import TestClient

from jev_clone.backend_llamacpp import LlamaCppBackend
from jev_clone.engine import SystemOneEngine
from prophet_studio.config import Paths
from prophet_studio.demo.fake_llama import FakeLlama
from prophet_studio.hardware import GPU, HardwareInfo, enrich
from prophet_studio.planner import degrade, make_plan, mono_plan, s1_on_cpu
from prophet_studio.server import Studio, build_app
from prophet_studio.supervisor import Runtime, build_args, server_env

TOKEN = "t0k3n-s1rt"
H = {"X-Prophet-Token": TOKEN}
RTX5060 = "NVIDIA GeForce RTX 5060:8151:650:display"
RTX5060TI = "NVIDIA GeForce RTX 5060 Ti:16311:700"
REQ = {"state": "Customer: my payouts failed three times this week and nobody answered.",
       "questions": {"team": {"type": "choice", "instructions": "Which team?", "criteria": ["payments", "account", "other"]},
                     "escalate": {"type": "noul", "instructions": "Should this be escalated?"},
                     "refund": {"type": "noul", "instructions": "Is a refund requested?"},
                     "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["can wait", "today", "now"]}}}


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


def studio(tmp_path: Path, monkeypatch, gpu: str, demo: bool = True) -> Studio:
    monkeypatch.setenv("PROPHET_FAKE_GPU", gpu)
    monkeypatch.setenv("FAKE_LLAMA_TPS", "0")
    paths = Paths(tmp_path / "home")
    paths.settings.write_text(json.dumps({"workspace": str(tmp_path / "ws"), "s2_port": free_port(), "s1_port": free_port(),
                                          "voice": {"enabled": False}}))
    return Studio(paths, TOKEN, 7878, demo=demo)


# ---- planificateur / arguments : S1 sur CPU = un slot, rien sur le GPU ; S1 sur GPU = 4 slots a KV unifie ---------------
def test_s1_on_cpu_single_slot_isolated_from_gpu_and_gpu_s1_unified_kv():
    p = make_plan(hw())                                    # RTX 5060 8 Go, equilibre : classifieur sur CPU
    assert p.s1.device == "cpu" and p.s1.np == 1
    a1 = build_args(p.s1, Path("s1.gguf"), 8081)
    assert arg(a1, "-np") == "1" and arg(a1, "--device") == "none" and "-kvu" not in a1
    assert server_env(p.s1) == {"CUDA_VISIBLE_DEVICES": "-1", "HIP_VISIBLE_DEVICES": "-1"}
    a2 = build_args(p.s2, Path("s2.gguf"), 8080)
    assert "--device" not in a2 and server_env(p.s2) == {} and arg(a2, "-np") == "1"
    assert any("prefill" in n and "Estimations" in n for n in p.notes)   # latence CPU qualifiee (etat neuf vs en cache)
    ti = make_plan(hw("NVIDIA GeForce RTX 5060 Ti", 16311, 700))       # 16 Go : classifieur sur GPU
    assert ti.s1.device == "gpu" and ti.s1.np == 4
    a = build_args(ti.s1, Path("s1.gguf"), 8081)
    assert "-kvu" in a and "--no-cache-idle-slots" in a and "--device" not in a and server_env(ti.s1) == {}
    assert make_plan(HardwareInfo("Linux", "6", "x86_64", "i7", 8, 16, 16, 12, True, False, [])).s1.np == 1   # tout CPU


def test_s1_ladder_and_mono_plan():
    ti = make_plan(hw("NVIDIA GeForce RTX 5060 Ti", 16311, 700))
    c = s1_on_cpu(ti)                                      # OOM du classifieur : directement sur CPU, Bonsai inchange
    assert (c.s1.device, c.s1.ngl, c.s1.np) == ("cpu", 0, 1) and c.s2 == ti.s2 and c.s2.mmproj == "gpu"
    assert not any(k.startswith("s1_") for k in c.budget) and c.rung == ti.rung + 1
    d = degrade(degrade(ti))                               # l'echelle de Bonsai deplace aussi le classifieur avec un slot
    assert d.s1.device == "cpu" and d.s1.np == 1
    m = mono_plan(ti)                                      # classifieur prevu sur GPU mais absent : Bonsai prend 2 slots
    assert m.s2.np == 2 and m.s1 == ti.s1
    a = build_args(m.s2, Path("s2.gguf"), 8080)
    assert arg(a, "-np") == "2" and "-kvu" in a and "--no-cache-idle-slots" in a
    assert degrade(m).s2.np == 1 and degrade(m).s2.mmproj == "gpu"   # 1er cran anti-OOM : on rend le 2e slot
    assert mono_plan(make_plan(hw())).s2.np == 1           # 8 Go, classifieur prevu sur CPU : pas de marge VRAM
    cpu = make_plan(HardwareInfo("Linux", "6", "x86_64", "i7", 8, 16, 16, 12, True, False, []))
    assert mono_plan(cpu).s2.np == 2                       # tout en RAM


# ---- OOM du classifieur sur GPU : directement sur CPU, sans GPU visible ----------------------------------------------------
WRAP = """import os, runpy, sys
a = sys.argv[1:]
print("env CUDA_VISIBLE_DEVICES=" + os.environ.get("CUDA_VISIBLE_DEVICES", "<absent>"), flush=True)
if a[a.index("--alias") + 1] == "systemone" and a[a.index("-ngl") + 1] != "0":
    print("ggml_backend_cuda_buffer_type_alloc_buffer: cudaMalloc failed: out of memory", flush=True)
    sys.exit(1)
sys.argv = ["fake_llama", *a]
runpy.run_module("prophet_studio.demo.fake_llama", run_name="__main__")
"""


def test_s1_gpu_oom_goes_straight_to_cpu_without_touching_bonsai(tmp_path):
    script = tmp_path / "oom_s1_gpu.py"
    script.write_text(WRAP)
    model = tmp_path / "m.gguf"; model.touch()
    events = []
    rt = Runtime(tmp_path / "logs", events.append, lambda mid: model, lambda mid: None, lambda: [sys.executable, str(script)],
                 lambda: set())
    plan = make_plan(hw("NVIDIA GeForce RTX 5060 Ti", 16311, 700))
    assert plan.s1.device == "gpu" and plan.s2.mmproj == "gpu"
    try:
        assert rt.start(plan, (free_port(), free_port()))
        assert rt.state == "ready" and not rt.mono, rt.message
        assert rt.plan.s1.device == "cpu" and rt.plan.s1.np == 1
        assert rt.plan.s2.mmproj == "gpu"                   # aucun cran de Bonsai consomme pour le classifieur
        s1_launches = [e for e in events if e["type"] == "runtime.server" and e["server"]["name"] == "s1" and e["server"]["state"] == "starting"]
        assert len(s1_launches) == 2                        # GPU (OOM) puis CPU
        assert arg(rt.s1.argv, "--device") == "none" and "env CUDA_VISIBLE_DEVICES=-1" in rt.logs("s1")
        assert "env CUDA_VISIBLE_DEVICES=-1" not in rt.logs("s2")
        assert any(e["type"] == "runtime.degraded" and "classifieur" in e["note"] for e in events)
    finally:
        rt.stop()


# ---- watchdog : S1 relance une fois, puis mode mono ; un tour Prophet obtient ses reponses S1 de Bonsai ----------------------
def test_watchdog_restarts_s1_once_then_mono_serves_s1_from_bonsai(tmp_path, monkeypatch):
    st = studio(tmp_path, monkeypatch, RTX5060)
    st.runtime.watch_interval = 0.2
    events = []
    emit = st.runtime.emit
    st.runtime.emit = lambda e: (events.append(e), emit(e))
    rt = st.runtime
    with TestClient(build_app(st), base_url="http://127.0.0.1:7878") as c:
        try:
            assert until(lambda: rt.state in ("ready", "degraded", "error"), 40) and rt.state == "ready", rt.message
            pid = rt.s1.proc.pid
            rt.s1.proc.kill()                               # crash de S1 apres le demarrage
            assert until(lambda: rt.restarts["s1"] == 1 and rt.state == "ready" and rt.s1.proc is not None
                         and rt.s1.proc.pid != pid and not rt.mono, 30), rt.public()
            assert any(e["type"] == "runtime.status" and e["runtime"]["mono"] and e["runtime"]["state"] == "degraded" for e in events)
            assert rt.s1.state == "ready" and rt.s1_url != rt.s2.url
            rt.s1.proc.kill()                               # 2e crash : plus de relance, Bonsai repond aux questions S1
            assert until(lambda: rt.mono and rt.state == "degraded", 30), rt.public()
            s = c.get("/api/state", headers=H).json()["runtime"]
            assert s["mono"] and s["state"] == "degraded" and s["s1_url"] == s["s2_url"] and s["restarts"]["s1"] == 1
            d = c.post("/v1/systemone", headers=H, json={"state": "Refund please", "questions": {
                "refund": {"type": "noul", "instructions": "Is a refund requested?"}}})
            assert d.status_code == 200 and d.json()["mono"] is True and d.json()["model"] == rt.plan.s2.model_id
            sid = c.post("/api/sessions", headers=H, json={}).json()["id"]
            assert c.post(f"/api/sessions/{sid}/turn", headers=H, json={"text": "Cree une app minuteur pomodoro"}).status_code == 200
            assert st.agent.wait_idle(sid, 40)
            item = c.get(f"/api/sessions/{sid}", headers=H).json()["transcript"][-1]
            assert item["status"] == "done" and "index.html" in item["response"]
            assert "s1_error" not in item["s1"]["pre"] and item["s1"]["path"] == "agent"   # decisions S1 lues sur Bonsai
            assert item["verification"] is not None
        finally:
            rt.stop()


def test_watchdog_restarts_bonsai_once(tmp_path):
    model = tmp_path / "m.gguf"; model.touch()
    rt = Runtime(tmp_path / "logs", lambda e: None, lambda mid: model, lambda mid: None,
                 lambda: [sys.executable, "-m", "prophet_studio.demo.fake_llama"], lambda: set())
    rt.watch_interval = 0.2
    try:
        assert rt.start(make_plan(hw()), (free_port(), free_port())) and rt.state == "ready"
        pid = rt.s2.proc.pid
        rt.s2.proc.kill()
        assert until(lambda: rt.restarts["s2"] == 1 and rt.state == "ready" and rt.s2.proc is not None and rt.s2.proc.pid != pid, 30), rt.public()
        assert requests.get(f"{rt.s2.url}/health", timeout=5).ok
        rt.s2.proc.kill()                                   # 2e arret : etat error explicite, plus de relance
        assert until(lambda: rt.state == "error", 30) and "Bonsai" in rt.message
    finally:
        rt.stop()
    assert rt.state == "stopped" and rt.s1.proc is None and rt.s2.proc is None


def test_systemone_and_bench_return_503_when_backend_unreachable(tmp_path, monkeypatch):
    st = studio(tmp_path, monkeypatch, RTX5060, demo=False)
    with TestClient(build_app(st), base_url="http://127.0.0.1:7878", raise_server_exceptions=False) as c:
        st.runtime.state = "ready"                          # etat encore "ready", serveurs morts (ports fermes)
        st.runtime.s1.port, st.runtime.s2.port = free_port(), free_port()
        r = c.post("/v1/systemone", headers=H, json={"state": "x", "questions": {"q": {"type": "noul", "instructions": "?"}}})
        assert r.status_code == 503 and "System One ne repond pas" in r.json()["detail"]
        b = c.post("/api/bench", headers=H)
        assert b.status_code == 503 and "ne repond pas" in b.json()["detail"]
        st.runtime.state = "stopped"


# ---- mode mono des le depart : 2e slot pour Bonsai, puis retour au duo quand le classifieur est installe ----------------------
def test_mono_at_start_two_slots_then_duo_restored_after_s1_install(tmp_path, monkeypatch):
    st = studio(tmp_path, monkeypatch, RTX5060TI)
    rt = st.runtime
    s1_id = st.plan().s1.model_id
    saved = st.installer.registry["models"].pop(s1_id)     # classifieur pas (encore) installe
    try:
        st.start_runtime()
        assert until(lambda: rt.state in ("ready", "degraded", "error"), 40)
        assert rt.state == "degraded" and rt.mono and rt.s1_url == rt.s2.url, rt.message
        assert arg(rt.s2.argv, "-np") == "2" and "-kvu" in rt.s2.argv and rt.plan.s2.np == 2
        pid = rt.s2.proc.pid
        st.installer.registry["models"][s1_id] = saved
        st.installer._save()                                # fin d'installation : rappel on_change
        assert until(lambda: not rt.mono and rt.state == "ready", 30), rt.public()
        assert rt.s2.proc.pid == pid                        # Bonsai n'a pas ete relance
        assert rt.s1.state == "ready" and rt.s1_url == rt.s1.url != rt.s2.url and not rt.can_attach_s1()
    finally:
        rt.stop()


# ---- backend : jamais plus de branches en parallele que de slots ; corps d'erreur du serveur ----------------------------------
def slot_server(slots: int, fail: str = ""):
    base = FakeLlama().handler()
    stats = {"props": 0, "cur": 0, "max": 0}
    lock = threading.Lock()

    class Handler(base):
        def do_GET(self):
            if self.path.startswith("/props"):
                stats["props"] += 1
                return self._json(200, {"model_path": "fake.gguf", "total_slots": slots})
            return super().do_GET()

        def do_POST(self):
            if fail and self.path.startswith("/completion"):
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                return self._json(400, {"error": {"code": 400, "message": fail, "type": "exceed_context_size_error"}})
            with lock:
                stats["cur"] += 1
                stats["max"] = max(stats["max"], stats["cur"])
            time.sleep(0.05)
            with lock:
                stats["cur"] -= 1
            return super().do_POST()

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, stats


def test_backend_caps_parallel_branches_to_server_slots():
    one, st1 = slot_server(1)
    four, st4 = slot_server(4)
    try:
        e1 = SystemOneEngine(LlamaCppBackend(f"http://127.0.0.1:{one.server_address[1]}", max_workers=4), model_name="t")
        e4 = SystemOneEngine(LlamaCppBackend(f"http://127.0.0.1:{four.server_address[1]}", max_workers=4), model_name="t")
        r1, r1b, r4 = e1.answer(REQ), e1.answer(REQ), e4.answer(REQ)
        assert st1["max"] == 1 and st1["props"] == 1        # un slot (S1 sur CPU) : en sequence ; /props lu une seule fois
        assert st4["max"] >= 2                              # 4 slots (S1 sur GPU) : parallele conserve
        a1 = {k: v.model_dump() for k, v in r1.answers.items()}
        assert a1 == {k: v.model_dump() for k, v in r4.answers.items()} == {k: v.model_dump() for k, v in r1b.answers.items()}
    finally:
        one.shutdown(); four.shutdown()


def test_backend_error_body_reaches_the_caller():
    msg = "the request exceeds the available context size, try increasing it"
    srv, _ = slot_server(1, fail=msg)
    try:
        eng = SystemOneEngine(LlamaCppBackend(f"http://127.0.0.1:{srv.server_address[1]}"), model_name="t")
        with pytest.raises(requests.HTTPError) as ei:
            eng.answer(REQ)
        assert msg in str(ei.value)[:200] and str(ei.value).startswith("llama-server 400")   # [:200] = puce s1_error
    finally:
        srv.shutdown()


# ---- core.json retire par le lifespan (SIGTERM : atexit ne tourne pas), seulement s'il est a nous ----------------------------
def test_lifespan_removes_core_json_only_when_it_describes_this_process(tmp_path, monkeypatch):
    st = studio(tmp_path, monkeypatch, RTX5060, demo=False)
    st.paths.core_info.write_text(json.dumps({"url": "http://127.0.0.1:7878", "pid": os.getpid()}), encoding="utf-8")
    with TestClient(build_app(st), base_url="http://127.0.0.1:7878") as c:
        assert c.get("/api/health").json()["app"] == "prophet-studio"
    assert not st.paths.core_info.exists()
    st.paths.core_info.write_text(json.dumps({"url": "http://127.0.0.1:7979", "pid": os.getpid() + 1}), encoding="utf-8")
    with TestClient(build_app(st), base_url="http://127.0.0.1:7878"):
        pass
    assert st.paths.core_info.exists()                     # une autre instance : on n'y touche pas

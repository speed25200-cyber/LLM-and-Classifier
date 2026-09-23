"""Coeur de Prophet Studio de bout en bout (mode demo : faux llama-server lances par le vrai superviseur)."""

import json
import socket
import time

import pytest
from fastapi.testclient import TestClient

from prophet_studio.config import Paths
from prophet_studio.server import Studio, build_app
from prophet_studio.supervisor import Runtime

TOKEN = "t0k3n-test"
H = {"X-Prophet-Token": TOKEN}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def studio(tmp_path_factory, monkeypatch_module):
    root = tmp_path_factory.mktemp("home")
    monkeypatch_module.setenv("PROPHET_FAKE_GPU", "NVIDIA GeForce RTX 5060:8151:650:display")
    monkeypatch_module.setenv("FAKE_LLAMA_TPS", "0")
    paths = Paths(root)
    paths.settings.write_text(json.dumps({"workspace": str(root / "ws"), "s2_port": free_port(), "s1_port": free_port(),
                                          "voice": {"enabled": False}}))
    st = Studio(paths, TOKEN, 7878, demo=True)
    app = build_app(st)
    with TestClient(app, base_url="http://127.0.0.1:7878") as c:
        t0 = time.time()
        while st.runtime.state not in ("ready", "degraded", "error") and time.time() - t0 < 40:
            time.sleep(0.1)
        yield st, c
    st.runtime.stop()


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def wait_idle(st, sid, timeout=30):
    assert st.agent.wait_idle(sid, timeout)


def test_security_host_origin_token(studio):
    st, c = studio
    assert c.get("/api/health").json()["app"] == "prophet-studio"
    assert c.get("/api/state").status_code == 401
    assert c.get("/api/state", headers={"X-Prophet-Token": "wrong"}).status_code == 401
    assert c.get("/api/state", headers={**H, "Host": "evil.example.com"}).status_code == 403          # DNS rebinding
    assert c.get("/api/state", headers={**H, "Origin": "https://evil.example.com"}).status_code == 403
    ok = c.get("/api/state", headers={**H, "Origin": "tauri://localhost"})
    assert ok.status_code == 200 and ok.headers["access-control-allow-origin"] == "tauri://localhost"
    assert c.get("/api/state", headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200
    # la page servie embarque le jeton pour la meme origine
    assert TOKEN in c.get("/").text or c.get("/").status_code == 503


def test_state_plan_and_runtime_ready(studio):
    st, c = studio
    s = c.get("/api/state", headers=H).json()
    assert s["demo"] and s["hardware"]["primary_gpu"]["name"] == "NVIDIA GeForce RTX 5060"
    assert s["plan"]["s2"]["model_id"] == "bonsai2-27b-ptq1"
    assert s["runtime"]["state"] == "ready", s["runtime"]
    assert s["runtime"]["servers"]["s2"]["state"] == "ready" and s["runtime"]["servers"]["s1"]["state"] == "ready"
    assert "-c" in s["runtime"]["servers"]["s2"]["argv"] and "--jinja" in s["runtime"]["servers"]["s2"]["argv"]
    assert st.metrics()["gpu"]["vram_total_mib"] == 8151


def test_agent_turn_streams_and_persists_transcript(studio):
    st, c = studio
    sid = c.post("/api/sessions", headers=H, json={}).json()["id"]
    r = c.post(f"/api/sessions/{sid}/turn", headers=H, json={"text": "Cree une app minuteur pomodoro"})
    assert r.status_code == 200
    assert c.post(f"/api/sessions/{sid}/turn", headers=H, json={"text": "encore"}).status_code in (200, 409)
    wait_idle(st, sid)
    s = c.get(f"/api/sessions/{sid}", headers=H).json()
    item = next(i for i in s["transcript"] if i["kind"] == "assistant")
    assert item["status"] == "done" and item["path"] == "agent" and "index.html" in item["response"]
    tools = [b for b in item["blocks"] if b["type"] == "tool"]
    assert [b["name"] for b in tools][:4] == ["glob", "write_file", "python", "done"] and all(b["status"] == "done" for b in tools)
    assert tools[1]["ui"]["diff"].startswith("--- a/index.html")
    assert any(b["type"] == "thinking" for b in item["blocks"]) and item["s1"]["path"] == "agent"
    assert s["title"].startswith("Cree une app") and len(s["history"]) == 2
    files = c.get("/api/files", headers=H, params={"session": sid}).json()["entries"]
    assert "index.html" in files
    assert "Pomodoro" in c.get("/api/file", headers=H, params={"session": sid, "path": "index.html"}).json()["content"]
    assert c.get("/api/file", headers=H, params={"session": sid, "path": "../../etc/passwd"}).status_code == 403
    assert any(x["id"] == sid for x in c.get("/api/sessions", headers=H).json())


def test_permission_prompt_round_trip_in_ask_mode(studio):
    st, c = studio
    sid = c.post("/api/sessions", headers=H, json={"workspace": str(st.paths.root / "ws2")}).json()["id"]
    c.post(f"/api/sessions/{sid}/turn", headers=H, json={"text": "Cree une app minuteur", "permission_mode": "ask"})
    t0 = time.time()
    while not st.agent.pending_permissions() and time.time() - t0 < 20:
        time.sleep(0.05)
    pend = st.agent.pending_permissions()
    assert pend and pend[0]["tool"] == "write_file"
    assert c.post(f"/api/permissions/{pend[0]['id']}", headers=H, json={"allow": True, "remember": True}).json()["ok"]
    t0 = time.time()
    while sid in st.agent.running and time.time() - t0 < 20:   # le python suivant demande aussi : on refuse
        for p in st.agent.pending_permissions():
            c.post(f"/api/permissions/{p['id']}", headers=H, json={"allow": False})
        time.sleep(0.05)
    s = c.get(f"/api/sessions/{sid}", headers=H).json()
    perms = [b for i in s["transcript"] for b in i.get("blocks", []) if b["type"] == "permission"]
    assert perms[0]["decision"] == "allow" and perms[0]["preview"]["diff"]
    assert (st.paths.root / "ws2" / "index.html").exists() and "write_file" in s["always_allow"]


def test_cancel_and_plan_mode(studio):
    st, c = studio
    sid = c.post("/api/sessions", headers=H, json={"workspace": str(st.paths.root / "ws3")}).json()["id"]
    c.post(f"/api/sessions/{sid}/turn", headers=H, json={"text": "Cree une app minuteur", "plan_mode": True})
    wait_idle(st, sid)
    item = c.get(f"/api/sessions/{sid}", headers=H).json()["transcript"][-1]
    assert item["response"].startswith("Plan propose") and not (st.paths.root / "ws3" / "index.html").exists()
    assert c.post(f"/api/sessions/{sid}/cancel", headers=H).json()["cancelled"] is False


def test_websocket_events(studio):
    st, c = studio
    with pytest.raises(Exception):
        with c.websocket_connect("/api/events?token=bad") as ws:
            ws.receive_json()
    with c.websocket_connect(f"/api/events?token={TOKEN}") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello" and hello["state"]["runtime"]["state"] == "ready"
        sid = c.post("/api/sessions", headers=H, json={"workspace": str(st.paths.root / "ws4")}).json()["id"]
        c.post(f"/api/sessions/{sid}/turn", headers=H, json={"text": "bonjour !"})
        seen = []
        for _ in range(400):
            e = ws.receive_json()
            if e.get("session_id") == sid:
                seen.append(e["type"])
            if e["type"] == "turn.finished" and e.get("session_id") == sid:
                break
        assert "s1.decision" in seen and "text.delta" in seen and seen[-1] == "turn.finished"


def test_voice_route_systemone_and_openai_proxy(studio):
    st, c = studio
    r = c.post("/api/voice/route", headers=H, json={"text": "OK Prophet, nouvelle session"}).json()
    assert r["kind"] == "command" and r["command"] == "new_session"
    assert c.post("/api/voice/route", headers=H, json={"text": "Explique-moi en détail comment fonctionne un cache KV quantifié en q4"}).json()["kind"] == "prompt"
    d = c.post("/v1/systemone", headers=H, json={"state": "Refund please, charged twice", "questions": {
        "refund": {"type": "noul", "instructions": "Is a refund requested?"}}}).json()
    assert 0 <= d["answers"]["refund"]["noul"] <= 1
    o = c.post("/v1/chat/completions", headers={"Authorization": f"Bearer {TOKEN}"}, json={"messages": [{"role": "user", "content": "bonjour"}]}).json()
    assert o["choices"][0]["message"]["content"].startswith("Bonjour")
    assert c.post("/api/voice/tts", headers=H, json={"text": "salut"}).status_code == 503   # voix non installee : 503 explicite
    b = c.post("/api/bench", headers=H).json()
    assert b["s1_p50_ms"] > 0 and b["s2_tok_s"] > 0


def test_settings_update_replans(studio):
    st, c = studio
    s = c.put("/api/settings", headers=H, json={"priority": "vitesse"}).json()
    assert s["priority"] == "vitesse"
    assert c.get("/api/plan", headers=H).json()["s2"]["model_id"] == "bonsai-27b-q1"
    c.put("/api/settings", headers=H, json={"priority": "equilibre"})
    assert c.put("/api/settings", headers=H, json={"priority": "n'importe quoi"}).status_code == 422


def test_supervisor_steps_down_on_oom(tmp_path, monkeypatch):
    """Le faux serveur echoue en 'out of memory' au-dela de 8 k : le superviseur doit descendre l'echelle seul."""
    import sys
    from prophet_studio.hardware import GPU, HardwareInfo, enrich
    from prophet_studio.planner import make_plan
    monkeypatch.setenv("FAKE_LLAMA_OOM_ABOVE_CTX", "8192")
    events = []
    model = tmp_path / "m.gguf"; model.touch()
    rt = Runtime(tmp_path / "logs", events.append, lambda mid: model, lambda mid: None,
                 lambda: [sys.executable, "-m", "prophet_studio.demo.fake_llama"], lambda: {"bonsai2-27b-ptq1", "ternary-1.7b"})
    g = enrich(GPU(0, "NVIDIA GeForce RTX 5060", "nvidia", 8151, 650, 7501, "580", "13.0"))
    plan = make_plan(HardwareInfo("Windows", "11", "amd64", "x", 8, 16, 32, 24, gpus=[g]))
    assert plan.s2.ctx > 8192
    try:
        assert rt.start(plan, (free_port(), free_port()))
        assert rt.plan.s2.ctx <= 8192 and rt.plan.rung >= 1 and rt.state == "ready"
        assert any(e["type"] == "runtime.degraded" for e in events)
    finally:
        rt.stop()


def test_models_start_by_themselves_once_installed(tmp_path, monkeypatch):
    """Installer = pouvoir utiliser : quand le runtime et Bonsai arrivent, le superviseur demarre seul."""
    monkeypatch.setenv("PROPHET_FAKE_GPU", "NVIDIA GeForce RTX 5060:8151:650")
    paths = Paths(tmp_path / "home")
    paths.settings.write_text(json.dumps({"s2_port": free_port(), "s1_port": free_port(), "voice": {"enabled": False}}))
    st = Studio(paths, TOKEN, 7878)
    started = []
    st.start_runtime = lambda: started.append(True)
    assert st.recommended_missing_core()
    import sys as _sys
    model = tmp_path / "b.gguf"; model.touch()
    st.installer.registry["runtime"] = {"server": _sys.executable, "tag": "t", "backend": "cpu"}
    st.installer._save()
    assert started == []                                   # Bonsai manque encore
    st.installer.registry["models"][st.plan().s2.model_id] = {"main": str(model), "role": "s2"}
    st.installer._save()
    assert started == [True]

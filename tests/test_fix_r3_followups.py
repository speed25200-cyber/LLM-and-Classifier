"""Suites de la revue : GGUF importe minuscule (plus de division par zero dans /api/state), modele impose puis supprime
(retour a auto pour tout client), note RAM du plan CPU, budget de reflexion reduit avec max_tokens lors d'une reprise
apres depassement de contexte, bouton Stop pendant un long silence du serveur (prefill)."""

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from fastapi.testclient import TestClient

from jev_clone.backend_llamacpp import LlamaCppBackend
from prophet_studio.config import Paths
from prophet_studio.hardware import HardwareInfo
from prophet_studio.planner import cpu_plan
from prophet_studio.server import Studio, build_app

TOKEN = "tok-r3"
H = {"X-Prophet-Token": TOKEN}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_tiny_imported_gguf_does_not_crash_state_and_removal_resets_the_pin(tmp_path, monkeypatch):
    monkeypatch.setenv("PROPHET_FAKE_GPU", "NVIDIA GeForce RTX 5060:8151:650")
    paths = Paths(tmp_path / "home")
    paths.settings.write_text(json.dumps({"s2_port": free_port(), "s1_port": free_port(), "voice": {"enabled": False}}))
    tiny = tmp_path / "mini.gguf"
    tiny.write_bytes(b"GGUF" + b"\0" * 100)
    st = Studio(paths, TOKEN, 7878)
    with TestClient(build_app(st), base_url="http://127.0.0.1:7878") as c:
        mid = c.post("/api/import", headers=H, json={"path": str(tiny), "role": "s2"}).json()["id"]
        c.put("/api/settings", headers=H, json={"s2_model": mid})
        s = c.get("/api/state", headers=H)                       # avant : ZeroDivisionError, l'interface ne chargeait plus
        assert s.status_code == 200 and s.json()["plan"]["s2"]["model_id"] == mid
        r = c.delete(f"/api/installed/{mid}", headers=H)          # client quelconque, pas seulement l'interface
        assert r.status_code == 200 and r.json()["reset"] == ["s2_model"]
        assert c.get("/api/state", headers=H).json()["settings"]["s2_model"] == "auto"


def test_cpu_plan_warns_when_ram_is_short_and_speaks_french_numbers():
    small = HardwareInfo("Windows", "11", "amd64", "x", 8, 16, 8, 6, gpus=[])
    p = cpu_plan(small, "equilibre", "bonsai2-27b-ptq1")
    ram = [n for n in p.notes if n.startswith("RAM insuffisante")]
    assert not p.fits and ram and "," in ram[0] and "." not in ram[0].split(";")[0]
    assert not [n for n in cpu_plan(HardwareInfo("Windows", "11", "amd64", "x", 8, 16, 64, 50, gpus=[]), "vitesse").notes
                if n.startswith("RAM insuffisante")]


class _Server:
    """Faux llama-server : `mode` = 'overflow' (500 contexte puis 200) ou 'silent' (flux SSE ouvert, rien avant 30 s)."""

    def __init__(self, mode: str):
        self.mode, self.payloads = mode, []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.payloads.append(body)
                if outer.mode == "overflow" and len(outer.payloads) == 1:
                    msg = json.dumps({"error": {"message": "the request exceeds the available context size"}}).encode()
                    self.send_response(500); self.send_header("Content-Length", str(len(msg))); self.end_headers()
                    self.wfile.write(msg); return
                if outer.mode == "overflow":
                    msg = json.dumps({"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}).encode()
                    self.send_response(200); self.send_header("Content-Length", str(len(msg))); self.end_headers()
                    self.wfile.write(msg); return
                self.send_response(200); self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
                try:
                    time.sleep(30)                                   # prefill interminable : aucune ligne SSE
                except Exception:
                    pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self):
        self.httpd.shutdown()


def test_context_retry_shrinks_the_thinking_budget_with_max_tokens():
    srv = _Server("overflow")
    try:
        LlamaCppBackend(srv.url, timeout=10).chat([{"role": "user", "content": "x"}], max_tokens=4096, thinking_budget=2048)
        first, retry = srv.payloads
        assert first["thinking_budget_tokens"] == 2048 and retry["max_tokens"] == 1024
        assert retry["thinking_budget_tokens"] <= int(retry["max_tokens"] * 0.6)
    finally:
        srv.close()


def test_stop_interrupts_a_silent_stream_without_waiting_for_a_token():
    srv = _Server("silent")
    stop = threading.Event()
    try:
        threading.Timer(0.5, stop.set).start()
        t0 = time.time()
        out = LlamaCppBackend(srv.url, timeout=60).chat([{"role": "user", "content": "x"}], max_tokens=64,
                                                        on_delta=lambda d: None, should_stop=stop.is_set)
        assert out["cancelled"] and out["choices"][0]["finish_reason"] == "cancelled"
        assert time.time() - t0 < 5                                   # avant : attente de la premiere ligne (30 s ici)
    finally:
        srv.close()


def test_turn_stats_include_bonsai_work_done_inside_desktop_and_browse_tools(tmp_path):
    """Les appels de Bonsai faits dans l'outil (politique lente du computer use) comptent dans les stats du tour."""
    from jev_clone.prophet import Prophet, Workspace
    from tests.conftest import MockS2
    from tests.test_fix_r1_guard import DONE, eng
    run = {"ok": True, "status": "done", "steps": 3, "s1_ms": 42.0, "s1_calls": 5, "s2_calls": 2, "s2_tokens": 300, "s2_prompt_ms": 120.0}
    s2 = MockS2([{"content": "", "tool_calls": [MockS2.tool_call("desktop", {"goal": "ouvre la calculatrice", "app": "calc"}, "c1")]}, DONE])
    t = Prophet(eng(), s2, Workspace(tmp_path / "ws"), permission_mode="auto",
                desktop_factory=lambda: (lambda goal, app, slots: dict(run))).handle("ouvre la calculatrice")
    assert any(c["tool"] == "desktop" for c in t.tool_calls)
    # le faux Bonsai des tests ne publie pas llm.end : seuls les appels faits dans l'outil sont comptes ici
    assert t.stats["llm_calls"] == 2 and t.stats["tokens"] == 300 and t.stats["prompt_ms"] == 120.0 and t.stats["s1_calls"] >= 5

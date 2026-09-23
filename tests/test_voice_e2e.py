"""Voix de bout en bout dans un vrai navigateur : micro factice de Chromium -> AudioWorklet 16 kHz -> PCM ->
/api/voice/transcribe (moteur sherpa_onnx factice) -> routage -> commande executee par l'interface."""

import json
import sys
import threading
import time
from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api")

from prophet_studio.config import Paths  # noqa: E402
from prophet_studio.server import WEB_DIR, Studio, build_app  # noqa: E402
from tests.test_ui_e2e import CHROMIUM, free_port  # noqa: E402
from tests.test_voice_pipeline import fake_sherpa  # noqa: E402


@pytest.mark.skipif(not (WEB_DIR / "index.html").exists(), reason="interface non construite")
def test_push_to_talk_command_in_the_browser(tmp_path, monkeypatch):
    import uvicorn
    calls: list = []
    monkeypatch.setitem(sys.modules, "sherpa_onnx", fake_sherpa(calls))
    monkeypatch.setenv("PROPHET_FAKE_GPU", "NVIDIA GeForce RTX 5060:8151:650")
    paths = Paths(tmp_path / "home")
    paths.settings.write_text(json.dumps({"workspace": str(tmp_path / "ws"), "s2_port": free_port(), "s1_port": free_port(),
                                          "onboarding_done": True, "voice": {"enabled": False}}))
    port = free_port()
    st = Studio(paths, "tok-voice", port, demo=True)
    files = {"stt-parakeet-v3": {"encoder": "e", "decoder": "d", "joiner": "j", "tokens": "t"}}
    st.voice.files = files.get
    server = uvicorn.Server(uvicorn.Config(build_app(st), host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    try:
        t0 = time.time()
        while not (server.started and st.runtime.state == "ready") and time.time() - t0 < 40:
            time.sleep(0.1)
        n_sessions = len(st.sessions.list())
        with playwright.sync_playwright() as p:
            args = ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", "--autoplay-policy=no-user-gesture-required"]
            browser = None
            for exe in [None] + [c for c in CHROMIUM if c and Path(c).exists()]:
                try:
                    browser = p.chromium.launch(executable_path=exe, args=args) if exe else p.chromium.launch(args=args)
                    break
                except Exception:
                    continue
            if browser is None:
                pytest.skip("aucun Chromium disponible")
            page = browser.new_page()
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"http://127.0.0.1:{port}/")
            page.wait_for_selector("textarea", timeout=15000)
            mic = page.locator("button.mic")
            box = mic.bounding_box()
            page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
            page.mouse.down()
            page.wait_for_selector("button.mic.rec", timeout=5000)      # capture en cours (AudioWorklet)
            time.sleep(1.6)                                              # ~1,6 s de "parole" du micro factice
            page.mouse.up()
            page.wait_for_selector(".toast.voice", timeout=10000)
            toast = page.locator(".toast.voice").first.inner_text()
            assert "new session" in toast and "grammaire" in toast
            t1 = time.time()
            while len(st.sessions.list()) <= n_sessions and time.time() - t1 < 5:
                time.sleep(0.1)
            assert len(st.sessions.list()) == n_sessions + 1           # la commande vocale a cree une session
            browser.close()
        rec = [kw for k, kw in calls if k == "transducer"]
        assert rec and rec[0]["model_type"] == "nemo_transducer"
        assert errors == []
    finally:
        server.should_exit = True
        th.join(timeout=10)
        st.runtime.stop()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

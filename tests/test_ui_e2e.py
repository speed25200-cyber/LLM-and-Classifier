"""L'interface construite (prophet_studio/web) servie par le coeur, pilotee par Chromium : un tour d'agent complet,
sans erreur de console. Ignore si Playwright ou un navigateur manque."""

import json
import os
import socket
import threading
import time
from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api")

from prophet_studio.config import Paths  # noqa: E402
from prophet_studio.server import WEB_DIR, Studio, build_app  # noqa: E402

CHROMIUM = [os.environ.get("JEV_CHROMIUM", ""), "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.skipif(not (WEB_DIR / "index.html").exists(), reason="interface non construite (cd ui && npm run build)")
def test_agent_turn_in_the_real_ui(tmp_path, monkeypatch):
    import uvicorn
    monkeypatch.setenv("PROPHET_FAKE_GPU", "NVIDIA GeForce RTX 5060:8151:650:display")
    monkeypatch.setenv("FAKE_LLAMA_TPS", "200")
    paths = Paths(tmp_path / "home")
    paths.settings.write_text(json.dumps({"workspace": str(tmp_path / "ws"), "s2_port": free_port(), "s1_port": free_port(),
                                          "voice": {"enabled": False}}))
    port = free_port()
    st = Studio(paths, "tok-e2e", port, demo=True)
    server = uvicorn.Server(uvicorn.Config(build_app(st), host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    try:
        t0 = time.time()
        while not (server.started and st.runtime.state == "ready") and time.time() - t0 < 40:
            time.sleep(0.1)
        assert st.runtime.state == "ready"
        errors: list[str] = []
        with playwright.sync_playwright() as p:
            browser = None
            for exe in [None] + [c for c in CHROMIUM if c and Path(c).exists()]:
                try:
                    browser = p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()
                    break
                except Exception:
                    continue
            if browser is None:
                pytest.skip("aucun Chromium disponible")
            page = browser.new_page(viewport={"width": 1280, "height": 820})
            page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"http://127.0.0.1:{port}/")
            page.wait_for_selector(".wiz", timeout=15000)
            page.get_by_role("button", name="Passer l'assistant").click()
            page.locator("textarea").fill("Cree une app minuteur")
            page.keyboard.press("Enter")
            page.wait_for_selector(".meta", timeout=30000)
            assert page.locator(".s1 .chip.s1").first.inner_text().startswith("System One")
            assert page.locator(".tool").count() >= 3
            assert "index.html" in page.locator(".final").inner_text()
            assert (tmp_path / "ws" / "index.html").exists()
            page.keyboard.press("Control+k")
            page.wait_for_selector(".pal")
            browser.close()
        assert errors == []
    finally:
        server.should_exit = True
        th.join(timeout=10)
        st.runtime.stop()

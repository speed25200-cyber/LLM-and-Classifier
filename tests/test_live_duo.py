"""Duo reel : un tour Prophet complet contre deux vrais llama-server (JEV_TEST_S1 = classifieur, JEV_TEST_S2 = modele de
raisonnement ; meme URL pour les deux en mode mono). Invariants de structure seulement, aucune exigence de qualite.

    JEV_TEST_S1=http://127.0.0.1:7881 JEV_TEST_S2=http://127.0.0.1:7880 uv run --extra dev pytest -q tests/test_live_duo.py

Les tests rapides (toujours lances) passent eval/measure_duo.py sur deux faux serveurs et sur le coeur de Studio en mode demo :
l'outil de mesure ne doit pas pourrir entre deux mesures reelles."""

import json
import os
import random
import socket
import threading
import time
from pathlib import Path

import pytest

from eval import duo_probes as dp
from eval import measure_duo as md
from prophet_studio.demo.fake_llama import FakeLlama

S1, S2 = os.environ.get("JEV_TEST_S1"), os.environ.get("JEV_TEST_S2")
live = pytest.mark.skipif(not (S1 and S2), reason="JEV_TEST_S1 et JEV_TEST_S2 non definis")
FAST = ["--n-froid", "1", "--n-chaud", "1", "--n-s2", "1"]


@live
def test_live_prophet_turn_agent_creates_file(tmp_path):
    # voie agent imposee (effort deep) avec une reflexion bornee : le tour reste court sur une RTX 5060
    t = dp.run_turn_local(S1, S2, dp.AGENT_TASK, tmp_path / "ws", effort="deep", budgets=(0, 256, 512, 512), agent=True,
                          timeout=float(os.environ.get("JEV_TEST_TIMEOUT", "900")))
    assert t["error"] is None, t["error"]
    assert t["s1_decision"] is not None and not t["s1_problems"], t["s1_problems"]   # decision S1, probabilites dans [0, 1]
    assert t["path"] == "agent" and t["tools"], t                                     # S2 a appele au moins un outil
    assert t["files_created"], t                                                       # un fichier est apparu
    assert all(t["checks"].values()), t["checks"]                                      # statistiques du tour coherentes


@live
def test_live_measure_plumbing(tmp_path, capsys):
    rc = md.main(["--s1", S1, "--s2", S2, "--out", str(tmp_path), "--sans-tours", *FAST])
    res = json.loads(next(tmp_path.glob("duo-*.json")).read_text(encoding="utf-8"))
    assert rc == 0 and not res["errors"], res["errors"]
    for kind in ("noul", "choice", "score"):   # lecture de Prophet : grammaire = toute la masse sur les etiquettes
        r = res["s1_readout"][kind]
        assert r["grammar"]["all_labels_found"] and abs(r["grammar"]["label_share"] - 1) < 1e-3 and r["answer_in_01"], r
        assert 0.0 <= r["free"]["label_mass"] <= 1.0 + 1e-6
    assert all(v["cold"]["p50"] > 0 and v["warm"]["p50"] > 0 for v in res["s1_latency"].values())
    assert res["s2"]["generation"]["tok_s"] > 0 and res["s2"]["prefill"]["tok_s"] > 0
    assert "| S2 generation (tok/s) |" in capsys.readouterr().out


# ---- toujours lances ------------------------------------------------------------------------------------------------------
def test_cold_states_are_fresh_and_prophet_sized():
    rng = random.Random(0)
    a, b = dp.prophet_state(rng), dp.prophet_state(rng)
    assert a["request"] != b["request"] and a["request"][:34] != b["request"][:34]   # nonce en tete : jamais en cache
    assert len(a["request"]) == 2500 and len(a["workspace_files"]) == 60 and len(a["recent_turns"]) == 4
    assert all(len(t["content"]) <= 400 for t in a["recent_turns"])
    kinds = dp.s1_kinds()
    assert set(kinds) == {"noul", "choice", "score", "pre_tour", "outils", "garde_fou"}
    assert all(q["type"] == "noul" for q in kinds["outils"][0].values()) and "t_write_file" in kinds["outils"][0]


def test_measure_duo_against_fake_servers(tmp_path, capsys):
    f1, f2 = FakeLlama("fake-s1.gguf", ctx=8192), FakeLlama("fake-s2.gguf", ctx=16384)
    (srv1, _), (srv2, _) = f1.serve(), f2.serve()
    try:
        rc = md.main(["--s1", f"http://127.0.0.1:{srv1.server_port}", "--s2", f"http://127.0.0.1:{srv2.server_port}",
                      "--out", str(tmp_path), "--label", "faux serveurs", *FAST])
    finally:
        srv1.shutdown(); srv2.shutdown()
    out = capsys.readouterr().out
    res = json.loads(next(tmp_path.glob("duo-*.json")).read_text(encoding="utf-8"))
    assert rc == 0 and res["mode"] == "servers" and not res["errors"], res["errors"]
    assert res["servers"]["s1"]["model"] == "fake-s1.gguf" and res["servers"]["s2"]["n_ctx"] == 16384 and not res["servers"]["mono"]
    lat = res["s1_latency"]
    assert lat["pre_tour"]["questions"] == 6 and lat["pre_tour"]["branches"] == 6 and lat["garde_fou"]["questions"] == 3
    assert all(v["cold"]["n"] == 1 and v["cold"]["p50"] > 0 and v["warm"]["p50"] > 0 for v in lat.values())
    rd = res["s1_readout"]
    assert all(rd[k]["ok"] and rd[k]["grammar"]["label_mass"] == pytest.approx(1.0) for k in ("noul", "choice", "score")), rd
    assert rd["choice"]["labels"] == 7 and rd["score"]["labels"] == 4
    assert res["s2"]["generation"]["tok_s"] == pytest.approx(48.0) and res["s2"]["prefill"]["tok_s"] > 0
    assert res["s2"]["generation"]["ttft_ms"] is not None and res["thinking"]["honored"]
    # le budget de reflexion part comme Prophet l'envoie : 0 (reflexion coupee) puis 512
    sent = [b.get("thinking_budget_tokens") for p, b in f2.requests if p.startswith("/v1/chat")]
    assert 0 in sent and 512 in sent
    d, a = res["turns"]["direct"], res["turns"]["agent"]
    assert d["path"] == "direct" and d["ok"] and d["s2_calls"] == 1 and d["s1_decision"]["path"] == "direct", d
    assert a["path"] == "agent" and a["ok"] and a["files_created"] == ["index.html"] and "write_file" in a["tools"], a
    assert a["s1_calls"] >= 2 and a["s1_ms"] <= a["latency_ms"] <= a["wall_ms"] and a["ttft_ms"] is not None
    assert not res["vram"]["available"] or res["vram"]["peak_mib"] >= res["vram"]["before_mib"]
    assert len(res["markdown"]) == 2 + 13 and res["markdown"] == md.markdown_rows(res)
    assert "| S2 generation (tok/s) | 47-59 | 48,0 | " in out and "faux serveurs" in out and str(tmp_path) in out
    assert all(line.isascii() for line in res["markdown"])


def test_measure_duo_failures_are_readable(tmp_path, monkeypatch):
    # core.json perime (coeur arrete) : message clair, rien n'est ecrit
    (tmp_path / "demo").mkdir()
    (tmp_path / "demo" / "core.json").write_text(json.dumps({"url": f"http://127.0.0.1:{_free_port()}", "token": "x"}))
    assert md.find_core(tmp_path) is None
    with pytest.raises(SystemExit, match="Prophet Studio ne tourne pas"):
        md.main(["--data-dir", str(tmp_path), "--out", str(tmp_path / "out")])
    with pytest.raises(SystemExit, match="S1 ne repond pas"):
        md.main(["--s1", f"http://127.0.0.1:{_free_port()}", "--s2", f"http://127.0.0.1:{_free_port()}", "--out", str(tmp_path / "out")])
    with pytest.raises(SystemExit, match="vont ensemble"):
        md.main(["--s1", "http://127.0.0.1:1"])
    assert not (tmp_path / "out").exists()
    # sections absentes ou en echec : le tableau reste complet, avec des tirets
    rows = md.markdown_rows({"started": "2026-09-23T10:00:00", "errors": {"s2": "ConnectionError"},
                             "turns": {"agent": {"error": "delai depasse", "wall_ms": None}}})
    assert len(rows) == 15 and all(r.count("|") == 5 for r in rows) and "echec : delai depasse" in rows[13]


def test_vram_sampler_reads_nvidia_smi(monkeypatch):
    used = iter([1800, 7300, 7900, 7450] + [7450] * 50)

    class Out:
        def __init__(self):
            self.stdout = f"NVIDIA GeForce RTX 5060, 581.57, {next(used)}, 8151\n"
    monkeypatch.setattr(dp, "nvidia_smi", lambda: "nvidia-smi")
    monkeypatch.setattr(dp.subprocess, "run", lambda *a, **k: Out())
    s = dp.VramSampler(interval=0.01).start()
    time.sleep(0.2)
    v = s.stop()
    assert v["available"] and v["gpu"] == "NVIDIA GeForce RTX 5060" and v["driver"] == "581.57" and v["total_mib"] == 8151
    assert v["before_mib"] == 1800 and v["peak_mib"] == 7900 and v["after_mib"] == 7450
    row = md.markdown_rows({"started": "2026-09-23T10:00:00", "errors": {}, "vram": v})
    assert "| 1 800 / 7 900 / 7 450 (sur 8 151) |" in row[-1] and "RTX 5060 (pilote 581.57)" in row[2]
    monkeypatch.setattr(dp, "nvidia_smi", lambda: None)
    assert dp.VramSampler().start().stop() == {"available": False, "note": "nvidia-smi absent : VRAM non mesuree"}


def test_scoreboard_table_and_commands_match_the_tool():
    root = Path(md.__file__).resolve().parents[1]
    board = (root / "eval" / "SCOREBOARD.md").read_text(encoding="utf-8")
    start = board.index(md.TABLE_HEAD[0])
    table = board[start:board.index("\n\n", start)].splitlines()
    cols = lambda r: [c.strip() for c in r.strip("|").split("|")][:2]   # noqa: E731
    assert [cols(r) for r in table] == [cols(r) for r in md.markdown_rows({"started": "2026-09-23T10:00:00", "errors": {}})]
    ps1 = (root / "scripts" / "measure_rtx5060.ps1").read_text(encoding="utf-8")
    sh = (root / "scripts" / "measure.sh").read_text(encoding="utf-8")
    assert "powershell -ExecutionPolicy Bypass -File .\\scripts\\measure_rtx5060.ps1" in board and "sh scripts/measure.sh" in board
    assert all(p in ps1 for p in ("[switch]$Rapide", "[string]$Label", "app-venv", "'venv'", "core.json", "eval.measure_duo"))
    assert all(p in sh for p in ("app-venv", "venv", "core.json", "eval.measure_duo", "--demarrer"))
    assert ps1.isascii() and sh.isascii()   # Windows PowerShell 5.1 lit les scripts sans BOM en ANSI


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_measure_duo_against_demo_studio(tmp_path, monkeypatch, capsys):
    """Coeur de Studio en mode demo servi en HTTP (faux llama-server lances par le vrai superviseur), trouve par core.json."""
    import uvicorn

    from prophet_studio.config import Paths
    from prophet_studio.server import Studio, build_app
    monkeypatch.setenv("FAKE_LLAMA_TPS", "0")
    root = tmp_path / "home"
    paths = Paths(root)
    paths.settings.write_text(json.dumps({"workspace": str(root / "ws"), "s2_port": _free_port(), "s1_port": _free_port(),
                                          "voice": {"enabled": False}}))
    port, token = _free_port(), "t0k3n-mesure"
    st = Studio(paths, token, port, demo=True)
    srv = uvicorn.Server(uvicorn.Config(build_app(st), host="127.0.0.1", port=port, log_level="warning", access_log=False))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    try:
        t0 = time.time()
        while not srv.started and time.time() - t0 < 20:
            time.sleep(0.05)
        paths.core_info.write_text(json.dumps({"url": f"http://127.0.0.1:{port}", "port": port, "token": token, "pid": os.getpid()}))
        rc = md.main(["--studio", "--data-dir", str(root), "--demarrer", "--out", str(tmp_path / "out"), *FAST])
        res = json.loads(next((tmp_path / "out").glob("duo-*.json")).read_text(encoding="utf-8"))
        assert rc == 0 and res["mode"] == "studio" and not res["errors"], res["errors"]
        assert res["studio"]["demo"] and res["studio"]["runtime_state"] in ("ready", "degraded")
        assert res["servers"]["s1"]["url"].endswith(str(json.loads(paths.settings.read_text())["s1_port"]))
        d, a = res["turns"]["direct"], res["turns"]["agent"]
        assert d["mode"] == "studio" and d["path"] == "direct" and d["ok"], d
        assert a["path"] == "agent" and a["ok"] and a["files_created"] == ["index.html"] and a["tools"], a
        assert st.sessions.list() == []                  # sessions de mesure supprimees
        assert "DEMO (faux serveurs)" in capsys.readouterr().out
    finally:
        srv.should_exit = True
        th.join(timeout=15)
        st.runtime.stop()

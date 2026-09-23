"""Classifieur (System One) dans Studio : GGUF entraine importe reellement utilise, calibration par modele,
et correction du routeur de fusion (meme statistique a la calibration et a la porte, reponses de Bonsai typees)."""

import dataclasses
import json
import math
import socket
import time

import numpy as np
import pytest
import requests
from fastapi.testclient import TestClient

from jev_clone.calibrate import SEEDS, calibrate, collect, fit, load_examples, main as calibrate_main
from jev_clone.engine import SystemOneEngine
from jev_clone.fusion import FusionRouter, GatePolicy, answer_confidence, coerce_answer, gate_statistic
from jev_clone.readout import Calibration, probs_to_logits, softmax
from jev_clone.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion
from prophet_studio import calibration as s1cal
from prophet_studio.catalog import CUSTOM, MODELS, catalog_dict, custom_spec
from prophet_studio.config import Paths
from prophet_studio.hardware import GPU, HardwareInfo, enrich
from prophet_studio.planner import degrade, make_plan
from prophet_studio.server import Studio, build_app
from prophet_studio.sessions import AgentService, SessionStore
from tests.conftest import MockBackend

TOKEN = "t0k3n-s1"
H = {"X-Prophet-Token": TOKEN}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def hw(name="NVIDIA GeForce RTX 5060", total=8151, used=650):
    g = enrich(GPU(0, name, "nvidia", total, used, total - used, "580.88", "13.0", display_active=True))
    return HardwareInfo("Windows", "11", "amd64", "AMD Ryzen 7 7700", 8, 16, 32, 22, True, True, [g])


CPU_ONLY = HardwareInfo("Linux", "6", "x86_64", "i7", 8, 16, 16, 12, True, False, [])


@pytest.fixture
def custom(tmp_path):
    """GGUF importes tels que l'installateur les declare (petits fichiers : la taille est reglee sur la spec)."""
    saved = dict(CUSTOM)
    CUSTOM.clear()
    out = {}
    for mid, role, gib in (("custom-s1-jev-clone-q8-0", "s1", 2.0), ("custom-s2-mon-cerveau", "s2", 5.0)):
        f = tmp_path / f"{mid}.gguf"
        f.write_bytes(b"GGUF" + b"\0" * 64)
        CUSTOM[mid] = dataclasses.replace(custom_spec(mid, role, f), weights_gib=gib)
        out[role] = (mid, f)
    yield out
    CUSTOM.clear()
    CUSTOM.update(saved)


# ---- 1. un GGUF entraine importe est applique par le planificateur ------------------------------------------------------
def test_planner_uses_imported_s1_on_gpu_and_cpu(custom):
    mid, _ = custom["s1"]
    p = make_plan(hw(), s1_override=mid)
    assert p.s1.model_id == mid and p.s2.model_id == "bonsai2-27b-ptq1" and p.fits
    assert any("GGUF importe" in n for n in p.notes)
    big = make_plan(hw("NVIDIA GeForce RTX 5060 Ti", 16311, 700), s1_override=mid)
    assert big.s1.model_id == mid and big.s1.device == "gpu"
    assert big.budget["s1_weights"] == round(2.0 * 1024)          # memoire prise depuis la taille du fichier
    assert big.budget["s1_kv"] > round(MODELS["ternary-4b"].kv_kib_f16 * 0.53 * 8)   # KV majore (architecture inconnue)
    cpu = make_plan(CPU_ONLY, s1_override=mid)
    assert cpu.s1.model_id == mid and cpu.s1.device == "cpu"
    assert cpu.budget["ram_needed_mib"] > make_plan(CPU_ONLY).budget["ram_needed_mib"] + 1024


def test_planner_uses_imported_s2_and_degrade_handles_it(custom):
    mid, _ = custom["s2"]
    p = make_plan(hw(), s2_override=mid)
    assert p.s2.model_id == mid and p.s2.device == "gpu" and p.s2.mmproj == "off"
    assert make_plan(CPU_ONLY, s2_override=mid).s2.model_id == mid
    d, steps = p, 0
    while d is not None and steps < 40:          # l'echelle anti-OOM ne doit pas planter sur un id hors catalogue
        d, steps = degrade(d), steps + 1
    assert steps < 40


def test_override_not_applied_is_never_silent(custom):
    mid, f = custom["s1"]
    f.unlink()                                   # GGUF deplace ou supprime
    p = make_plan(hw(), s1_override=mid)
    assert p.s1.model_id == "ternary-1.7b" and any("introuvable" in n and mid in n for n in p.notes)
    assert any("introuvable" in n for n in make_plan(CPU_ONLY, s2_override="custom-s2-inconnu").notes)


def test_installer_registers_imported_gguf_and_studio_plans_with_it(tmp_path, monkeypatch):
    monkeypatch.setenv("PROPHET_FAKE_GPU", "NVIDIA GeForce RTX 5060:8151:650")
    paths = Paths(tmp_path / "home")
    paths.settings.write_text(json.dumps({"s2_port": free_port(), "s1_port": free_port(), "voice": {"enabled": False}}))
    gguf = tmp_path / "jev-clone-Q8_0.gguf"
    gguf.write_bytes(b"GGUF" + b"\0" * 1024)
    st = Studio(paths, TOKEN, 7878)
    with TestClient(build_app(st), base_url="http://127.0.0.1:7878") as c:
        r = c.post("/api/import", headers=H, json={"path": str(gguf), "role": "s1"})
        mid = r.json()["id"]
        assert r.status_code == 200 and mid == "custom-s1-jev-clone-q8-0" and mid in CUSTOM
        c.put("/api/settings", headers=H, json={"s1_model": mid})
        assert c.get("/api/plan", headers=H).json()["s1"]["model_id"] == mid        # avant : ternary-1.7b relance
        s = c.get("/api/state", headers=H).json()
        spec = next(m for m in s["catalog"]["models"] if m["id"] == mid)
        assert spec["custom"] and spec["role"] == "s1" and s["installed"]["models"][mid]["installed"]
        assert c.post("/api/import", headers=H, json={"path": str(gguf), "role": "voice"}).status_code == 400
        # re-import (nouveaux poids, meme id) : l'ancienne calibration ne s'applique plus
        s1cal.store(paths.runs, mid, Calibration(temperature={"noul": 2.0}))
        c.post("/api/import", headers=H, json={"path": str(gguf), "role": "s1"})
        assert mid not in c.get("/api/state", headers=H).json()["installed"]["calibration"]
        assert c.delete(f"/api/installed/{mid}", headers=H).status_code == 200
        assert mid not in CUSTOM and all(m["id"] != mid for m in catalog_dict()["models"]) and gguf.exists()
        p = c.get("/api/plan", headers=H).json()
        assert p["s1"]["model_id"] == "ternary-1.7b" and any("introuvable" in n for n in p["notes"])


def test_demo_does_not_clobber_an_imported_model(tmp_path, monkeypatch):
    """--demo sur un dossier de donnees ou un clone importe est choisi : ni plantage ni ecrasement du registre."""
    monkeypatch.setenv("PROPHET_FAKE_GPU", "NVIDIA GeForce RTX 5060:8151:650")
    paths = Paths(tmp_path / "home")
    gguf = tmp_path / "clone.gguf"
    gguf.write_bytes(b"GGUF")
    paths.installed.write_text(json.dumps({"models": {"custom-s1-clone": {"main": str(gguf), "role": "s1", "custom": True}},
                                           "runtime": None, "voice": {}}))
    paths.settings.write_text(json.dumps({"s1_model": "custom-s1-clone", "voice": {"enabled": False}}))
    st = Studio(paths, TOKEN, 7878, demo=True)
    assert st.installer.registry["models"]["custom-s1-clone"]["main"] == str(gguf)
    assert st.plan().s1.model_id == "custom-s1-clone"


# ---- 2. calibration : commande documentee, fichier par modele, cache invalide -----------------------------------------------
def seed_engine() -> SystemOneEngine:
    return SystemOneEngine(MockBackend({"direct": [0.2, 0.8], "risk": [0.7, 0.2, 0.05, 0.05]}), model_name="mock")


def test_seeds_are_shipped_in_the_package_and_match_training_seeds():
    shipped = load_examples()
    assert SEEDS.parent.name == "seeds" and SEEDS.parent.parent.name == "jev_clone" and len(shipped) == 82
    with open("training/seeds/prophet_seeds.jsonl", encoding="utf-8") as f:
        assert shipped == [json.loads(l) for l in f if l.strip()]
    toml = open("pyproject.toml", encoding="utf-8").read()
    assert 'jev_clone = ["seeds/*.jsonl"]' in toml
    import shutil
    import subprocess
    if shutil.which("git"):
        r = subprocess.run(["git", "check-ignore", "-q", "jev_clone/seeds/prophet_seeds.jsonl"], capture_output=True)
        assert r.returncode != 0          # 0 = exclu par .gitignore (comme data/) : le fichier ne serait jamais publie


def test_collect_uses_prophet_turn_when_seeds_have_no_questions():
    per_kind, per_qid = collect(seed_engine(), load_examples())        # avant : KeyError 'questions'
    assert set(per_qid) == {"direct", "clarify", "intent", "language", "needs_reasoning", "risk"}
    assert all(len(v[1]) == 82 for v in per_qid.values()) and len(per_kind["noul"][0]) == 3 * 82
    with pytest.raises(ValueError):                                    # lectures deja calibrees : refusees
        collect(SystemOneEngine(MockBackend({}), calibration=Calibration(temperature={"noul": 2.0})), load_examples()[:1])


def test_documented_cli_writes_the_studio_file_for_one_model(tmp_path, monkeypatch):
    monkeypatch.setenv("PROPHET_HOME", str(tmp_path / "home"))
    cal = calibrate_main(["--data", "training/seeds/prophet_seeds.jsonl", "--studio-model", "ternary-1.7b"], engine=seed_engine())
    f = tmp_path / "home" / "runs" / "calibration" / "ternary-1.7b.json"
    loaded = Calibration.load(f)
    assert loaded.temperature == cal.temperature and loaded.meta["model_id"] == "ternary-1.7b" and loaded.meta["n"] == 82
    assert set(loaded.thresholds) == {"direct", "clarify", "intent", "language", "needs_reasoning", "risk"}
    out = tmp_path / "libre.json"
    calibrate_main(["--out", str(out)], engine=seed_engine())           # --data facultatif : graines livrees
    assert Calibration.load(out).meta["source"] == "cli"


def test_agent_service_loads_calibration_of_the_running_s1_only(tmp_path):
    runs = tmp_path / "runs"
    running = {"model": "ternary-1.7b"}
    svc = AgentService(SessionStore(tmp_path / "s"), lambda e: None, urls=lambda: ("http://127.0.0.1:1", "http://127.0.0.1:2"),
                       settings=lambda: None, ctx=lambda: 8192, calibration=lambda: s1cal.calibration_file(runs, running["model"]))
    assert svc.engines()[0].cal.t("choice") == 1.0                     # pas encore calibre
    s1cal.store(runs, "ternary-4b", Calibration(temperature={"choice": 3.0}))
    assert svc.engines()[0].cal.t("choice") == 1.0                     # la calibration d'un autre S1 n'est jamais appliquee
    s1cal.store(runs, "ternary-1.7b", Calibration(temperature={"choice": 2.0}))
    assert svc.engines()[0].cal.t("choice") == 2.0                     # cache invalide a l'ecriture
    running["model"] = "ternary-4b"
    assert svc.engines()[0].cal.t("choice") == 3.0
    running["model"] = None                                            # mode mono : Bonsai n'est pas le modele calibre
    assert svc.engines()[0].cal.t("choice") == 1.0
    running["model"] = "ternary-1.7b"
    s1cal.calibration_file(runs, "ternary-1.7b").write_text("{pas du json")
    assert svc.engines()[0].cal.t("choice") == 1.0                     # fichier illisible : lecture brute, pas de panne
    assert "error" in s1cal.status(runs)["ternary-1.7b"] and s1cal.calibration_file(runs, "../x") is None


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setenv("PROPHET_FAKE_GPU", "NVIDIA GeForce RTX 5060:8151:650:display")
    mp.setenv("FAKE_LLAMA_TPS", "0")
    root = tmp_path_factory.mktemp("home-cal")
    paths = Paths(root)
    paths.settings.write_text(json.dumps({"workspace": str(root / "ws"), "s2_port": free_port(), "s1_port": free_port(),
                                          "voice": {"enabled": False}}))
    st = Studio(paths, TOKEN, 7878, demo=True)
    with TestClient(build_app(st), base_url="http://127.0.0.1:7878") as c:
        t0 = time.time()
        while st.runtime.state not in ("ready", "degraded", "error") and time.time() - t0 < 40:
            time.sleep(0.1)
        yield st, c
    st.runtime.stop()
    mp.undo()


def test_studio_calibrates_the_running_s1_with_progress_events(demo):
    st, c = demo
    assert st.runtime.state == "ready" and not st.runtime.mono
    mid = st.runtime.plan.s1.model_id
    assert mid not in c.get("/api/state", headers=H).json()["installed"]["calibration"]    # "non calibre" affichable
    events, publish = [], st.bus.publish
    st.bus.publish = lambda e: (events.append(e), publish(e))
    try:
        r = c.post("/api/calibrate", headers=H, json={})
    finally:
        st.bus.publish = publish
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["model_id"] == mid and out["n"] == 82 and set(out["temperature"]) == {"noul", "choice", "score"}
    assert set(out["thresholds"]) == {"direct", "clarify", "intent", "language", "needs_reasoning", "risk"}
    f = st.paths.runs / "calibration" / f"{mid}.json"
    assert f.exists() and not (st.paths.runs / "calibration.json").exists()
    prog = [e for e in events if e["type"] == "s1.calibration"]
    assert prog[0]["done"] == 0 and prog[-1]["status"] == "done" and any(e.get("done") == 41 for e in prog)
    assert any(e["type"] == "install.changed" and mid in e["installed"]["calibration"] for e in events)
    cal = c.get("/api/state", headers=H).json()["installed"]["calibration"][mid]
    assert cal["source"] == "studio" and cal["n"] == 82 and "after" in cal["report"]["noul"]
    assert st.agent.engines()[0].cal.temperature == Calibration.load(f).temperature   # applique au S1 en marche
    # une seule calibration a la fois ; rien a calibrer en mode mono
    assert s1cal._running.acquire(blocking=False)
    try:
        assert c.post("/api/calibrate", headers=H, json={}).status_code == 409
    finally:
        s1cal._running.release()
    st.runtime.mono = True
    try:
        assert c.post("/api/calibrate", headers=H, json={}).status_code == 409
        assert st.agent.calibration_path() is None
    finally:
        st.runtime.mono = False


def test_studio_imports_a_calibration_for_a_model(demo, tmp_path):
    st, c = demo
    good = tmp_path / "calibration.json"
    good.write_text(json.dumps({"temperature": {"noul": 1.3, "choice": 1.8, "score": 1.1}, "thresholds": {"direct": 0.9, "risk": None}}))
    r = c.post("/api/calibration/import", headers=H, json={"path": str(good), "model_id": "ternary-4b"})
    assert r.status_code == 200 and r.json()["model_id"] == "ternary-4b"
    cal = Calibration.load(st.paths.runs / "calibration" / "ternary-4b.json")
    assert cal.t("choice") == 1.8 and cal.threshold("risk", 0.5) == math.inf and cal.meta["source"] == "import"
    bad = tmp_path / "vram.json"
    bad.write_text(json.dumps({"bonsai2-27b-ptq1": {"overhead_mib": 900}}))
    assert c.post("/api/calibration/import", headers=H, json={"path": str(bad), "model_id": "ternary-4b"}).status_code == 400
    assert c.post("/api/calibration/import", headers=H, json={"data": {"temperature": {"noul": -1}}, "model_id": "ternary-4b"}).status_code == 400
    assert c.post("/api/calibration/import", headers=H, json={"path": str(good), "model_id": "inconnu"}).status_code == 404
    assert c.post("/api/calibration/import", headers=H, json=[str(good)]).status_code == 422
    other = {"temperature": {"noul": 1.2}, "meta": {"model_id": "ternary-8b"}}
    assert c.post("/api/calibration/import", headers=H, json={"data": other, "model_id": "ternary-4b"}).status_code == 400
    assert c.post("/api/calibration/import", headers=H, json={"data": other, "model_id": "ternary-4b", "force": True}).status_code == 200


# ---- 3. routeur de fusion : meme statistique, reponses typees, replis explicites ----------------------------------------------
@pytest.fixture
def req3():
    return {"state": {"ticket": "Payouts failed three times."},
            "questions": {"team": {"type": "choice", "instructions": "Which team?", "criteria": {"payments": None, "account": None, "other": None}},
                          "escalate": {"type": "noul", "instructions": "Escalate?"},
                          "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["can wait", "handle today", "blocked now"]}}}


def backend(team=(0.9, 0.05, 0.05)):
    be = MockBackend({"team": list(team), "escalate": [0.7, 0.3], "urgency": [0.1, 0.2, 0.7]})
    be.declared = {"team": ["payments", "account", "other"]}
    return be


def test_gate_uses_the_calibration_statistic(req3):
    # [.9, .05, .05] : 1 - entropie normalisee = 0.64 ; la porte comparait cela a un seuil ajuste sur max-prob (0.9)
    r = FusionRouter(SystemOneEngine(backend(), model_name="m"), MockBackend({}, chat_reply="{}"), GatePolicy(thresholds={"team": 0.9}))
    d = r.decide(req3)
    assert d.gated["team"] and d.sources["team"] == "s1" and abs(answer_confidence(d.s1.answers["team"]) - 0.9) < 1e-6


def test_fit_thresholds_after_temperature_on_gate_statistic(req3):
    # 15 lectures justes a [.9, .05, .05], 5 fausses a [.6, .3, .1] : T != 1, seuils calcules sur les probabilites mises a l'echelle
    rows = [probs_to_logits(np.array([.9, .05, .05]))] * 15 + [probs_to_logits(np.array([.6, .3, .1]))] * 5
    gold = [0] * 15 + [1] * 5
    cal, _ = fit({"noul": ([], []), "choice": (rows, gold), "score": ([], [])}, {"team": ("choice", rows, gold)}, 0.95)
    T = cal.t("choice")
    assert abs(T - 1.0) > 0.05
    scaled = gate_statistic(softmax(rows[0], T))
    assert cal.thresholds["team"] == math.floor(scaled * 1e4) / 1e4 and cal.thresholds["team"] != 0.9
    # a l'execution, l'engine applique T : un cas qui a servi a fixer le seuil passe la porte
    router = FusionRouter(SystemOneEngine(backend(), calibration=cal, model_name="m"), MockBackend({}, chat_reply="{}"),
                          GatePolicy.from_calibration(cal))
    assert router.decide(req3).gated["team"]


def test_question_without_reachable_threshold_always_escalates(req3, tmp_path):
    rows = [probs_to_logits(np.array([.9, .05, .05]))] * 10
    cal, _ = fit({"noul": ([], []), "choice": (rows, [1] * 10), "score": ([], [])}, {"team": ("choice", rows, [1] * 10)}, 0.95)
    assert cal.thresholds["team"] is None
    cal.save(tmp_path / "c.json")
    cal = Calibration.load(tmp_path / "c.json")
    assert cal.thresholds["team"] is None and cal.threshold("team", 0.5) == math.inf and cal.threshold("absent", 0.5) == 0.5
    sure = backend(team=(1.0, 0.0, 0.0))
    d = FusionRouter(SystemOneEngine(sure, model_name="m"), None, GatePolicy.from_calibration(cal, default_threshold=0.0)).decide(req3)
    assert not d.gated["team"] and d.gated["escalate"] and d.unresolved == ["team"]    # avant : repli sur 0.85


def test_s2_answers_are_typed(req3):
    s2 = MockBackend({}, chat_reply='Reasoning... {"team": "ACCOUNT", "escalate": "no", "urgency": "handle today"}')
    d = FusionRouter(SystemOneEngine(backend(), model_name="m"), s2, GatePolicy(default_threshold=1.01)).decide(req3)
    assert d.path == "escalated" and d.decisions == {"team": "account", "escalate": False, "urgency": 1.0}   # avant : "no" (vrai)
    assert set(d.sources.values()) == {"s2"} and d.unresolved == []
    q = NoulQuestion(type="noul", instructions="x")
    assert coerce_answer(q, "Yes.") == (True, True) and coerce_answer(q, 0) == (True, False) and not coerce_answer(q, "maybe")[0]
    sq = ScoreQuestion(type="score", instructions="x", criteria=["a", "b", "c"])
    assert coerce_answer(sq, "2") == (True, 2.0) and not coerce_answer(sq, 5)[0] and not coerce_answer(sq, True)[0]
    assert not coerce_answer(ChoiceQuestion(type="choice", instructions="x", criteria=["a", "b"]), "z")[0]
    assert "level number" in s2.chat_calls[0][0][1]["content"]


def test_s2_omission_or_invalid_answer_is_flagged(req3, tmp_path):
    s2 = MockBackend({}, chat_reply='{"team": "nonexistent"}')
    d = FusionRouter(SystemOneEngine(backend(), model_name="m"), s2, GatePolicy(default_threshold=1.01), ledger=tmp_path / "l.jsonl").decide(req3)
    assert d.path == "escalated" and d.decisions["team"] == "payments" and d.decisions["escalate"] is True
    assert d.sources == {"team": "s1_fallback", "escalate": "s1_fallback", "urgency": "s1_fallback"}
    assert sorted(d.unresolved) == ["escalate", "team", "urgency"]
    assert json.loads((tmp_path / "l.jsonl").read_text())["unresolved"] == d.unresolved


def test_s2_down_falls_back_to_s1_without_http_500(req3):
    class DownS2:
        def chat(self, *a, **kw):
            raise requests.ConnectionError("connexion refusee")
    router = FusionRouter(SystemOneEngine(backend(), model_name="m"), DownS2(), GatePolicy(default_threshold=1.01))
    d = router.decide(req3)
    assert d.path == "fallback" and "ConnectionError" in d.s2_error and d.decisions["team"] == "payments"
    assert sorted(d.unresolved) == ["escalate", "team", "urgency"]
    from jev_clone.server import build_app as jev_app
    with TestClient(jev_app(router.s1, router)) as c:
        r = c.post("/v1/fusion/decide", json=req3)
    assert r.status_code == 200 and r.json()["path"] == "fallback" and r.json()["s2_error"]


def test_end_to_end_calibration_then_gate_on_mock(req3):
    """calibrate() puis porte : les deux cotes lisent la meme statistique, a la meme temperature."""
    exs = [{"state": {"ticket": f"t{i}"}, "questions": req3["questions"], "labels": {"team": "payments", "escalate": True, "urgency": 2}}
           for i in range(30)]
    cal, reports = calibrate(SystemOneEngine(backend(), model_name="m"), exs, 0.95, source="test")
    assert cal.meta["n"] == 30 and cal.meta["statistic"] == "top1" and reports["choice"]["after"].nll <= reports["choice"]["before"].nll
    eng = SystemOneEngine(backend(), calibration=cal, model_name="m")
    a = eng.answer(req3).answers
    for qid in ("team", "escalate", "urgency"):
        thr = cal.thresholds[qid]
        assert thr is not None and answer_confidence(a[qid]) >= thr

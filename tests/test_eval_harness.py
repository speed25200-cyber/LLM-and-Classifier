import json

from eval.jev_benchmark import CRITERIA, DATA, analyze, metrics, run
from jev_clone.engine import SystemOneEngine
from tests.conftest import MockBackend


def test_jev_recorded_results_reproduce_published_numbers():
    rows = [json.loads(l) for l in open(DATA / "jev-jev-latest.jsonl") if l.strip()]
    m = metrics(rows)
    assert m["n"] == 60 and abs(m["accuracy"] - 55 / 60) < 1e-9
    assert m["acc_clear"] == 1.0 and abs(m["acc_ambiguous"] - 10 / 14) < 1e-9 and abs(m["acc_adversarial"] - 11 / 12) < 1e-9
    assert abs(m["p50_ms"] - 421.6) < 0.2 and abs(m["ece10"] - 0.0712) < 5e-4
    assert m["misses"] == 5 and m["misses_at_1"] == 0


def test_run_clone_on_benchmark(tmp_path, monkeypatch):
    import eval.jev_benchmark as jb
    monkeypatch.setattr(jb, "RESULTS", tmp_path)
    be = MockBackend({"risk": [0.1, 0.1, 0.7, 0.1]})
    be.declared = {"risk": list(CRITERIA)}  # readonly, destructive, privileged, exfiltration
    out = run(SystemOneEngine(be, model_name="mock"), "mock")
    rows = [json.loads(l) for l in open(out) if l.strip()]
    assert len(rows) == 60 and all(r["choice"] == "privileged" for r in rows)
    m = metrics(rows)
    n_priv = sum(1 for r in rows if r["label"] == "privileged")
    assert abs(m["accuracy"] - n_priv / 60) < 1e-9 and m["ece10"] is not None
    tables = analyze([str(DATA / "jev-jev-latest.jsonl"), str(out)])
    assert [t["name"] for t in tables] == ["jev/jev-latest", "jev_clone/mock"]

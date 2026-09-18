import json
import subprocess
import sys

from jev_clone.engine import SystemOneEngine
from jev_clone.ledger_report import render, summarize
from jev_clone.presets import GUARDRAILS, ROUTING, TICKET_TRIAGE, extraction, rerank, skill_selection
from jev_clone.schema import SystemOneRequest
from tests.conftest import MockBackend


def test_presets_validate():
    for qs in (ROUTING, GUARDRAILS, TICKET_TRIAGE, skill_selection({"search": "web", "code": "write code"}),
               rerank("q", ["a", "b"]), extraction({"amount_band": {"type": "choice", "instructions": "?", "criteria": ["<100", "100-1000", ">1000"]}})):
        req = SystemOneRequest(state="x", questions=qs)
        assert "risk" in req.questions or all(k.startswith("p") for k in req.questions)  # rerank n'a pas de meta
    be = MockBackend({"route": [0.8, 0.15, 0.05]}); be.declared = {"route": ["auto", "reasoning_model", "human"]}
    r = SystemOneEngine(be, model_name="mock").answer({"state": "x", "questions": ROUTING})
    assert r.answers["route"].choice == "auto" and "needs_reasoning" in r.answers


def test_ledger_report():
    rows = [{"path": "system_one", "latency_ms": 120, "gated": {"team": True, "risk": True}},
            {"path": "system_one", "latency_ms": 140, "gated": {"team": True, "risk": True}},
            {"path": "escalated", "latency_ms": 9000, "gated": {"team": False, "risk": True}}]
    s = summarize(rows, s2_tokens=1000, s2_ms=10000, s2_cost=0.01)
    assert s["items"] == 3 and s["escalated"] == 1 and abs(s["escalation_rate"] - 1 / 3) < 1e-9
    assert s["saved_llm_calls"] == 2 and s["saved_tokens_est"] == 2000 and abs(s["saved_cost_est"] - 0.02) < 1e-9
    assert abs(s["gate_rate_per_question"]["team"] - 2 / 3) < 1e-9
    assert "taux d'escalade=33.3%" in render(s)


def test_make_from_history(tmp_path):
    csv_path = tmp_path / "h.csv"
    csv_path.write_text("text,team,refund_flag,priority,outcome\n"
                        "charged twice,Billing,yes,high,resolved\n"
                        "app crashes,technical,no,1,resolved\n"
                        "unknown,marketing,no,low,resolved\n"
                        "bad outcome,billing,yes,low,reopened\n")
    schema = {"team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": None, "technical": None}, "col": "team"},
              "refund": {"type": "noul", "instructions": "Refund?", "col": "refund_flag"},
              "priority": {"type": "score", "instructions": "Urgency?", "criteria": ["low", "mid", "high"], "col": "priority"}}
    (tmp_path / "s.json").write_text(json.dumps(schema))
    out = tmp_path / "out.jsonl"
    subprocess.run([sys.executable, "training/make_from_history.py", "--in", str(csv_path), "--state-col", "text", "--schema",
                    str(tmp_path / "s.json"), "--out", str(out), "--outcome-col", "outcome"], check=True, capture_output=True)
    rows = [json.loads(l) for l in open(out) if l.strip()]
    assert len(rows) == 3  # 'reopened' ecartee ; 'marketing' garde refund/priority sans team
    by_state = {r["state"]: r["labels"] for r in rows}
    assert by_state["charged twice"] == {"team": "billing", "refund": True, "priority": 2}
    assert by_state["app crashes"] == {"team": "technical", "refund": False, "priority": 1}
    assert "team" not in by_state["unknown"] and by_state["unknown"]["priority"] == 0
    SystemOneRequest.model_validate({"state": rows[0]["state"], "questions": rows[0]["questions"]})

import json

from jev_clone.engine import SystemOneEngine
from jev_clone.fusion import FusionRouter, GatePolicy, _extract_json
from tests.conftest import MockBackend


def test_gate_passes_when_confident(mock_backend, ticket_request, tmp_path):
    eng = SystemOneEngine(mock_backend, model_name="mock")
    s2 = MockBackend({}, chat_reply="unused")
    router = FusionRouter(eng, s2, GatePolicy(default_threshold=0.25), ledger=tmp_path / "ledger.jsonl")
    r = router.decide(ticket_request)
    assert r.path == "system_one" and all(r.gated.values())
    assert r.decisions["team"] == "payments" and r.decisions["escalate"] is True
    assert not s2.chat_calls
    rec = json.loads((tmp_path / "ledger.jsonl").read_text().splitlines()[0])
    assert rec["path"] == "system_one" and rec["decisions"]["team"] == "payments"


def test_escalates_and_uses_bonsai_answer(mock_backend, ticket_request, tmp_path):
    eng = SystemOneEngine(mock_backend, model_name="mock")
    s2 = MockBackend({}, chat_reply='After thinking... final: {"team": "account", "escalate": false, "urgency": 1}')
    policy = GatePolicy(default_threshold=0.99, risk_question="urgency", budgets=(0, 512, 4096))
    router = FusionRouter(eng, s2, policy, ledger=tmp_path / "ledger.jsonl")
    r = router.decide(ticket_request)
    assert r.path == "escalated" and not any(r.gated.values())
    assert r.decisions["team"] == "account" and r.decisions["escalate"] is False
    # risque = round(score 1.6) = 2 -> budget 4096
    assert s2.chat_calls[0][1]["thinking_budget"] == 4096
    assert r.s2_reasoning == "thinking..."


def test_forced_escalation_and_verification(mock_backend, ticket_request):
    eng = SystemOneEngine(mock_backend, model_name="mock")
    s2 = MockBackend({}, chat_reply='{"escalate": true}')
    policy = GatePolicy(default_threshold=0.0, always_escalate_if={"escalate": True}, verify_with_s1=True)
    mock_backend.table["consistent"] = [0.8, 0.2]
    router = FusionRouter(eng, s2, policy)
    r = router.decide(ticket_request)
    assert r.path == "escalated" and r.gated["escalate"] is False and r.gated["team"] is True
    assert r.verification["noul"] == 0.8


def test_no_s2_backend_keeps_s1(mock_backend, ticket_request):
    router = FusionRouter(SystemOneEngine(mock_backend, model_name="mock"), None, GatePolicy(default_threshold=0.99))
    r = router.decide(ticket_request)
    assert r.path == "system_one" and r.decisions["team"] == "payments"


def test_extract_json():
    assert _extract_json('blah {"a": 1} end') == {"a": 1}
    assert _extract_json('x {"a": {"b": 2}} {"c": 3}') == {"c": 3}
    assert _extract_json("no json") == {}

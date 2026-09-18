import pytest

from jev_clone.prompt import PromptFormat, build_branches, label_grammar
from jev_clone.schema import SystemOneRequest


def test_request_validation(ticket_request):
    req = SystemOneRequest.model_validate(ticket_request)
    assert set(req.questions) == {"team", "escalate", "urgency"}
    assert req.questions["team"].type == "choice"


def test_choice_limits():
    with pytest.raises(ValueError):
        SystemOneRequest.model_validate({"state": "x", "questions": {"q": {"type": "choice", "instructions": "?", "criteria": ["only"]}}})
    with pytest.raises(ValueError):
        SystemOneRequest.model_validate({"state": "x", "questions": {"q": {"type": "choice", "instructions": "?",
                                                                            "criteria": [str(i) for i in range(27)]}}})
    with pytest.raises(ValueError):
        SystemOneRequest.model_validate({"state": "x", "questions": {"q": {"type": "score", "instructions": "?", "criteria": ["one"]}}})
    with pytest.raises(ValueError):
        SystemOneRequest.model_validate({"state": "x", "questions": {}})


def test_prefix_is_shared_and_branches_isolated(ticket_request):
    req = SystemOneRequest.model_validate(ticket_request)
    fmt = PromptFormat()
    prefix = fmt.prefix(req.state)
    assert prefix.startswith("<|im_start|>system") and "# State" in prefix and "payouts" in prefix
    branches = []
    for qid, q in req.questions.items():
        branches += build_branches(qid, q, fmt)
    assert len(branches) == 3
    for b in branches:
        assert "# Question" in b.text and b.text.endswith("<think>\n\n</think>\n\n")
        assert "payouts have failed" not in b.text  # l'etat n'est que dans le prefixe
    team = next(b for b in branches if b.qid == "team")
    assert team.labels == ["A", "B", "C"] and team.keys == ["payments", "account", "other"]
    assert "A. payments: payouts, refunds" in team.text and "C. other" in team.text
    noul = next(b for b in branches if b.qid == "escalate")
    assert noul.labels == ["Yes", "No"] and noul.keys == [True, False]
    score = next(b for b in branches if b.qid == "urgency")
    assert "(level 0 of 2) can wait a week" in score.text and score.keys == [0, 1, 2]


def test_permutations_change_order_but_keep_keys(ticket_request):
    req = SystemOneRequest.model_validate(ticket_request)
    brs = build_branches("team", req.questions["team"], PromptFormat(), permutations=4)
    assert len(brs) == 4
    assert brs[0].keys == ["payments", "account", "other"]
    assert any(b.keys != brs[0].keys for b in brs[1:])
    assert all(sorted(b.keys) == sorted(brs[0].keys) for b in brs)


def test_grammar():
    assert label_grammar(["A", "B", "C"]) == 'root ::= "A" | "B" | "C"'
    assert label_grammar(["Yes", "No"]) == 'root ::= "Yes" | "No"'


def test_base_model_format():
    fmt = PromptFormat(chat=False)
    assert "<|im_start|>" not in fmt.prefix("s")
    assert fmt.branch("body").endswith("Answer:")

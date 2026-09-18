import json

from jev_clone.jarvis import JARVIS_TURN
from jev_clone.schema import SystemOneRequest
from training.make_synthetic_jarvis import expand


class FakeChat:
    def __init__(self):
        self.calls = []

    def chat(self, messages, **kw):
        self.calls.append((messages, kw))
        return {"choices": [{"message": {"content": json.dumps({"requests": ["Fais une appli météo", "Build a weather app", " ", "Une app"]})}}]}


def test_seeds_are_valid_examples():
    rows = [json.loads(l) for l in open("training/seeds/jarvis_seeds.jsonl") if l.strip()]
    assert len(rows) >= 80
    intents = {r["labels"]["intent"] for r in rows}
    assert intents == {"chat", "create_app", "modify_code", "run_command", "browse", "remember", "other"}
    for r in rows:
        SystemOneRequest(state=r["state"], questions=JARVIS_TURN)
        assert r["labels"]["intent"] in JARVIS_TURN["intent"]["criteria"] and r["labels"]["language"] in JARVIS_TURN["language"]["criteria"]
        assert 0 <= r["labels"]["risk"] <= 3
    assert sum(1 for r in rows if r["labels"]["risk"] == 3) >= 8   # des cas dangereux pour apprendre la porte


def test_expand_uses_json_schema_and_filters():
    fc = FakeChat()
    seed = {"state": {"request": "Crée une appli météo"}, "labels": {"intent": "create_app", "language": "python", "risk": 0}}
    out = expand(fc, seed, 3)
    assert out == ["Fais une appli météo", "Build a weather app", "Une app"]
    kw = fc.calls[0][1]
    assert kw["extra"]["response_format"]["type"] == "json_schema" and kw["thinking_budget"] == 0
    assert "create_app" in fc.calls[0][0][0]["content"]

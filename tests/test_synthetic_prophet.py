import json

from jev_clone.prophet import PROPHET_TURN
from jev_clone.schema import SystemOneRequest
from training.make_synthetic_prophet import expand


class FakeChat:
    def __init__(self):
        self.calls = []

    def chat(self, messages, **kw):
        self.calls.append((messages, kw))
        return {"choices": [{"message": {"content": json.dumps({"requests": ["Fais une appli météo", "Build a weather app", " ", "Une app"]})}}]}


def test_seeds_are_valid_examples():
    rows = [json.loads(l) for l in open("training/seeds/prophet_seeds.jsonl") if l.strip()]
    assert len(rows) >= 80
    assert {r["labels"]["intent"] for r in rows} == set(PROPHET_TURN["intent"]["criteria"])
    for r in rows:
        SystemOneRequest(state=r["state"], questions=PROPHET_TURN)
        assert r["labels"]["language"] in PROPHET_TURN["language"]["criteria"] and 0 <= r["labels"]["risk"] <= 3
        assert isinstance(r["labels"]["direct"], bool) and isinstance(r["labels"]["clarify"], bool)
    assert sum(1 for r in rows if r["labels"]["risk"] == 3) >= 8 and sum(1 for r in rows if r["labels"]["direct"]) >= 8


def test_expand():
    fc = FakeChat()
    seed = {"state": {"request": "Crée une appli météo"}, "labels": {"intent": "create_app", "language": "python", "risk": 0}}
    assert expand(fc, seed, 3) == ["Fais une appli météo", "Build a weather app", "Une app"]
    assert fc.calls[0][1]["extra"]["response_format"]["type"] == "json_schema" and "Prophet" in fc.calls[0][0][0]["content"]

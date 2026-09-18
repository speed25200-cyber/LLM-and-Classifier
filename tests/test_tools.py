import json

from jev_clone.engine import SystemOneEngine
from jev_clone.tools import AgentLoop, SystemOneToolbox
from tests.conftest import MockBackend, MockS2


def _engine():
    be = MockBackend({"q": [0.8, 0.15, 0.05], "c0": [0.9, 0.1], "c1": [0.2, 0.8], "c2": [0.6, 0.4]})
    be.declared = {"q": ["billing", "technical", "other"]}
    return SystemOneEngine(be, model_name="mock"), be


def test_definitions():
    tb = SystemOneToolbox(_engine()[0])
    names = [d["function"]["name"] for d in tb.definitions()]
    assert names == ["judge_choice", "judge_noul", "judge_score", "judge_rank", "judge_batch"]
    assert all(d["function"]["parameters"]["type"] == "object" for d in tb.definitions())


def test_calls():
    eng, be = _engine()
    tb = SystemOneToolbox(eng)
    r = tb.call("judge_choice", {"state": '{"ticket": "charged twice"}', "question": "Which team?", "options": ["billing", "technical", "other"]})
    assert r["choice"] == "billing" and abs(sum(r["probabilities"].values()) - 1) < 1e-6
    assert json.loads(be.calls[-1][0].split("# State\n")[1].split("\n\n")[0])["ticket"] == "charged twice"  # etat JSON decode
    be.table.pop("q")  # plus d'entree "q" : le noul retombe sur default_noul
    r = tb.call("judge_noul", {"state": "x", "question": "Is it urgent?"})
    assert abs(r["noul"] - 0.3) < 1e-6
    be.table["q"] = [0.1, 0.2, 0.7]
    r = tb.call("judge_score", {"state": "x", "question": "How urgent?", "levels": ["low", "mid", "high"]})
    assert abs(r["score"] - 1.6) < 1e-6
    r = tb.call("judge_rank", {"state": "x", "question": "Is this the element?", "candidates": ["A", "B", "C"]})
    assert [x["candidate"] for x in r["ranked"]] == ["A", "C", "B"]
    r = tb.call("judge_batch", {"state": "x", "questions": json.dumps({"c0": {"type": "noul", "instructions": "?"}})})
    assert abs(r["c0"]["noul"] - 0.9) < 1e-6


def test_agent_loop_tool_then_final(tmp_path):
    eng, _ = _engine()
    s2 = MockS2([
        {"content": "", "tool_calls": [MockS2.tool_call("judge_noul", {"state": "s", "question": "ok?"}),
                                       MockS2.tool_call("lookup", {"id": 7}, "call_2")]},
        {"content": "Final answer: yes", "reasoning_content": "..."},
    ])
    seen = []
    loop = AgentLoop(s2, SystemOneToolbox(eng), {"lookup": ({"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}}},
                                                          lambda a: seen.append(a) or {"record": a["id"]})},
                     max_turns=5, ledger=tmp_path / "agent.jsonl")
    res = loop.run([{"role": "user", "content": "go"}])
    assert res.stopped_by == "final" and res.content == "Final answer: yes"
    assert seen == [{"id": 7}]
    assert res.steps[0].tool_calls[0]["name"] == "judge_noul" and "noul" in res.steps[0].tool_calls[0]["result"]
    tool_msgs = [m for m in res.messages if m["role"] == "tool"]
    assert len(tool_msgs) == 2 and tool_msgs[1]["tool_call_id"] == "call_2"
    names = [d["function"]["name"] for d in s2.calls[0][1]["tools"]]
    assert "judge_rank" in names and "lookup" in names
    assert (tmp_path / "agent.jsonl").read_text().count("\n") == 1


def test_agent_loop_stop_tool_and_max_turns():
    eng, _ = _engine()
    s2 = MockS2([{"content": "", "tool_calls": [MockS2.tool_call("done", {"summary": "ok"})]}])
    loop = AgentLoop(s2, SystemOneToolbox(eng), {"done": ({"type": "function", "function": {"name": "done", "parameters": {"type": "object", "properties": {}}}},
                                                        lambda a: {"__stop__": True, **a})}, max_turns=3)
    assert loop.run([{"role": "user", "content": "go"}]).stopped_by == "stop_tool"
    s2 = MockS2([{"content": "", "tool_calls": [MockS2.tool_call("judge_noul", {"state": "s", "question": "?"})]}])
    res = AgentLoop(s2, SystemOneToolbox(eng), max_turns=3).run([{"role": "user", "content": "go"}])
    assert res.stopped_by == "max_turns" and len(res.steps) == 3
    s2 = MockS2([{"content": "", "tool_calls": [MockS2.tool_call("nope", {})]}, {"content": "end"}])
    res = AgentLoop(s2, SystemOneToolbox(eng), max_turns=3).run([{"role": "user", "content": "go"}])
    assert "error" in res.steps[0].tool_calls[0]["result"]

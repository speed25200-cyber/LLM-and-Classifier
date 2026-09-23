import json

import pytest

from jev_clone.engine import SystemOneEngine
from jev_clone.prophet import CORE_TOOLS, INTENTS, LANGUAGES, Prophet, Workspace
from tests.conftest import MockS2, ScriptedBackend

I, L = list(INTENTS), list(LANGUAGES)
TR = ["readonly", "workspace_write", "destructive", "privileged", "exfiltration"]


def s1(direct=0.1, clarify=0.1, reasoning=0.2, risk=0, tool_risk="readonly", cmd_risk=0, ok=0.9, relevant=("write_file", "run_command")):
    pre = {"direct": [direct, 1 - direct], "clarify": [clarify, 1 - clarify], "needs_reasoning": [reasoning, 1 - reasoning],
           "risk": [1.0 if k == risk else 0.0 for k in range(4)],
           "intent": [0.9 if i == "create_app" else 0.1 / 6 for i in I], "language": [0.9 if l == "python" else 0.1 / 5 for l in L]}
    guard = {"tool_risk": [0.9 if t == tool_risk else 0.1 / 4 for t in TR], "risk": [1.0 if k == cmd_risk else 0.0 for k in range(4)],
             "policy_violation": [0.1, 0.9], "ok": [ok, 1 - ok]}
    rel = {f"t_{n}": [0.9, 0.1] for n in relevant}
    be = ScriptedBackend([pre, dict(guard, **rel)] + [dict(guard, **rel)] * 20, declared={"intent": I, "language": L, "tool_risk": TR}, default_noul=(0.2, 0.8))
    return SystemOneEngine(be, model_name="mock")


def test_direct_path(tmp_path):
    ws = Workspace(tmp_path / "ws")
    s2 = MockS2([{"content": "Canberra."}])
    t = Prophet(s1(direct=0.95), s2, ws).handle("Capitale de l'Australie ?")
    assert t.path == "direct" and t.response == "Canberra." and s2.calls[0][1]["thinking_budget"] == 0 and t.verification == 0.9


def test_agent_path_creates_app_and_selects_tools(tmp_path):
    ws = Workspace(tmp_path / "ws")
    s2 = MockS2([
        {"content": "", "tool_calls": [MockS2.tool_call("write_file", {"path": "app/main.py", "content": "print('hi')\n"}),
                                       MockS2.tool_call("run_command", {"command": "python3 app/main.py"}, "c2")]},
        {"content": "", "tool_calls": [MockS2.tool_call("done", {"summary": "app/main.py created; run: python3 app/main.py"}, "c3")]},
    ])
    p = Prophet(s1(), s2, ws, max_tools=5)
    t = p.handle("Crée un script python qui dit hi")
    assert t.path == "agent" and t.response.startswith("app/main.py created") and (ws.root / "app/main.py").exists()
    assert [c["tool"] for c in t.tool_calls] == ["write_file", "run_command", "done"] and t.tool_calls[1]["ok"] is True
    # outils exposes : les 2 pertinents en tete, les outils de base toujours presents, le reste filtre (max_tools=5)
    assert t.tools_exposed[:2] == ["write_file", "run_command"] and set(CORE_TOOLS) <= set(t.tools_exposed) and len(t.tools_exposed) == 5
    names = [d["function"]["name"] for d in s2.calls[0][1]["tools"]]
    assert "write_file" in names and "judge_rank" in names and "browse" not in names
    assert "Fast judge" in s2.calls[0][0][-1]["content"] and "Prophet" in s2.calls[0][0][0]["content"]
    rec = json.loads(open(ws.meta / "ledger.jsonl").readline())
    assert rec["pre"]["tool_relevance"]["write_file"] == 0.9 and rec["path"] == "agent"


def test_create_tool_then_use_it(tmp_path):
    ws = Workspace(tmp_path / "ws")
    body = "import math\nreturn {'ok': True, 'sqrt': math.sqrt(float(args['x']))}"
    s2 = MockS2([
        {"content": "", "tool_calls": [MockS2.tool_call("create_tool", {"name": "sqrt_tool", "description": "square root of x",
                                                                          "parameters": {"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"]},
                                                                          "python_body": body})]},
        {"content": "", "tool_calls": [MockS2.tool_call("done", {"summary": "tool created"}, "c2")]},
    ])
    p = Prophet(s1(relevant=("create_tool",)), s2, ws, confirm=lambda d, j: True)
    t = p.handle("Fais-toi un outil racine carrée")
    assert t.tool_calls[0]["tool"] == "create_tool" and t.tool_calls[0]["ok"] is True and (ws.skills_dir / "sqrt_tool.py").exists()
    # tour suivant : la competence est chargee et appelable
    s2 = MockS2([{"content": "", "tool_calls": [MockS2.tool_call("sqrt_tool", {"x": 16})]},
                 {"content": "", "tool_calls": [MockS2.tool_call("done", {"summary": "4.0"}, "c2")]}])
    p = Prophet(s1(relevant=("sqrt_tool",)), s2, ws, confirm=lambda d, j: True)
    t = p.handle("racine de 16 ?")
    tool_msg = json.loads(s2.calls[1][0][-1]["content"])
    assert tool_msg["sqrt"] == 4.0 and "sqrt_tool" in t.tools_exposed
    # une competence cassee ne bloque pas
    (ws.skills_dir / "bad.py").write_text("this is not python (")
    assert any(n.startswith("broken_bad") for n in ws.load_skills())


def test_guarded_actions(tmp_path):
    ws = Workspace(tmp_path / "ws")
    s2 = MockS2([{"content": "", "tool_calls": [MockS2.tool_call("python", {"code": "import os; os.system('rm -rf /')"})]},
                 {"content": "", "tool_calls": [MockS2.tool_call("done", {"summary": "x"}, "c2")]}])
    t = Prophet(s1(tool_risk="destructive", cmd_risk=3, relevant=("python",)), s2, ws).handle("nettoie tout")
    assert t.tool_calls[0]["blocked"] is True and t.tool_calls[0]["judged"]["needs_confirmation"]
    seen = []
    s2 = MockS2([{"content": "", "tool_calls": [MockS2.tool_call("python", {"code": "print(6*7)"})]},
                 {"content": "", "tool_calls": [MockS2.tool_call("done", {"summary": "42"}, "c2")]}])
    t = Prophet(s1(tool_risk="destructive", cmd_risk=3, relevant=("python",)), s2, ws, confirm=lambda d, j: seen.append(j["tool_risk"]) or True).handle("calcule 6*7")
    out = json.loads(s2.calls[1][0][-1]["content"])
    assert out["stdout"].strip() == "42" and seen == ["destructive"]


def test_workspace_memory_and_browse_without_browser(tmp_path):
    ws = Workspace(tmp_path / "ws")
    with pytest.raises(PermissionError):
        ws.write("../out.txt", "x")
    ws.remember("prefers TypeScript")
    s2 = MockS2([{"content": "", "tool_calls": [MockS2.tool_call("browse", {"goal": "find docs"})]},
                 {"content": "", "tool_calls": [MockS2.tool_call("done", {"summary": "x"}, "c2")]}])
    t = Prophet(s1(relevant=("browse",)), s2, ws).handle("cherche la doc")
    assert "prefers TypeScript" in s2.calls[0][0][0]["content"]
    # sans navigateur configure, browse n'est plus propose au modele (un appel force reste un outil inconnu)
    assert "browse" not in [d["function"]["name"] for d in s2.calls[0][1]["tools"]] and "browse" not in t.tools_exposed
    assert json.loads(s2.calls[1][0][-1]["content"])["error"] == "unknown tool browse"

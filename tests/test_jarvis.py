import json

import pytest

from jev_clone.engine import SystemOneEngine
from jev_clone.jarvis import INTENTS, LANGUAGES, Jarvis, Workspace
from tests.conftest import MockS2, ScriptedBackend

I, L = list(INTENTS), list(LANGUAGES)
TR = ["readonly", "destructive", "privileged", "exfiltration"]


def s1(intent="create_app", clarify=0.1, reasoning=0.2, risk=0, tool_risk="readonly", cmd_risk=0, ok=0.9):
    """S1 factice : pre-traitement, puis (eventuellement) garde-fous de commandes, puis verification."""
    pre = {"intent": [0.9 if i == intent else 0.1 / 6 for i in I], "language": [0.9 if l == "python" else 0.1 / 5 for l in L],
           "clarify": [clarify, 1 - clarify], "needs_reasoning": [reasoning, 1 - reasoning],
           "risk": [1.0 if k == risk else 0.0 for k in range(4)]}
    guard = {"tool_risk": [0.9 if t == tool_risk else 0.1 / 3 for t in TR], "risk": [1.0 if k == cmd_risk else 0.0 for k in range(4)],
             "policy_violation": [0.1, 0.9]}
    verif = {"ok": [ok, 1 - ok]}
    be = ScriptedBackend([pre, guard, guard, guard, verif, verif], declared={"intent": I, "language": L, "tool_risk": TR})
    # la table "guard" sert aussi de defaut pour la verification si elle vient plus tot : les qids ne se recouvrent pas
    be.tables = [pre] + [dict(guard, **verif)] * 8
    return SystemOneEngine(be, model_name="mock")


def test_create_app_path(tmp_path):
    ws = Workspace(tmp_path / "ws")
    s2 = MockS2([
        {"content": "", "tool_calls": [MockS2.tool_call("write_file", {"path": "app/main.py", "content": "print('hello from jarvis')\n"}),
                                       MockS2.tool_call("run_command", {"command": "python3 app/main.py"}, "call_2")]},
        {"content": "", "tool_calls": [MockS2.tool_call("done", {"summary": "Created app/main.py; run: python3 app/main.py"}, "call_3")]},
    ])
    j = Jarvis(s1(), s2, ws)
    t = j.handle("Crée-moi un petit script python qui dit bonjour")
    assert t.path == "agent" and t.response.startswith("Created app/main.py")
    assert (ws.root / "app" / "main.py").exists()
    assert [c["tool"] for c in t.tool_calls] == ["write_file", "run_command", "done"]
    assert t.tool_calls[1]["ok"] is True and t.tool_calls[1]["judged"]["tool_risk"] == "readonly"
    assert t.verification == 0.9
    rec = json.loads(open(ws.root / ".jarvis" / "ledger.jsonl").readline())
    assert rec["path"] == "agent" and rec["pre"]["intent"]["choice"] == "create_app"
    # le prompt systeme de Bonsai contient les regles et le chemin de l'espace de travail
    assert str(ws.root) in s2.calls[0][0][0]["content"] and "Judged intent: create_app" in s2.calls[0][0][-1]["content"]


def test_chat_and_clarify_paths(tmp_path):
    ws = Workspace(tmp_path / "ws")
    s2 = MockS2([{"content": "Bonjour ! Je suis Jarvis."}])
    t = Jarvis(s1(intent="chat"), s2, ws).handle("salut ça va ?")
    assert t.path == "chat" and t.response == "Bonjour ! Je suis Jarvis." and s2.calls[0][1]["thinking_budget"] == 0
    s2 = MockS2([{"content": "Quelle technologie préférez-vous : web ou bureau ?"}])
    t = Jarvis(s1(intent="create_app", clarify=0.9), s2, ws).handle("fais-moi une app")
    assert t.path == "clarify" and "?" in t.response


def test_risky_command_needs_confirmation(tmp_path):
    ws = Workspace(tmp_path / "ws")
    calls = [MockS2.tool_call("run_command", {"command": "rm -rf build"})]
    s2 = MockS2([{"content": "", "tool_calls": calls}, {"content": "", "tool_calls": [MockS2.tool_call("done", {"summary": "ok"}, "c2")]}])
    j = Jarvis(s1(tool_risk="destructive", cmd_risk=3), s2, ws)   # pas de callback : refus
    t = j.handle("nettoie le dossier build")
    assert t.tool_calls[0]["blocked"] is True
    res = json.loads(s2.calls[1][0][-1]["content"]) if s2.calls[1][0][-1]["role"] == "tool" else None
    assert res and res["blocked"] and "confirmation" in res["reason"]
    seen = []
    s2 = MockS2([{"content": "", "tool_calls": calls}, {"content": "", "tool_calls": [MockS2.tool_call("done", {"summary": "ok"}, "c2")]}])
    j = Jarvis(s1(tool_risk="destructive", cmd_risk=3), s2, ws, confirm=lambda cmd, judged: seen.append((cmd, judged["tool_risk"])) or True)
    t = j.handle("nettoie le dossier build")
    assert seen == [("rm -rf build", "destructive")] and t.tool_calls[0].get("ok") is True


def test_workspace_sandbox_and_memory(tmp_path):
    ws = Workspace(tmp_path / "ws")
    with pytest.raises(PermissionError):
        ws.write("../outside.txt", "x")
    ws.write("a/b.txt", "hello"); assert ws.read("a/b.txt")["content"] == "hello" and "a/b.txt" in ws.listing()["entries"]
    assert ws.run("echo hi")["stdout"].strip() == "hi" and ws.run("exit 3")["returncode"] == 3
    ws.remember("prefers French"); assert ws.memory() == ["prefers French"]
    s2 = MockS2([{"content": "ok"}])
    Jarvis(s1(intent="chat"), s2, ws).handle("hi")
    assert "prefers French" in s2.calls[0][0][0]["content"]

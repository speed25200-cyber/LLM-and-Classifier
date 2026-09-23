"""Computer use sur le bureau : arbre UI Automation -> etat, boucle a deux vitesses sur un bureau simule, outil Prophet."""

import json
import types

from jev_clone.computer_use import ACTIONS, ComputerUseAgent, FastPolicy
from jev_clone.desktop_use import (DesktopSession, DesktopSlowPolicy, SimulatedDesktop, keys_to_sendkeys, make_desktop_factory, nodes_to_state,
                                   walk_uia)
from jev_clone.engine import SystemOneEngine
from jev_clone.prophet import Prophet, Workspace
from jev_clone.tools import SystemOneToolbox
from tests.conftest import MockS2, ScriptedBackend

A = list(ACTIONS)


def test_keys_to_sendkeys():
    assert keys_to_sendkeys("ctrl+shift+s") == "{Ctrl}{Shift}s"
    assert keys_to_sendkeys("Alt + F4") == "{Alt}{F4}"
    assert keys_to_sendkeys("enter") == "{Enter}"


def ctl(t, name, children=(), rect=(0, 0, 100, 30), off=False, value=None):
    r = types.SimpleNamespace(left=rect[0], top=rect[1], right=rect[0] + rect[2], bottom=rect[1] + rect[3])
    c = types.SimpleNamespace(ControlTypeName=t, Name=name, BoundingRectangle=r, IsOffscreen=off, IsEnabled=True,
                              GetChildren=lambda: list(children))
    if value is not None:
        c.GetValuePattern = lambda: types.SimpleNamespace(Value=value)
    return c


def test_walk_uia_tree_to_page_state():
    win = ctl("WindowControl", "Facture - Excel", [
        ctl("ToolBarControl", "Ruban", [ctl("ButtonControl", "Enregistrer"), ctl("ButtonControl", "Cache", off=True)]),
        ctl("PaneControl", "", [ctl("EditControl", "Cellule A1", value="1250"), ctl("TextControl", ""), ctl("ButtonControl", "Zero", rect=(0, 0, 0, 0))]),
    ])
    nodes = walk_uia(win)
    assert [n.name for n in nodes] == ["Facture - Excel", "Ruban", "Enregistrer", "", "Cellule A1", "Zero"]
    state, targets = nodes_to_state("Facture - Excel", "excel.exe", nodes)
    assert state.url == "app://excel.exe" and [e.line() for e in state.elements] == ['[0] button "Enregistrer"', '[1] textbox "Cellule A1" value="1250"']
    assert "- textbox \"Cellule A1\" = '1250'" in state.aria and targets[1].value == "1250"


def plan(steps):
    """Tables S1 : pour chaque pas (action, cible) une decision puis une verification reussie."""
    tables = []
    for action, target in steps:
        t = {"action": [0.9 if a == action else 0.02 for a in A]}
        if target is not None:
            t[f"t{target}"] = [0.95, 0.05]
        tables += [t, {"ok": [0.9, 0.1]}]
    return tables


def test_fast_policy_drives_simulated_calculator():
    desk = SimulatedDesktop()
    # boutons de la calculatrice : 7 8 9 4 5 6 1 2 3 0 + - x / Egal Effacer -> indices 0..15
    s1 = SystemOneEngine(ScriptedBackend(plan([("click", 6), ("click", 10), ("click", 4), ("click", 14), ("done", None)]),
                                         declared={"action": A}, default_noul=(0.2, 0.8)), model_name="mock")
    session = DesktopSession(desk)
    out = ComputerUseAgent(session, FastPolicy(s1), None, max_steps=8, ledger=None).run("calcule 1 + 5")
    assert out["status"] == "done" and desk.display == "6"
    assert desk.log == ["click:1", "click:+", "click:5", "click:Egal"]
    assert all(r["path"] == "fast" for r in out["records"])


def test_escalation_to_bonsai_with_desktop_tools():
    desk = SimulatedDesktop()
    s1 = SystemOneEngine(ScriptedBackend([{"action": [0.02, 0.02, 0.02, 0.02, 0.02, 0.9]}], declared={"action": A}), model_name="mock")
    s2 = MockS2([
        {"content": "", "tool_calls": [MockS2.tool_call("open_app", {"name": "notepad"})]},
        {"content": "", "tool_calls": [MockS2.tool_call("type", {"index": 0, "text": "Liste de courses"}, "c2")]},
        {"content": "", "tool_calls": [MockS2.tool_call("press_keys", {"keys": "ctrl+s"}, "c3")]},
        {"content": "", "tool_calls": [MockS2.tool_call("done", {"summary": "note enregistree"}, "c4")]},
    ])
    session = DesktopSession(desk)
    slow = DesktopSlowPolicy(s2, session, SystemOneToolbox(s1))
    out = ComputerUseAgent(session, FastPolicy(s1), slow, max_steps=4, ledger=None).run("ecris une note et enregistre-la")
    assert out["status"] == "done" and desk.front == "notepad" and desk.notes == "Liste de courses" and desk.saved
    names = [d["function"]["name"] for d in s2.calls[0][1]["tools"]]
    assert "press_keys" in names and "open_app" in names


def test_desktop_factory_and_prophet_tool_are_guarded():
    run = make_desktop_factory(None, None, SimulatedDesktop)()
    assert callable(run)
    got = {}

    def fake_factory():
        def go(goal, app, slots):
            got.update(goal=goal, app=app)
            return {"ok": True, "status": "done", "steps": 3, "fast_steps": 3, "window": "Calculatrice"}
        return go
    tr = ["readonly", "workspace_write", "destructive", "privileged", "exfiltration"]
    pre = {"direct": [0.1, 0.9], "clarify": [0.1, 0.9], "needs_reasoning": [0.2, 0.8], "risk": [1, 0, 0, 0]}
    guard = {"tool_risk": [0.1, 0.0, 0.8, 0.05, 0.05], "risk": [0, 0, 1, 0], "policy_violation": [0.1, 0.9], "ok": [0.9, 0.1], "t_desktop": [0.9, 0.1]}
    s1 = SystemOneEngine(ScriptedBackend([pre, guard] + [guard] * 10, declared={"tool_risk": tr}, default_noul=(0.2, 0.8)), model_name="mock")
    s2 = MockS2([{"content": "", "tool_calls": [MockS2.tool_call("desktop", {"goal": "calcule 1+5", "app": "calc"})]},
                 {"content": "", "tool_calls": [MockS2.tool_call("done", {"summary": "6"}, "c2")]}])
    asked = []
    import tempfile
    ws = Workspace(tempfile.mkdtemp())
    t = Prophet(s1, s2, ws, confirm=lambda d, j: asked.append(j) or True, desktop_factory=fake_factory).handle("fais le calcul sur la calculatrice")
    assert got == {"goal": "calcule 1+5", "app": "calc"} and asked and asked[0]["tool"] == "desktop"
    assert json.loads(s2.calls[1][0][-1]["content"])["fast_steps"] == 3 and t.response == "6"
    # sans fabrique, l'outil n'est pas propose au modele
    s2b = MockS2([{"content": "", "tool_calls": [MockS2.tool_call("done", {"summary": "x"})]}])
    Prophet(s1, s2b, ws).handle("fais le calcul")
    assert "desktop" not in [d["function"]["name"] for d in s2b.calls[0][1]["tools"]]

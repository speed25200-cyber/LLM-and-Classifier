import os

import pytest

from jev_clone.engine import SystemOneEngine
from tests.conftest import MockS2, ScriptedBackend

pytest.importorskip("playwright")
from jev_clone.computer_use import (ACTIONS, BrowserSession, ComputerUseAgent, FastPolicy, SlowPolicy,  # noqa: E402
                                    trajectory_to_examples)

HTML = """<html><body><h1>Newsletter</h1>
<a href="/about">About us</a>
<input id="email" placeholder="email address">
<button id="go" onclick="document.getElementById('out').innerText='submitted:'+document.getElementById('email').value">Subscribe</button>
<div id="out"></div></body></html>"""

A = list(ACTIONS)  # ordre declare : click, type, scroll_down, go_back, done, escalate


@pytest.fixture(scope="module")
def session():
    try:
        s = BrowserSession(headless=True)
    except Exception as e:  # pas de Chromium utilisable
        pytest.skip(f"navigateur indisponible: {e}")
    yield s
    s.close()


def test_failed_launch_releases_playwright(tmp_path, monkeypatch):
    # sans navigateur (runner Windows de la CI) : l'echec ne doit pas laisser la boucle de Playwright active,
    # sinon tout sync_playwright() suivant du meme fil echoue ("Sync API inside the asyncio loop")
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path / "vide"))
    with pytest.raises(Exception):
        BrowserSession(headless=True, chromium=str(tmp_path / "absent" / "chrome"))
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        assert p.chromium is not None


def _probs(action, target=None, slot=None):
    t = {"action": [0.9 if a == action else 0.1 / 5 for a in A]}
    if target is not None:
        t[f"t{target}"] = [0.95, 0.05]
    if slot:
        t["slot"] = [0.9, 0.1]  # declared ["email", "none"]
    return t


def test_observe(session):
    session.set_content(HTML)
    st = session.observe()
    assert st.title == "" or isinstance(st.title, str)
    roles = [e.role for e in st.elements]
    assert roles[0].startswith("a") and roles[1].startswith("input") and roles[2].startswith("button")
    assert st.elements[1].name == "email address" and st.elements[2].name == "Subscribe"
    assert "Subscribe" in st.aria and "[1] input" in st.text()["elements"][1]


def test_fast_policy_type_then_click_then_done(session, tmp_path):
    session.set_content(HTML)
    be = ScriptedBackend([_probs("type", target=1, slot="email"), {"ok": [0.9, 0.1]},
                          _probs("click", target=2), {"ok": [0.9, 0.1]},
                          _probs("done")],
                         declared={"action": A, "slot": ["email", "none"]})
    fast = FastPolicy(SystemOneEngine(be, model_name="mock"))
    agent = ComputerUseAgent(session, fast, slow=None, max_steps=5, ledger=tmp_path / "traj.jsonl")
    out = agent.run("Subscribe to the newsletter with my email", slots={"email": "ana@example.com"})
    assert out["status"] == "done" and len(out["history"]) == 2
    assert out["history"][0]["type"] == "type" and out["history"][0]["text"] == "ana@example.com" and out["history"][0]["ok"]
    assert out["history"][1]["type"] == "click" and out["history"][1]["target"] == 2
    assert session.page.locator("#out").inner_text() == "submitted:ana@example.com"
    assert out["records"][0]["verify"] == 0.9 and out["records"][0]["path"] == "fast"
    assert (tmp_path / "traj.jsonl").exists()
    ex = trajectory_to_examples(out)
    assert len(ex) == 2 and ex[0]["labels"]["action"] == "type" and ex[0]["labels"]["t1"] is True and ex[0]["labels"]["t2"] is False


def test_escalation_to_slow_policy(session, tmp_path):
    session.set_content(HTML)
    be = ScriptedBackend([_probs("escalate")], declared={"action": A})
    fast = FastPolicy(SystemOneEngine(be, model_name="mock"))
    s2 = MockS2([
        {"content": "", "tool_calls": [MockS2.tool_call("type", {"index": 1, "text": "bob@example.com"}),
                                       MockS2.tool_call("click", {"index": 2}, "call_2")]},
        {"content": "", "tool_calls": [MockS2.tool_call("done", {"summary": "subscribed"}, "call_3")]},
    ])
    slow = SlowPolicy(s2, session, thinking_budget=512, max_turns=4)
    agent = ComputerUseAgent(session, fast, slow, max_steps=3, ledger=tmp_path / "traj.jsonl")
    out = agent.run("Subscribe", slots={"email": "bob@example.com"})
    assert out["status"] == "done" and out["records"][0]["path"] == "escalated"
    assert [a["type"] for a in out["history"]] == ["type", "click", "done"]
    assert session.page.locator("#out").inner_text() == "submitted:bob@example.com"
    user_msg = s2.calls[0][0][1]["content"]
    assert "# Goal" in user_msg and "Subscribe" in user_msg and "[2] button" in user_msg
    names = [d["function"]["name"] for d in s2.calls[0][1]["tools"]]
    assert {"click", "type", "done", "judge_rank"} <= set(names) or {"click", "type", "done"} <= set(names)
    ex = trajectory_to_examples(out)
    assert ex and ex[0]["source"] == "bonsai" and ex[0]["labels"]["action"] == "type" and ex[0]["labels"]["t1"] is True


def test_uncertain_target_escalates_without_slow(session):
    session.set_content(HTML)
    be = ScriptedBackend([_probs("click")], declared={"action": A}, default_noul=(0.2, 0.8))  # aucun element sur
    agent = ComputerUseAgent(session, FastPolicy(SystemOneEngine(be, model_name="mock")), slow=None, max_steps=2, ledger=None)
    out = agent.run("Click subscribe")
    assert out["status"] == "needs_reasoning" and out["records"][0]["fast"]["escalate"]

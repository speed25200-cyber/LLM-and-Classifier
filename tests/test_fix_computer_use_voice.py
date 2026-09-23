"""Computer use (navigateur et bureau) et commandes vocales : regressions des ecarts corriges.

Porte sur probabilite + marge et "done" verifie ; garde-fou par pas avec confirmation ; annulation, progression,
plafond d'escalades, clone en panne ; boucle DAgger fermee (slow_state, slots, make_from_trajectories) ;
disponibilite du navigateur ; approbation vocale sur la seule phrase exacte."""

import json
import os
import random
import subprocess
import time
import types

import pytest

from jev_clone.computer_use import (ACTIONS, ComputerUseAgent, Element, FastPolicy, PageState, browser_available, run_result,
                                    trajectory_to_examples)
from jev_clone.desktop_use import DesktopSession, DesktopSlowPolicy, SimulatedDesktop, WindowsUIABackend, make_desktop_factory
from jev_clone.engine import SystemOneEngine
from jev_clone.prompt import PromptFormat, build_branches
from jev_clone.prophet import make_browser_factory
from jev_clone.schema import SystemOneRequest
from prophet_studio.voice import COMMANDS, route_utterance
from tests.conftest import MockBackend, MockS2, ScriptedBackend

A = list(ACTIONS)
TR = ["readonly", "destructive", "privileged", "exfiltration"]
SAFE = {"tool_risk": [0.9, 0.04, 0.03, 0.03], "risk": [0.9, 0.06, 0.02, 0.02], "policy_violation": [0.05, 0.95]}
RISKY = {"tool_risk": [0.05, 0.85, 0.05, 0.05], "risk": [0.05, 0.1, 0.25, 0.6], "policy_violation": [0.3, 0.7]}
# calculatrice simulee : 7 8 9 4 5 6 1 2 3 0 + - x / Egal Effacer -> indices 0..15
ONE, PLUS, FIVE, EQUALS, CLEAR = 6, 10, 4, 14, 15


def act(action, target=None, p=0.9, achieved=None):
    """Table de decision S1 : action choisie avec la probabilite p, element cible sur, noul objectif atteint."""
    t = {"action": [p if a == action else (1 - p) / (len(A) - 1) for a in A]}
    if target is not None:
        t[f"t{target}"] = [0.95, 0.05]
    if achieved is not None:
        t["achieved"] = [achieved, 1 - achieved]
    return t


class RuleBackend(MockBackend):
    """S1 factice : une regle par type d'appel (decision, garde-fou du pas, verification). Une regle est une table, une
    liste de tables (une par appel de ce type, la derniere reutilisee) ou une fonction du prefixe (l'etat rendu)."""

    def __init__(self, decide, guard=None, verify=None, default_noul=(0.2, 0.8)):
        super().__init__({}, default_noul=default_noul)
        self.rules = {"decide": decide, "guard": SAFE if guard is None else guard, "verify": verify or {"ok": [0.9, 0.1]}}
        self.declared = {"action": A, "tool_risk": TR}
        self.kinds: list[str] = []
        self.prefixes: list[str] = []

    def score_branches(self, prefix, branches):
        kind = "guard" if '"proposed_action"' in prefix else "verify" if '"before"' in prefix else "decide"
        n = self.kinds.count(kind)
        self.kinds.append(kind); self.prefixes.append(prefix)
        r = self.rules[kind]
        self.table = r(prefix) if callable(r) else r[min(n, len(r) - 1)] if isinstance(r, list) else r
        return super().score_branches(prefix, branches)


def s1(decide, **kw):
    be = RuleBackend(decide, **kw)
    return SystemOneEngine(be, model_name="mock"), be


def tc(name, args, cid="c1"):
    return {"content": "", "tool_calls": [MockS2.tool_call(name, args, cid)]}


def agent(desk, engine, s2=None, **kw):
    session = DesktopSession(desk)
    slow = DesktopSlowPolicy(s2, session) if s2 is not None else None
    return ComputerUseAgent(session, FastPolicy(engine), slow, ledger=kw.pop("ledger", None), kind="desktop", **kw)


# ---- 1. porte : probabilite de tete + marge, "done" verifie -------------------------------------------------------------
def test_near_tie_between_actions_escalates_instead_of_acting():
    desk = SimulatedDesktop()
    tie = {"action": [0.51, 0.49, 0.0, 0.0, 0.0, 0.0], f"t{ONE}": [0.95, 0.05]}   # 1 - entropie normalisee = 0,61 : passait
    eng, _ = s1([tie])
    out = agent(desk, eng, max_steps=3).run("tape 1")
    assert out["status"] == "needs_reasoning" and desk.log == []
    assert out["records"][0]["fast"]["escalate"] and "not decisive" in out["records"][0]["why"]


def test_done_is_verified_and_escalated_to_bonsai_when_the_goal_is_not_reached():
    desk = SimulatedDesktop()
    eng, _ = s1([act("done", achieved=0.1)])     # le clone dit "fini" mais juge l'objectif non atteint
    s2 = MockS2([tc("click", {"index": ONE}), tc("click", {"index": PLUS}, "c2"), tc("click", {"index": FIVE}, "c3"),
                 tc("click", {"index": EQUALS}, "c4"), tc("done", {"summary": "1 + 5 = 6"}, "c5")])
    events = []
    out = agent(desk, eng, s2, max_steps=3, on_event=events.append).run("calcule 1 + 5")
    assert out["status"] == "done" and desk.display == "6" and out["summary"] == "1 + 5 = 6"
    assert out["records"][0]["path"] == "escalated" and "done not confirmed" in out["records"][0]["why"]
    steps = [e for e in events if e["type"] == "computer.step"]
    assert steps[0]["path"] == "escalated" and steps[0]["tool"] == "desktop" and steps[0]["slow_actions"][:2] == ["click", "click"]
    assert sum(1 for e in events if e["type"] == "computer.action") == 4
    # sans Bonsai : on ne declare pas un faux succes
    desk2 = SimulatedDesktop()
    assert agent(desk2, s1([act("done", achieved=0.1)])[0], max_steps=3).run("calcule 1 + 5")["status"] == "needs_reasoning"
    # objectif juge atteint : "done" rapide accepte
    assert agent(SimulatedDesktop(), s1([act("done", achieved=0.9)])[0], max_steps=3).run("rien")["status"] == "done"


def test_done_is_verified_in_the_browser_too():
    pytest.importorskip("playwright")
    from jev_clone.computer_use import BrowserSession
    try:
        session = BrowserSession(headless=True)
    except Exception as e:
        pytest.skip(f"navigateur indisponible: {e}")
    try:
        session.set_content("<html><body><input placeholder='email'><button>Subscribe</button></body></html>")
        eng, _ = s1([act("done", achieved=0.1)])
        out = ComputerUseAgent(session, FastPolicy(eng), None, max_steps=2, ledger=None).run("Subscribe")
        assert out["status"] == "needs_reasoning" and "done not confirmed" in out["records"][0]["why"]
    finally:
        session.close()


# ---- 7. pre-classement des candidats et marge entre les deux meilleurs elements -------------------------------------------
def test_candidates_are_preranked_and_ambiguous_targets_escalate():
    els = [Element(i, "button", f"Option {i}") for i in range(40)]
    els[35] = Element(35, "button", "Subscribe to the newsletter")
    st = PageState("https://x", "t", "", els)
    eng, be = s1([act("click", 35)])
    d = FastPolicy(eng, max_candidates=24).decide("Subscribe to the newsletter", st, [], {})
    assert d.target == 35 and not d.escalate                     # au-dela du 24e element, mais pertinent
    assert sum(1 for c in be.calls[-1][1] if "Is element" in c) == 24
    two = {**act("click", 3), "t5": [0.95, 0.05]}                # deux elements aussi probables : on ne devine pas
    d2 = FastPolicy(s1([two])[0]).decide("Clique", PageState("u", "t", "", els[:10]), [], {})
    assert d2.escalate and "target uncertain" in d2.why


# ---- 2. garde-fou par pas ----------------------------------------------------------------------------------------------------
def guard_on(word):
    return lambda prefix: RISKY if word in prefix else SAFE


def test_risky_fast_step_escalates_and_requires_human_approval():
    desk = SimulatedDesktop(); desk.display = "42"
    eng, be = s1([act("click", CLEAR)], guard=guard_on("Effacer"))
    s2 = MockS2([tc("click", {"index": CLEAR}), tc("done", {"summary": "refuse par l'utilisateur"}, "c2")])
    asked = []
    out = agent(desk, eng, s2, max_steps=3, confirm=lambda d, j: asked.append((d, j)) or False).run("remets la calculatrice a zero")
    assert desk.display == "42" and "click:Effacer" not in desk.log
    assert out["records"][0]["path"] == "escalated" and "risky step" in out["records"][0]["why"]
    assert len(asked) == 1 and "Effacer" in asked[0][0]
    assert asked[0][1]["tool"] == "desktop_step" and asked[0][1]["tool_risk"] == "destructive" and asked[0][1]["preview"]["command"]
    blocked = json.loads(s2.calls[1][0][-1]["content"])
    assert blocked["blocked"] and blocked["reason"] == "the user declined this step"
    assert run_result(out)["blocked_steps"] == [{"type": "click", "target": CLEAR, "blocked": True}]
    # approuve : le pas est execute
    desk2 = SimulatedDesktop(); desk2.display = "42"
    s2b = MockS2([tc("click", {"index": CLEAR}), tc("done", {"summary": "ok"}, "c2")])
    agent(desk2, s1([act("click", CLEAR)], guard=guard_on("Effacer"))[0], s2b, max_steps=3, confirm=lambda d, j: True).run("remets a zero")
    assert desk2.display == "0"
    # personne pour confirmer : refuse et signale a Bonsai
    desk3 = SimulatedDesktop(); desk3.display = "42"
    s2c = MockS2([tc("click", {"index": CLEAR}), tc("done", {"summary": "impossible"}, "c2")])
    agent(desk3, s1([act("click", CLEAR)], guard=guard_on("Effacer"))[0], s2c, max_steps=3).run("remets a zero")
    assert desk3.display == "42" and "nobody can approve" in json.loads(s2c.calls[1][0][-1]["content"])["reason"]
    # sans Bonsai : le pas risque rapide demande directement l'autorisation
    desk4 = SimulatedDesktop(); desk4.display = "42"
    out4 = agent(desk4, s1([act("click", CLEAR)], guard=guard_on("Effacer"))[0], max_steps=3, confirm=lambda d, j: False).run("remets a zero")
    assert out4["status"] == "blocked" and desk4.display == "42"


def test_bonsai_shortcuts_and_app_launches_are_judged_before_execution():
    desk = SimulatedDesktop()
    eng, be = s1([act("escalate")], guard=lambda p: RISKY if "alt+f4" in p or "format" in p else SAFE)
    s2 = MockS2([tc("open_app", {"name": "format C:"}), tc("press_keys", {"keys": "alt+f4"}, "c2"),
                 tc("open_app", {"name": "notepad"}, "c3"), tc("done", {"summary": "ok"}, "c4")])
    agent(desk, eng, s2, max_steps=2).run("ouvre le bloc-notes")
    assert desk.log == ["open:notepad"] and desk.front == "notepad"
    results = [json.loads(m["content"]) for m in s2.calls[-1][0] if m["role"] == "tool"]
    assert results[0]["blocked"] and results[1]["blocked"] and results[2]["ok"]
    assert be.kinds.count("guard") == 3
    assert any("open the application or file format C:" in p for p in be.prefixes)


def test_windows_open_app_never_goes_through_a_shell(monkeypatch):
    b = object.__new__(WindowsUIABackend)   # le constructeur exige Windows ; open_app n'utilise que os / subprocess
    opened, popen = [], []
    monkeypatch.setattr(os, "startfile", lambda n: opened.append(n), raising=False)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: popen.append((a, kw)))
    monkeypatch.setattr(time, "sleep", lambda s: None)
    evil = 'calc" & del /q C:\\x & "'
    b.open_app(evil)
    assert opened == [evil] and popen == []
    def missing(n):
        raise FileNotFoundError(n)
    monkeypatch.setattr(os, "startfile", missing, raising=False)
    b.open_app("notepad")
    assert popen == [((["notepad"],), {})]      # argv, jamais shell=True
    with pytest.raises(ValueError):
        b.open_app("calc\nwhoami")


# ---- 3. annulation, progression, plafond d'escalades, clone en panne --------------------------------------------------------
def test_stop_cancels_the_run_and_the_inner_bonsai_loop():
    eng, _ = s1([act("escalate")])
    s2 = MockS2([tc("scroll_down", {})])        # Bonsai ne finit jamais
    run = make_desktop_factory(eng, s2, SimulatedDesktop, should_stop=lambda: len(s2.calls) >= 2)()
    r = run("tache sans fin")
    assert r["status"] == "cancelled" and not r["ok"] and len(s2.calls) == 2   # et non 24 pas x 6 tours


def test_escalations_are_capped_and_a_failing_bonsai_stops_the_run():
    s2 = MockS2([{"content": "je ne sais pas"}])  # aucune action, jamais done
    out = agent(SimulatedDesktop(), s1([act("escalate")])[0], s2, max_steps=20, max_escalations=3).run("x")
    assert out["status"] == "max_escalations" and out["escalations"] == 3 and len(s2.calls) == 3

    class DeadS2:
        calls = 0

        def chat(self, *a, **kw):
            DeadS2.calls += 1
            raise ConnectionError("Bonsai ne repond pas")
    out = agent(SimulatedDesktop(), s1([act("escalate")])[0], DeadS2(), max_steps=20).run("x")
    assert out["status"] == "s2_error" and DeadS2.calls == 2


def test_clone_failure_escalates_to_bonsai_instead_of_failing_the_tool():
    class DeadS1:
        def answer(self, req):
            raise ConnectionError("S1 down")
    desk = SimulatedDesktop()
    s2 = MockS2([tc("click", {"index": ONE}), tc("done", {"summary": "1"}, "c2")])
    asked = []
    out = agent(desk, DeadS1(), s2, max_steps=3, confirm=lambda d, j: asked.append(j) or True).run("tape 1")
    assert out["status"] == "done" and desk.display == "1"
    assert out["records"][0]["path"] == "escalated" and "fast policy unavailable" in out["records"][0]["why"]
    assert len(asked) == 1 and asked[0]["s1_error"]   # classifieur muet : prudence, le pas est confirme par l'humain
    # verification en panne apres une action rapide : Bonsai reprend aussi
    class HalfS1:
        def __init__(self):
            self.inner = s1([act("click", ONE)])[0]
        def answer(self, req):
            if "before" in (req.get("state") or {}):
                raise ConnectionError("S1 down")
            return self.inner.answer(req)
    desk2 = SimulatedDesktop()
    out2 = agent(desk2, HalfS1(), MockS2([tc("done", {"summary": "1"})]), max_steps=3).run("tape 1")
    assert out2["status"] == "done" and out2["records"][0]["path"] == "escalated_after_verify" and out2["records"][0]["verify"] is None


def test_desktop_factory_streams_one_event_per_step_and_logs_trajectories(tmp_path):
    eng, _ = s1([act("click", ONE), act("click", PLUS), act("click", FIVE), act("click", EQUALS), act("done", achieved=0.9)])
    events = []
    run = make_desktop_factory(eng, MockS2([{"content": "inutile"}]), SimulatedDesktop, on_event=events.append,
                               ledger=tmp_path / "desktop_trajectories.jsonl")()
    r = run("calcule 1 + 5")
    assert r["ok"] and r["fast_steps"] == 5 and r["escalations"] == 0 and "'6'" in r["screen"]
    steps = [e for e in events if e["type"] == "computer.step"]
    assert len(steps) == 5 and all(e["tool"] == "desktop" and e["path"] == "fast" and e["p"] >= 0.8 for e in steps)
    line = json.loads((tmp_path / "desktop_trajectories.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert line["kind"] == "desktop" and line["status"] == "done" and len(line["records"]) == 5


def test_sessions_wire_confirm_events_stop_and_ledgers_into_both_factories(tmp_path, monkeypatch):
    import prophet_studio.sessions as ss
    from prophet_studio.config import Settings
    got, published = {}, []

    def fake_browser(s1_, s2_, **kw):
        got["browser"] = kw
        return lambda: None

    def fake_desktop(s1_, s2_, backend=None, **kw):
        got["desktop"] = kw
        return lambda: None

    class FakeProphet:
        def __init__(self, *a, **kw):
            got["prophet"] = kw

        def handle(self, text, history, effort="auto"):
            got["desktop"]["on_event"]({"type": "computer.step", "step": 0})
            return types.SimpleNamespace(response="ok")
    monkeypatch.setattr(ss, "make_browser_factory", fake_browser)
    monkeypatch.setattr(ss, "make_desktop_factory", fake_desktop)
    monkeypatch.setattr(ss, "Prophet", FakeProphet)
    for mode in ("smart", "auto"):
        st = Settings(browser_tool=True, desktop_tool=True, workspace=str(tmp_path / "ws"), permission_mode=mode)
        svc = ss.AgentService(ss.SessionStore(tmp_path / "sessions"), published.append, urls=lambda: ("http://127.0.0.1:9", "http://127.0.0.1:9"),
                              settings=lambda st=st: st, ctx=lambda: 8192, desktop_backend=SimulatedDesktop)
        sid = svc.store.create(str(tmp_path / "ws"))["id"]
        svc.submit(sid, "calcule")
        assert svc.wait_idle(sid, 10)
        assert got["browser"]["ledger"] == tmp_path / "runs" / "trajectories.jsonl"
        assert got["desktop"]["ledger"] == tmp_path / "runs" / "desktop_trajectories.jsonl"
        for k in ("browser", "desktop"):
            assert callable(got[k]["confirm"]) and callable(got[k]["on_event"]) and got[k]["should_stop"]() is False
        if mode == "smart":
            assert got["desktop"]["confirm"] is got["prophet"]["confirm"]     # meme mecanisme d'autorisation que les outils
        else:
            assert got["desktop"]["confirm"]("x", {}) is True                  # mode « jamais » : aucune question
    assert any(e["type"] == "computer.step" and e.get("session_id") for e in published)
    # bureau reel indisponible (hors Windows, sans bureau simule) : l'outil n'est pas propose, meme si le reglage est actif
    import jev_clone.desktop_use as du
    monkeypatch.setattr(du, "desktop_available", lambda: (False, "Windows requis"))
    got.clear(); got["desktop"] = {"on_event": lambda e: None}
    svc = ss.AgentService(ss.SessionStore(tmp_path / "sessions"), published.append, urls=lambda: ("http://127.0.0.1:9", "http://127.0.0.1:9"),
                          settings=lambda: Settings(desktop_tool=True, workspace=str(tmp_path / "ws")), ctx=lambda: 8192)
    sid = svc.store.create(str(tmp_path / "ws"))["id"]
    svc.submit(sid, "calcule")
    assert svc.wait_idle(sid, 10)
    assert got["prophet"]["desktop_factory"] is None and got["prophet"]["browser_factory"] is None


# ---- 4. DAgger ------------------------------------------------------------------------------------------------------------------
BEFORE = {"goal": "g", "available_slots": {"email": "bob@example.com"}, "last_actions": [], "url": "u1", "title": "A", "page": "",
          "elements": ['[0] a "Next"', '[1] button "Continue"']}
AFTER = {"goal": "g", "available_slots": {"email": "bob@example.com"}, "last_actions": [{"type": "click", "target": 0, "ok": True}],
         "url": "u2", "title": "B", "page": "", "elements": ['[0] link "Help"', '[1] input "email address"', '[2] button "Subscribe"']}
TRAJ = {"goal": "g", "status": "done", "records": [
    {"step": 0, "path": "escalated_after_verify", "state": BEFORE, "slow_state": AFTER, "result": {"ok": True}, "verify": 0.1,
     "fast": {"action": "click", "target": 0},
     "slow": {"actions": [{"type": "click", "target": 2, "blocked": True}, {"type": "type", "target": 1, "text": "bob@example.com"}]}},
    {"step": 1, "path": "escalated", "state": AFTER, "slow": {"actions": [{"type": "type", "target": 1, "text": "texte libre"}]}},
    {"step": 2, "path": "escalated", "state": AFTER, "slow": {"actions": [{"type": "done", "summary": "ok"}]}},
]}


def test_trajectory_examples_use_the_state_bonsai_saw_and_label_slots():
    ex = trajectory_to_examples(TRAJ)
    assert len(ex) == 3 and all(e["source"] == "bonsai" for e in ex)
    e0 = ex[0]
    assert e0["state"] is AFTER and set(e0["questions"]) == {"action", "achieved", "t0", "t1", "t2", "slot"}
    assert e0["labels"] == {"action": "type", "achieved": False, "t0": False, "t1": True, "t2": False, "slot": "email"}
    assert ex[1]["labels"]["action"] == "escalate" and ex[1]["labels"]["slot"] == "none"   # texte libre : a escalader
    assert ex[2]["labels"]["action"] == "done" and ex[2]["labels"]["achieved"] is True
    for e in ex:
        SystemOneRequest(state=e["state"], questions=e["questions"])


def test_escalation_after_verify_records_the_state_bonsai_saw():
    desk = SimulatedDesktop()
    eng, _ = s1([act("click", ONE)], verify={"ok": [0.1, 0.9]})
    out = agent(desk, eng, MockS2([tc("done", {"summary": "11"})]), max_steps=4).run("tape 11")
    rec = out["records"][1]
    assert rec["path"] == "escalated_after_verify" and desk.display == "11"
    assert "'11'" in rec["slow_state"]["page"] and "'11'" not in rec["state"]["page"]
    assert trajectory_to_examples(out)[-1]["state"] == rec["slow_state"]


def test_make_from_trajectories_writes_the_training_format(tmp_path):
    from training.make_from_trajectories import main
    p = tmp_path / "trajectories.jsonl"
    self_traj = {"records": [{"path": "fast", "state": BEFORE, "verify": 0.95, "result": {"ok": True},
                              "fast": {"action": "click", "target": 1, "slot": None}}]}
    p.write_text("\n".join([json.dumps(TRAJ), "pas du json", "", json.dumps(self_traj)]), encoding="utf-8")
    out, val = tmp_path / "train.jsonl", tmp_path / "val.jsonl"
    stats = main(["--in", str(p), "--out", str(out), "--val", str(val), "--val-frac", "0.25"])
    rows = [json.loads(l) for f in (out, val) for l in f.read_text(encoding="utf-8").splitlines()]
    assert stats["bad_lines"] == 1 and stats["bonsai"] == 3 and stats["self"] == 1 and len(rows) == 4
    fmt = PromptFormat()
    for r in rows:   # meme lecture que load_examples/gold_index de training/train_lora_rlcd.py
        req = SystemOneRequest(state=r["state"], questions=r["questions"])
        for qid, q in req.questions.items():
            for br in build_branches(qid, q, fmt, 2, random.Random(0)):
                lab = r["labels"][qid]
                assert (bool(lab) if q.type == "noul" else lab) in br.keys
    assert main(["--in", str(p), "--out", str(out), "--no-self"])["self"] == 0


# ---- 5. disponibilite du navigateur ------------------------------------------------------------------------------------------
def test_browser_availability_is_detected_without_launching_a_browser(tmp_path, monkeypatch):
    import importlib.util

    import jev_clone.computer_use as cu
    monkeypatch.setattr(cu, "DEFAULT_CHROMIUM", [])
    monkeypatch.delenv("JEV_CHROMIUM", raising=False)
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path / "pw"))
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: object() if name == "playwright" else real(name, *a))
    ok, why = browser_available()
    assert not ok and "playwright install chromium" in why
    (tmp_path / "pw" / "chromium-1234").mkdir(parents=True)
    (tmp_path / "pw" / "chromium-1234" / "INSTALLATION_COMPLETE").touch()
    assert browser_available() == (True, "")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "playwright" else real(name, *a))
    ok, why = browser_available()
    assert not ok and "uv pip install playwright && playwright install chromium" in why


def test_browse_tool_and_state_report_a_missing_browser(tmp_path, monkeypatch):
    import jev_clone.computer_use as cu
    from prophet_studio.config import Paths
    from prophet_studio.server import Studio
    monkeypatch.setattr(cu, "browser_available", lambda: (False, "Playwright n'est pas installe"))
    assert make_browser_factory(None, None)()("cherche x") == {"ok": False, "error": "browser unavailable: Playwright n'est pas installe"}
    st = Studio(Paths(tmp_path / "home"), "tok", 7878, demo=True)
    s = st.state()
    assert s["browser"] == {"available": False, "reason": "Playwright n'est pas installe"} and s["desktop"]["available"]


# ---- 6. voix : une autorisation ne se donne que par la phrase exacte ---------------------------------------------------------
def test_voice_permission_answers_need_the_exact_phrase():
    crit = list(COMMANDS) + ["prompt"]

    def s1_says(cmd):
        table = [0.02] * len(crit)
        table[crit.index(cmd)] = 0.9
        return SystemOneEngine(ScriptedBackend([{"intent": table}], declared={"intent": crit}), model_name="mock")
    r = route_utterance("ouais c'est bon", s1_says("approve"))
    assert (r.kind, r.command, r.source) == ("command", "confirm_approve", "systemone")
    assert route_utterance("laisse tomber ça", s1_says("deny")).command == "confirm_deny"
    assert route_utterance("accepte", s1_says("deny")).to_dict()["command"] == "approve"          # grammaire exacte
    assert route_utterance("tu peux me relire ça", s1_says("read_last")).command == "read_last"   # les autres commandes restent

    class Broken:
        def answer(self, req):
            raise RuntimeError("S1 down")
    r = route_utterance("tu peux me relire ça", Broken())
    assert (r.kind, r.source, r.confidence) == ("prompt", "systemone", 0.0) and "S1 down" in r.error

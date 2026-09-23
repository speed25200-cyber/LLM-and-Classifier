"""Revue adverse, 3e vague : decisions de S1 (calibration a chaque point d'entree, porte de la voie directe, affirmations
d'action de la voie directe, seuils perimes, ancien format sous jev serve).

Un fichier de calibration n'applique jamais une temperature a une question sur laquelle il n'a pas ete ajuste (Studio,
jev prophet, exemples, jev serve, jev decide) ; la porte calibree de la voie directe tient la precision visee sur les seules
decisions 'direct' ; des seuils d'une version precedente ne sont jamais appliques en silence ; CLAIMS_ACTION sans faux
positifs (abreviations, domaines, exemples de code, verification de tete) et avec les affirmations courantes."""

import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from jev_clone import cli
from jev_clone.calibrate import calibrate, load_examples
from jev_clone.engine import SystemOneEngine
from jev_clone.fusion import FusionRouter, GatePolicy
from jev_clone.guard import JUDGE_QUESTIONS
from jev_clone.presets import META, TICKET_TRIAGE
from jev_clone.prophet import CLAIMS_ACTION, INTENTS, LANGUAGES, NEEDS_TOOLS_RX, PROPHET_TURN, Prophet, Workspace
from jev_clone.readout import CAL_VERSION, Calibration, probs_to_logits, question_fingerprint, softmax
from jev_clone.schema import SystemOneRequest
from prophet_studio import calibration as s1cal
from prophet_studio.sessions import AgentService, SessionStore
from tests.conftest import MockBackend, MockS2, ScriptedBackend

ROOT = Path(__file__).resolve().parent.parent
I, L = list(INTENTS), list(LANGUAGES)
TR = ["readonly", "workspace_write", "destructive", "privileged", "exfiltration"]
DONE = {"content": "", "tool_calls": [MockS2.tool_call("done", {"summary": "fini"}, "cd")]}
GUARD = {"user_request": "nettoie le dossier", "proposed_action": "run_command: rm -rf ./data"}
RM_RF = dict(tool_risk=(0.55, 0.05, 0.14, 0.13, 0.13), risk=(0.4, 0.3, 0.2, 0.1))   # lecture brute : 0.40 risque -> confirmation
TURN_STATE = {"request": "x", "workspace_files": [], "recent_turns": []}
UNDER = {"noul": 0.45, "choice": 0.45, "score": 0.45}                                # fine-tune peu assure : T < 1


def table(direct=0.1, ok=0.9, nr=0.2, risk=(1, 0, 0, 0), tool_risk=(0.9, 0.04, 0.02, 0.02, 0.02)):
    return {"direct": [direct, 1 - direct], "clarify": [0.1, 0.9], "needs_reasoning": [nr, 1 - nr], "risk": list(risk),
            "intent": [1 / len(I)] * len(I), "language": [1 / len(L)] * len(L), "tool_risk": list(tool_risk),
            "policy_violation": [0.1, 0.9], "ok": [ok, 1 - ok]}


def backend(**kw) -> MockBackend:
    be = MockBackend(table(**kw), default_noul=(0.2, 0.8))
    be.declared = {"intent": I, "language": L, "tool_risk": TR}
    return be


def engine(cal=None, **kw) -> SystemOneEngine:
    return SystemOneEngine(backend(**kw), calibration=cal, model_name="mock")


def judge(cal, tmp_path) -> dict:
    return Prophet(engine(cal=cal, **RM_RF), None, Workspace(tmp_path / "ws"))._judge_one(GUARD["user_request"], GUARD["proposed_action"])


def service(tmp_path, runs, mid) -> AgentService:
    return AgentService(SessionStore(tmp_path / "s"), lambda e: None, urls=lambda: ("http://127.0.0.1:1", "http://127.0.0.1:2"),
                        settings=lambda: None, ctx=lambda: 8192, calibration=lambda: s1cal.calibration_file(runs, mid))


# ---- 1. Studio : un fichier sans question reconnue n'est jamais une calibration generique ----------------------------------------
@pytest.mark.parametrize("thresholds", [{"action": 0.8, "achieved": 0.9}, {"tool_risk": 0.9}])
def test_studio_refuses_old_files_without_a_pre_turn_question(tmp_path, thresholds):
    runs, mid = tmp_path / "runs", "custom-s1-clone"
    s1cal.import_calibration(runs, mid, data={"temperature": {"noul": 0.5, "choice": 0.45, "score": 0.5}, "thresholds": thresholds})
    st = s1cal.status(runs)[mid]
    assert "recalibrez" in st.get("error", "") and st["questions"] == {}          # avant : aucune erreur, "calibre"
    cal, why, _ = s1cal.load_for_engine(s1cal.calibration_file(runs, mid))
    assert why and not cal.is_active()
    s1 = service(tmp_path, runs, mid).engines()[0]
    assert s1.cal.t_for("tool_risk", JUDGE_QUESTIONS["tool_risk"], GUARD) == 1.0    # avant : 0.45 sur le garde-fou
    j = judge(s1.cal, tmp_path)
    assert j["needs_confirmation"] and j["p_risky"] == pytest.approx(0.4, abs=1e-3)  # avant : 0.114, execute sans demander


def test_studio_ignores_entries_without_a_known_state_shape(tmp_path):
    teams = list(TICKET_TRIAGE["team"]["criteria"])
    exs, rows = [], []
    for i in range(60):                              # CLI sur des etats texte (jev decide, tickets) : `risk` sans forme d'etat
        lvl, ok = i % 4, i % 10 < 7
        exs.append({"state": f"Ticket #{i}: message du client", "questions": TICKET_TRIAGE,
                    "labels": {"team": teams[i % 5], "risk": lvl, "needs_reasoning": i % 3 == 0}})
        rows.append({"team": [0.96 if k == teams[(i if ok else i + 1) % 5] else 0.01 for k in teams],
                     "risk": [0.97 if k == (lvl if ok else (lvl + 2) % 4) else 0.01 for k in range(4)],
                     "needs_reasoning": [0.97, 0.03] if (i % 3 == 0) == ok else [0.03, 0.97]})
    cal = calibrate(SystemOneEngine(ScriptedBackend(rows, declared={"team": teams}), model_name="m"), exs, source="cli")[0]
    assert cal.questions["risk"]["state"] is None and cal.t_for("risk", JUDGE_QUESTIONS["risk"], GUARD) != 1.0
    assert s1cal.for_studio(cal).t_for("risk", JUDGE_QUESTIONS["risk"], GUARD) == 1.0      # avant : T = 2.36 sur le garde-fou
    assert not s1cal.usable(cal)[0]
    # formes d'etat disjointes (intersection vide) : meme chose
    empty = Calibration(questions={"risk": {"T": 0.25, "kind": "score", "fp": question_fingerprint(META["risk"]), "state": [], "n": 50}},
                        meta={"version": CAL_VERSION})
    assert s1cal.for_studio(empty).t_for("risk", JUDGE_QUESTIONS["risk"], GUARD) == 1.0    # avant : 0.25


# ---- 2. jev prophet, exemples, jev serve, jev decide : meme regle ----------------------------------------------------------------
def test_cli_prophet_and_browser_example_never_apply_a_foreign_temperature(tmp_path, monkeypatch, capsys):
    generic, legacy = tmp_path / "generic.json", tmp_path / "legacy.json"
    Calibration(temperature=dict(UNDER)).save(generic)                          # training/train_lora_rlcd.py, eval/mmlu_ece.py
    Calibration.from_dict({"temperature": dict(UNDER), "thresholds": {"direct": 0.7, "tool_risk": 0.9}}).save(legacy)
    got: dict = {}
    monkeypatch.setattr("jev_clone.backend_llamacpp.LlamaCppBackend", lambda *a, **k: backend(**RM_RF))
    monkeypatch.setattr("jev_clone.prophet.repl", lambda p: got.__setitem__("p", p))
    for f in (generic, legacy):
        cli.main(["prophet", "--browser", "--workspace", str(tmp_path / "ws"), "--calibration", str(f)])
        p = got["p"]
        assert p.s1.cal.t_for("tool_risk", JUDGE_QUESTIONS["tool_risk"], GUARD) == 1.0   # avant : 0.45 (garde-fou et navigateur)
        assert p._judge_one(GUARD["user_request"], GUARD["proposed_action"])["needs_confirmation"]
    assert p.s1.cal.t_for("direct", PROPHET_TURN["direct"], TURN_STATE) == 0.45 and p.s1.cal.thresholds == {}   # pre-tour : T, seuil perime ignore
    err = capsys.readouterr().err
    assert "generique" in err and "version precedente" in err
    # exemple navigateur : le clone decide chaque pas et garde les pas risques
    import jev_clone.computer_use as cu

    class Agent:
        def __init__(self, session, fast, slow, max_steps=20):
            got["fast"] = fast

        def run(self, goal, url=None, slots=None):
            return {"status": "done", "steps": 0, "history": [], "records": []}
    monkeypatch.setattr(cu, "BrowserSession", lambda **k: type("S", (), {"close": lambda self: None})())
    monkeypatch.setattr(cu, "ComputerUseAgent", Agent)
    monkeypatch.setattr(cu, "FastPolicy", lambda s1: s1)
    monkeypatch.setenv("JEV_CALIBRATION", str(generic))
    monkeypatch.delenv("JEV_S2_URL", raising=False)
    monkeypatch.setattr(sys, "argv", ["browser_agent.py", "--url", "https://example.com", "--goal", "ouvre le lien"])
    runpy.run_path(str(ROOT / "examples" / "browser_agent.py"), run_name="__main__")
    assert got["fast"].cal.t_for("tool_risk", JUDGE_QUESTIONS["tool_risk"], GUARD) == 1.0   # avant : 0.45


def test_jev_serve_and_decide_ignore_a_temperature_only_file(tmp_path, monkeypatch, capsys):
    from jev_clone.server import build_app
    f = tmp_path / "generic.json"
    Calibration(temperature=dict(UNDER)).save(f)
    be = MockBackend({"team": [0.7, 0.3]})
    monkeypatch.setattr("jev_clone.server.LlamaCppBackend", lambda *a, **k: be)
    monkeypatch.setenv("JEV_CALIBRATION", str(f))
    q = {"team": {"type": "choice", "instructions": "Which team?", "criteria": ["a", "b"]}}
    r = TestClient(build_app()).post("/v1/systemone", json={"state": {"ticket": "x"}, "questions": q}).json()
    assert r["answers"]["team"]["probabilities"] == {"a": 0.7, "b": 0.3}         # avant : T = 0.45 -> 0.87 / 0.13
    monkeypatch.setattr("jev_clone.backend_llamacpp.LlamaCppBackend", lambda *a, **k: be)
    capsys.readouterr()
    cli.main(["decide", "--state", "x", "--choice", "team=a,b", "--calibration", str(f)])
    out = capsys.readouterr()
    assert json.loads(out.out)["answers"]["team"]["probabilities"] == {"a": 0.7, "b": 0.3} and "generique" in out.err


# ---- 3. porte de la voie directe : precision des seules decisions 'direct' ---------------------------------------------------------
def seed_rows(direct_of):
    """Lecture de chaque graine : 'direct' selon direct_of, reflexion et risque justes et surs."""
    return [table(direct=direct_of(i, ex["labels"]), nr=0.9 if ex["labels"]["needs_reasoning"] else 0.1,
                  risk=(0.9, 0.05, 0.03, 0.02) if ex["labels"]["risk"] == 0 else (0.05, 0.05, 0.05, 0.85))
            for i, ex in enumerate(load_examples())]


def seeds_calibration(rows) -> Calibration:
    return calibrate(SystemOneEngine(ScriptedBackend(rows, declared={"intent": I, "language": L}), model_name="m"), load_examples(), source="t")[0]


def test_direct_gate_meets_the_precision_target_on_direct_decisions(tmp_path):
    exs = load_examples()
    wrong = set([i for i, ex in enumerate(exs) if not ex["labels"]["direct"]][::18])   # 4 demandes d'outil lues 'direct' a 0.80
    assert len(wrong) == 4
    rows = seed_rows(lambda i, lab: 0.93 if lab["direct"] else 0.80 if i in wrong else 0.03)
    cal = seeds_calibration(rows)
    gate = cal.gate("direct", PROPHET_TURN["direct"], TURN_STATE, option=0)
    T = cal.questions["direct"]["T"]
    p_yes = [round(float(softmax(probs_to_logits(np.array(r["direct"])), T)[0]), 6) for r in rows]
    took = [bool(ex["labels"]["direct"]) for ex, p in zip(exs, p_yes) if p >= max(0.5, gate)]
    assert len(took) == 10 and all(took)                  # avant : seuil 0.8417 sur les deux reponses, 14 admises dont 10 justes
    assert cal.thresholds["direct"] < gate                  # FusionRouter garde sa porte sur la reponse retenue
    ev: list = []
    be = MockBackend(rows[min(wrong)], default_noul=(0.2, 0.8))
    be.declared = {"intent": I, "language": L}
    turn = Prophet(SystemOneEngine(be, calibration=cal, model_name="m"), MockS2([DONE]), Workspace(tmp_path / "ws"),
                   on_event=ev.append).handle(exs[min(wrong)]["state"]["request"])
    d = next(e for e in ev if e["type"] == "s1.decision")
    assert turn.path == "agent" and d["path"] == "agent" and d["gates"]["direct"] == gate      # avant : voie directe
    assert d["gates"]["needs_reasoning"] == cal.questions["needs_reasoning"]["gates"]["1"]     # decision 'sans reflexion'
    assert d["gates"]["risk"] == cal.questions["risk"]["gates"]["0"]                           # decision 'risque 0'
    cal.save(tmp_path / "c.json")
    back = Calibration.load(tmp_path / "c.json")
    assert back.questions["direct"]["gates"] == cal.questions["direct"]["gates"] and back.version == CAL_VERSION and not back.stale_thresholds
    with pytest.raises(ValueError):
        Calibration.from_dict({**back.to_dict(), "questions": {"direct": {**back.questions["direct"], "gates": {"0": 2.0}}}})
    # 'non' surs mais faux sur la moitie des graines directes, tous les 'oui' justes : la voie directe reste ouverte
    direct = [i for i, ex in enumerate(exs) if ex["labels"]["direct"]][::2]
    cal2 = seeds_calibration(seed_rows(lambda i, lab: (0.02 if i in direct else 0.93) if lab["direct"] else 0.03))
    g2 = cal2.gate("direct", PROPHET_TURN["direct"], TURN_STATE, option=0)
    assert cal2.thresholds["direct"] is None and g2 is not None and g2 < 0.93                 # avant : voie directe coupee


# ---- 4. CLAIMS_ACTION : table FR / EN de reponses directes realistes -------------------------------------------------------------
REPLIES = [   # (reponse de la voie directe, pretend avoir agi sur la machine ?)
    # abreviations, domaines, appels de bibliotheque : pas des fichiers
    ("I have edited the draft, i.e. shortened it.", False),
    ("I've updated the list, e.g. added eggs and milk.", False),
    ("J'ai modifié la phrase, p.ex. « il est parti » au lieu de « il part ».", False),
    ("I've added the U.S. holidays to your list.", False),
    ("I've added a link to wikipedia.org for reference.", False),
    ("J'ai ajouté la source (lemonde.fr).", False),
    ("I've updated the code to use os.path.join so it works on Windows too.", False),
    ("I've added a try/except around json.loads so bad input doesn't crash.", False),
    ("I've updated the formula so it uses r.json() correctly.", False),
    ("I've added Node.js, Vue.js and Angular.js to the comparison table.", False),
    # fichiers montres en exemple (bloc de code, modele, ci-dessous)
    ("I've written a README.md template for you:\n```md\n# Project\n```", False),
    ("J'ai créé un exemple de config.yaml ci-dessous :\n```yaml\nport: 8080\n```", False),
    ("I've updated the example to use index.js instead of main.js.", False),
    ("I have written the function below; copy it into utils.py:\n```python\ndef add(a, b):\n    return a + b\n```", False),
    ("J'ai écrit un exemple de fichier .env :\n```\nPORT=3000\n```", False),
    # verifications de tete, tournures sans objet reel
    ("I tested it on the example you gave: f(2) = 5.", False),
    ("I ran it through a quick mental check: 17 is prime.", False),
    ("I tested this idea mentally: it works for n = 3.", False),
    ("Je l'ai testé de tête : 7 x 8 = 56.", False),
    ("I launched into the explanation too fast; here it is step by step.", False),
    ("I started this answer with the basics.", False),
    ("I made a mistake above: the answer is 12, not 10.", False),
    ("J'ai corrigé les deux fautes d'accord dans ta phrase.", False),
    ("Voici un poème que j'ai écrit pour toi :\nLa lune veille sur la ville.", False),
    ("J'ai lancé les dés : tu obtiens un 4.", False),
    ("Pour créer le fichier, tape `touch notes.txt` dans le terminal.", False),
    ("I've pushed back the deadline in your plan to Friday.", False),
    # affirmations d'action (rien n'a ete fait sur la voie directe)
    ("I've made the changes to main.py.", True),
    ("Voilà, le fichier notes.txt a été créé avec ta liste.", True),
    ("The file todo.md has been updated with your tasks.", True),
    ("I've put the code in hello.py.", True),
    ("J'ai mis le code dans hello.py.", True),
    ("I've started the dev server, open http://localhost:3000.", True),
    ("J'ai lancé le petit serveur de dev.", True),
    ("Les tests passent, je les ai exécutés.", True),
    ("Je l'ai exécuté, il affiche bien 42.", True),
    ("I've pushed the changes to GitHub.", True),
    ("I've committed the changes.", True),
    ("I've deployed the app.", True),
    ("J'ai déployé l'application sur ton serveur.", True),
    ("I've fixed the bug in app.py.", True),
    ("J'ai corrigé le bug dans app.py.", True),
    ("I tested it locally and everything passes.", True),
    ("I ran it and it prints 42.", True),
    ("I've installed numpy and pandas for you.", True),
    ("J'ai créé le fichier index.html.", True),
    ("C'est fait : j'ai enregistré ta liste dans courses.txt.", True),
    ("I've cloned the repo into the workspace folder.", True),
    ("I don’t have access to your file system.", True),
]


def test_claims_table_is_large_and_bilingual():
    assert len(REPLIES) >= 40 and sum(c for _, c in REPLIES) >= 15 and sum(not c for _, c in REPLIES) >= 15


@pytest.mark.parametrize("reply,claims", REPLIES)
def test_claims_action_on_realistic_direct_replies(reply, claims):
    assert bool(CLAIMS_ACTION.search(reply)) is claims, CLAIMS_ACTION.search(reply)


def test_needs_tools_marker_is_not_a_code_identifier(tmp_path):
    for text in ("`NEEDS_TOOLS`", "**NEEDS_TOOLS**", "NEEDS\\_TOOLS", "Il faut lire vos fichiers. NEEDS_TOOLS", "needs_tools", " `needs_tools`. "):
        assert NEEDS_TOOLS_RX.search(text), text
    for text in ("The flag `needs_tools` in your config controls this.", "def needs_tools(x):\n    return x > 1"):
        assert not NEEDS_TOOLS_RX.search(text), text                          # avant : reponse sur du code ecartee
    ws = Workspace(tmp_path / "ws")
    for answer, rerouted in (("I've written a README.md template for you:\n```md\n# Project\n```", False),
                             ("Set `needs_tools = True` in the config to enable it.", False), ("I've pushed the changes to GitHub.", True)):
        s2 = MockS2([{"content": answer}, DONE])
        t = Prophet(engine(direct=0.95, ok=0.9), s2, ws).handle("aide-moi")
        assert (t.path == "agent") is rerouted and len(s2.calls) == (2 if rerouted else 1), answer


# ---- 5. seuils d'une version precedente : signales, jamais appliques en silence ---------------------------------------------------
OLD_STUDIO = {"temperature": {"noul": 2.5079, "choice": 2.9517, "score": 1.6262},   # ce que "Calibrer" ecrivait (seuils sans ex aequo)
              "thresholds": {"intent": 0.5365, "language": 0.301, "clarify": 0.7258, "needs_reasoning": 0.6347, "risk": 0.6473, "direct": 0.7371},
              "meta": {"source": "studio", "n": 82, "target_precision": 0.95, "statistic": "top1"}}


def test_stale_thresholds_are_flagged_and_never_enforced(tmp_path, monkeypatch, capsys):
    runs, mid = tmp_path / "runs", "custom-s1-myjevclone-q4"
    fresh = seeds_calibration(seed_rows(lambda i, lab: 0.93 if lab["direct"] else 0.03))
    no_version = Calibration.from_dict({**fresh.to_dict(), "meta": {k: v for k, v in fresh.meta.items() if k != "version"}})
    for data in (OLD_STUDIO, no_version.to_dict()):                              # ancien format ; format par question d'avant la version
        s1cal.store(runs, mid, Calibration.from_dict(data))
        st = s1cal.status(runs)[mid]
        assert "error" not in st and st["stale_thresholds"] and "recalibrez" in st["warning"]   # avant : rien ne le disait
        cal = service(tmp_path, runs, mid).engines()[0].cal
        assert cal.thresholds == {} and cal.is_active()                          # temperatures ajustees, seuils ignores
        ev: list = []
        Prophet(SystemOneEngine(backend(direct=0.93), calibration=cal, model_name="m"), MockS2([DONE]), Workspace(tmp_path / "ws"),
                on_event=ev.append).handle("Salut")
        d = next(e for e in ev if e["type"] == "s1.decision")
        assert d["gates"] == {} and d["calibrated"] is True                      # avant : {direct 0.7371, ...} appliques
    s1cal.store(runs, mid, fresh)                                               # recalibre : seuils par decision appliques
    assert "stale_thresholds" not in s1cal.status(runs)[mid]
    assert service(tmp_path, runs, mid).engines()[0].cal.gate("direct", PROPHET_TURN["direct"], TURN_STATE, option=0) is not None
    old = tmp_path / "old.json"
    old.write_text(json.dumps(OLD_STUDIO))
    got: dict = {}
    monkeypatch.setattr("jev_clone.backend_llamacpp.LlamaCppBackend", lambda *a, **k: backend())
    monkeypatch.setattr("jev_clone.prophet.repl", lambda p: got.__setitem__("p", p))
    cli.main(["prophet", "--workspace", str(tmp_path / "ws"), "--calibration", str(old)])
    assert got["p"]._gates(TURN_STATE) == {} and "version precedente" in capsys.readouterr().err


# ---- 6. jev serve : ancien format d'un autre jeu de donnees, seuils et temperature coherents ------------------------------------
def test_jev_serve_old_format_ticket_file_scales_what_it_gates(tmp_path, monkeypatch, capsys):
    legacy = Calibration.from_dict({"temperature": {"noul": 2.5, "choice": 2.0, "score": 2.0},
                                    "thresholds": {"team": 0.8, "priority": 0.7, "refund_requested": 0.85, "churn_risk": 0.85,
                                                   "needs_reasoning": 0.85, "risk": 0.7}})
    be = MockBackend({"team": [0.9, 0.04, 0.03, 0.02, 0.01], "needs_reasoning": [0.07, 0.93], "risk": [0.9, 0.06, 0.03, 0.01],
                      "priority": [0.1, 0.2, 0.7], "refund_requested": [0.9, 0.1], "churn_risk": [0.1, 0.9]})
    be.declared = {"team": list(TICKET_TRIAGE["team"]["criteria"])}
    req = {"state": {"ticket": "Charged twice for my subscription, please refund."}, "questions": TICKET_TRIAGE}
    assert legacy.t_for("needs_reasoning", TICKET_TRIAGE["needs_reasoning"], req["state"]) == 2.5    # avant : 1 (lu brut)
    res = FusionRouter(SystemOneEngine(be, calibration=legacy, model_name="m"), None, GatePolicy.from_calibration(legacy)).decide(req)
    assert res.s1.answers["needs_reasoning"].noul > 0.2 and not res.gated["needs_reasoning"] and not res.gated["risk"]   # avant : S1 seul
    # fichier des graines de Prophet : le `risk` du garde-fou reste brut
    seeds = Calibration.from_dict({"temperature": {"noul": 2.5, "choice": 2.0, "score": 2.0}, "thresholds": {"direct": 0.7, "risk": 0.7}})
    assert seeds.t_for("risk", JUDGE_QUESTIONS["risk"], GUARD) == 1.0
    # un seuil ajuste apres temperature ne s'applique jamais a une lecture brute (autre forme d'etat) : seuil par defaut
    q3 = {"team": {"type": "choice", "instructions": "Which team?", "criteria": {"payments": None, "account": None, "other": None}}}
    sure, wrong = {"team": [0.97, 0.02, 0.01]}, {"team": [0.9, 0.05, 0.05]}     # 16 justes, 4 fausses mais sures : T > 1
    exs = [{"state": {"ticket": f"t{i}"}, "questions": q3, "labels": {"team": "payments" if i < 16 else "account"}} for i in range(20)]
    cal = calibrate(SystemOneEngine(ScriptedBackend([sure] * 16 + [wrong] * 4, declared={"team": ["payments", "account", "other"]}),
                                    model_name="m"), exs, 0.95, source="test")[0]
    pol = GatePolicy.from_calibration(cal, default_threshold=0.99)
    q = SystemOneRequest(state={"ticket": "x"}, questions=q3).questions["team"]
    assert cal.t_for("team", q, {"ticket": "x"}) > 1 and pol.threshold("team", q, {"ticket": "x"}) == cal.thresholds["team"] < 0.97
    assert pol.threshold("team", q, {"email": "x"}) == 0.99                     # avant : seuil ajuste apres T sur une lecture brute
    tb = MockBackend(dict(sure))
    tb.declared = {"team": ["payments", "account", "other"]}
    router = FusionRouter(SystemOneEngine(tb, calibration=cal, model_name="m"), None, pol)
    assert router.decide({"state": {"ticket": "x"}, "questions": q3}).gated["team"]
    assert not router.decide({"state": {"email": "x"}, "questions": q3}).gated["team"]   # lecture brute 0.97 : avant, S1 seul
    # jev serve : ancien format applique tel quel, mais signale
    from jev_clone.server import build_app
    f = tmp_path / "legacy.json"
    legacy.save(f)
    monkeypatch.setattr("jev_clone.server.LlamaCppBackend", lambda *a, **k: be)
    monkeypatch.setenv("JEV_CALIBRATION", str(f))
    capsys.readouterr()
    out = TestClient(build_app()).post("/v1/systemone", json=req).json()
    assert out["answers"]["needs_reasoning"]["noul"] > 0.2 and "ancien format" in capsys.readouterr().err

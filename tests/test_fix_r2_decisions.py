"""Revue adverse, 2e vague : decisions de S1 (calibration, portes, aiguillage) et leur affichage.

Calibration limitee aux questions ajustees (garde-fou, voix : lectures brutes) ; portes de Prophet sur les seuils
calibres ; seuils sans ex aequo ni arrondi qui elargit l'ensemble ; calibration liee aux poids ; ids longs ; message
d'import ; CLAIMS_ACTION sans faux positifs ; appels d'outil coupes par max_tokens ; apostrophes et NEEDS_TOOLS ;
reprise en voie agent (blocs, budget, reponse finale) ; etat de calibration par tour ; lectures S1 des outils."""

import json
import os
import random
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from jev_clone.calibrate import calibrate, fit, load_examples, threshold_for_precision
from jev_clone.engine import SystemOneEngine
from jev_clone.guard import JUDGE_QUESTIONS
from jev_clone.prophet import CLAIMS_ACTION, INTENTS, LANGUAGES, PROPHET_TURN, Prophet, Workspace
from jev_clone.readout import Calibration, probs_to_logits, question_fingerprint
from jev_clone.tools import AgentLoop, _tool
from prophet_studio import calibration as s1cal
from prophet_studio.config import Paths
from prophet_studio.installer import custom_id
from prophet_studio.sessions import AgentService, SessionStore, reduce_event
from prophet_studio.voice import COMMANDS, route_utterance
from tests.conftest import MockBackend, MockS2, ScriptedBackend

I, L = list(INTENTS), list(LANGUAGES)
TR = ["readonly", "workspace_write", "destructive", "privileged", "exfiltration"]
DONE = {"content": "", "tool_calls": [MockS2.tool_call("done", {"summary": "fini"}, "cd")]}
ROOT = Path(__file__).resolve().parent.parent


def engine(direct=0.1, ok=0.9, nr=0.2, risk=(1, 0, 0, 0), tool_risk=(0.9, 0.04, 0.02, 0.02, 0.02), cal=None, table=None):
    be = MockBackend({"direct": [direct, 1 - direct], "clarify": [0.1, 0.9], "needs_reasoning": [nr, 1 - nr], "risk": list(risk),
                      "intent": [1 / len(I)] * len(I), "language": [1 / len(L)] * len(L), "tool_risk": list(tool_risk),
                      "policy_violation": [0.1, 0.9], "ok": [ok, 1 - ok], **(table or {})}, default_noul=(0.2, 0.8))
    be.declared = {"intent": I, "language": L, "tool_risk": TR}
    return SystemOneEngine(be, calibration=cal, model_name="mock")


# ---- 1. une temperature ne vaut que pour les questions sur lesquelles elle a ete ajustee ----------------------------------------
def _reading(ex, i, sure_noul, sure_choice, right_every):
    """Lecture d'une graine : juste `right_every` fois sur 10, avec l'assurance donnee (sous- ou sur-confiante)."""
    lab, ok = ex["labels"], (i % 10) < right_every

    def noul(g):
        g = bool(g) == ok
        return [sure_noul, 1 - sure_noul] if g else [1 - sure_noul, sure_noul]

    def choice(keys, g):
        g = g if ok else next(k for k in keys if k != g)
        return [sure_choice if k == g else (1 - sure_choice) / (len(keys) - 1) for k in keys]
    return {"direct": noul(lab["direct"]), "clarify": noul(lab["clarify"]), "needs_reasoning": noul(lab["needs_reasoning"]),
            "risk": choice([0, 1, 2, 3], lab["risk"]), "intent": choice(I, lab["intent"]), "language": choice(L, lab["language"])}


def seeds_calibration(sure_noul, sure_choice, right_every) -> Calibration:
    exs = load_examples()
    be = ScriptedBackend([_reading(ex, i, sure_noul, sure_choice, right_every) for i, ex in enumerate(exs)], declared={"intent": I, "language": L})
    return calibrate(SystemOneEngine(be, model_name="m"), exs, source="test")[0]


UNDER = dict(sure_noul=0.6, sure_choice=0.45, right_every=10)   # juste mais peu assure : T < 1 (avant : garde-fou moins prudent)
OVER = dict(sure_noul=0.99, sure_choice=0.99, right_every=7)    # sur-confiant : T > 1 (avant : commandes vocales perdues)


@pytest.mark.parametrize("kind", ["under", "over"])
def test_calibration_leaves_guard_voice_and_computer_use_readings_raw(tmp_path, kind):
    cal = seeds_calibration(**(UNDER if kind == "under" else OVER))
    raw = engine(tool_risk=(0.55, 0.05, 0.14, 0.13, 0.13), risk=(0.4, 0.3, 0.2, 0.1))
    calibrated = engine(tool_risk=(0.55, 0.05, 0.14, 0.13, 0.13), risk=(0.4, 0.3, 0.2, 0.1), cal=cal)
    ws = Workspace(tmp_path / "ws")
    j_raw = Prophet(raw, None, ws)._judge_one("nettoie le dossier", "run_command: rm -rf ./data")
    j_cal = Prophet(calibrated, None, ws)._judge_one("nettoie le dossier", "run_command: rm -rf ./data")
    assert j_cal["needs_confirmation"] and {k: j_cal[k] for k in ("p_risky", "risk", "policy_violation", "tool_risk")} == \
        {k: j_raw[k] for k in ("p_risky", "risk", "policy_violation", "tool_risk")}          # avant : p_risky 0.40 -> 0.07 (under)
    # voix : meme `intent` de nom, autre question ; computer use : autres questions, autre etat
    crit = list(COMMANDS) + ["prompt"]
    table = {"intent": [0.75 if c == "new_session" else 0.25 / (len(crit) - 1) for c in crit], "action": [0.7, 0.3], "achieved": [0.3, 0.7]}
    be_raw, be_cal = MockBackend(dict(table)), MockBackend(dict(table))
    be_raw.declared = be_cal.declared = {"intent": crit}
    r_raw = route_utterance("start a new chat session please", SystemOneEngine(be_raw, model_name="m"))
    r_cal = route_utterance("start a new chat session please", SystemOneEngine(be_cal, calibration=cal, model_name="m"))
    assert r_raw.command == r_cal.command == "new_session" and r_raw.confidence == r_cal.confidence   # avant (over) : prompt
    cu = {"state": {"goal": "open the page", "url": "x"}, "questions": {"achieved": {"type": "noul", "instructions": "Is the goal achieved?"}}}
    assert SystemOneEngine(be_cal, calibration=cal, model_name="m").answer(cu).answers["achieved"].noul == pytest.approx(0.3, abs=1e-6)
    # ... alors que le pre-tour de Prophet, lui, est bien lu a la temperature ajustee
    pre = {"state": {"request": "Salut", "workspace_files": [], "recent_turns": []}, "questions": PROPHET_TURN}
    assert calibrated.answer(pre).answers["direct"].noul != raw.answer(pre).answers["direct"].noul
    ts = {q: e["T"] for q, e in cal.questions.items()}                        # une temperature par question du pre-tour
    assert set(ts) == {"direct", "clarify", "intent", "language", "needs_reasoning", "risk"}
    assert (all(t < 1 for t in ts.values()) if kind == "under" else all(t > 1 for t in ts.values())), ts


def test_legacy_and_generic_calibration_files():
    legacy = Calibration.from_dict({"temperature": {"noul": 2.5, "choice": 0.2, "score": 2.0},
                                    "thresholds": {"direct": 0.7, "intent": 0.5, "risk": 0.5}})
    raw, cal = engine(direct=0.93, tool_risk=(0.55, 0.05, 0.14, 0.13, 0.13)), engine(direct=0.93, tool_risk=(0.55, 0.05, 0.14, 0.13, 0.13), cal=legacy)
    pre = {"state": {"request": "Salut", "workspace_files": []}, "questions": PROPHET_TURN}
    a_raw, a_cal = raw.answer(pre).answers, cal.answer(pre).answers
    assert a_cal["direct"].noul < a_raw["direct"].noul and a_cal["clarify"].noul == a_raw["clarify"].noul   # clarify : pas dans les seuils
    guard = {"state": {"user_request": "x", "proposed_action": "rm -rf ./data"}, "questions": JUDGE_QUESTIONS}
    assert cal.answer(guard).model_dump()["answers"] == raw.answer(guard).model_dump()["answers"]
    # Studio : un ancien fichier ne vaut que pour les questions du pre-tour (pas un tool_risk d'un jeu de validation)
    mixed = s1cal.for_studio(Calibration.from_dict({"temperature": {"noul": 2.0, "choice": 0.2, "score": 1.0},
                                                    "thresholds": {"direct": 0.7, "tool_risk": 0.9}}))
    assert set(mixed.questions) == {"direct"} and mixed.questions["direct"]["T"] == 2.0
    assert mixed.t_for("tool_risk", JUDGE_QUESTIONS["tool_risk"], guard["state"]) == 1.0
    # temperature seule (entrainement, benchmark) : generique pour jev serve, jamais appliquee par Studio (garde-fou, voix)
    generic = Calibration(temperature={"noul": 1.0, "choice": 3.0, "score": 1.0})
    assert generic.generic and generic.t_for("tool_risk", None, None, "choice") == 3.0
    ok, why = s1cal.usable(generic)
    assert not ok and "generique" in why


# ---- 2. les portes de Prophet lisent les seuils calibres ------------------------------------------------------------------------
def turn_calibration(**spec) -> Calibration:
    """{question: (T, seuil)} ajustes sur les graines (forme d'etat `request`)."""
    return Calibration(questions={q: {"T": T, "kind": PROPHET_TURN[q]["type"], "fp": question_fingerprint(PROPHET_TURN[q]), "state": ["request"],
                                      "n": 82} for q, (T, _) in spec.items()},
                       thresholds={q: thr for q, (_, thr) in spec.items()})


def test_prophet_gates_use_calibrated_thresholds(tmp_path):
    ws = Workspace(tmp_path / "ws")
    # fichier de Studio de l'ancien format (temperature par primitive) : ce que "Calibrer" ecrivait (T noul 2.51...)
    for make in (lambda: Calibration.from_dict({"temperature": {"noul": 2.51, "choice": 2.95, "score": 2.5},
                                                "thresholds": {"direct": 0.73, "needs_reasoning": 0.6, "risk": 0.45}}),
                 lambda: turn_calibration(direct=(2.51, 0.73), needs_reasoning=(2.51, 0.6), risk=(2.5, 0.45))):
        cal, ev = make(), []
        s2 = MockS2([{"content": "Ca va tres bien, merci !"}])
        t = Prophet(engine(direct=0.93, risk=(0.8, 0.1, 0.05, 0.05), cal=cal), s2, ws, on_event=ev.append).handle("Salut Jarvis, ca va ?")
        d = next(e for e in ev if e["type"] == "s1.decision")
        assert 0.73 < t.pre["direct"]["noul"] < 0.8 and t.pre["risk"]["score"] > 0.5            # lectures calibrees
        assert t.path == "direct" and d["risk_level"] == 0 and d["budget"] == 0 and len(s2.calls) == 1   # avant : agent, risque 1, 512
        assert d["calibrated"] is True and d["gates"] == {"direct": 0.73, "needs_reasoning": 0.6, "risk": 0.45}
    # la meme calibration ne change rien au garde-fou (etat sans `request`)
    j = Prophet(engine(risk=(0.8, 0.1, 0.05, 0.05), cal=cal), None, ws)._judge_one("x", "shell: ls")
    assert j == {**Prophet(engine(risk=(0.8, 0.1, 0.05, 0.05)), None, ws)._judge_one("x", "shell: ls"), "latency_ms": j["latency_ms"]}
    # seuil null (precision visee jamais atteinte) : jamais de voie directe, et on le montre
    ev = []
    t = Prophet(engine(direct=0.99, cal=turn_calibration(direct=(1.0, None))), MockS2([DONE]), ws, on_event=ev.append).handle("Salut")
    assert t.path == "agent" and next(e for e in ev if e["type"] == "s1.decision")["gates"] == {"direct": None}
    # lecture de risque peu sure sous son seuil : pas de voie directe, un peu de reflexion, niveau affiche non gonfle
    ev = []
    t = Prophet(engine(direct=0.99, risk=(0.5, 0.2, 0.2, 0.1), cal=turn_calibration(risk=(1.0, 0.8))), MockS2([DONE]), ws,
                on_event=ev.append).handle("Salut")
    d = next(e for e in ev if e["type"] == "s1.decision")
    assert t.path == "agent" and d["risk_level"] == 0 and d["budget"] == 512
    # sans calibration : portes d'origine (0.8, 0.5, E[risque] arrondi)
    ev = []
    t = Prophet(engine(direct=0.93), MockS2([{"content": "Bien."}]), ws, on_event=ev.append).handle("Salut")
    d = next(e for e in ev if e["type"] == "s1.decision")
    assert t.path == "direct" and d["calibrated"] is False and d["gates"] == {} and d["s1_model"] == "mock"


# ---- 3. seuil de precision : ex aequo et arrondi ------------------------------------------------------------------------------
def test_threshold_for_precision_handles_ties_and_never_widens_the_set():
    conf = [0.999999] * 10 + [0.7] * 5
    correct = [1] * 9 + [0] + [1, 0, 1, 0, 1]
    rng = random.Random(0)
    for _ in range(5):                       # 10 ex aequo a 90 % : aucun seuil n'atteint 95 %, quel que soit l'ordre
        idx = list(range(len(conf)))
        rng.shuffle(idx)
        assert threshold_for_precision(np.array(conf)[idx], np.array(correct)[idx], 0.95) is None
    # l'arrondi a 1e-4 ne doit pas admettre les cas juste sous le seuil (avant : 0.9500 admettait 0.95002)
    ps = [0.99] * 10 + [0.95008] + [0.95002] * 3
    ok = [1] * 11 + [0] * 3
    rows = [probs_to_logits(np.array([p, 1 - p])) for p in ps]
    cal, _ = fit({"noul": ([], []), "choice": ([], []), "score": ([], [])}, {"q": ("noul", rows, [0 if o else 1 for o in ok])}, 0.95)
    thr = cal.thresholds["q"]
    admitted = [o for p, o in zip(ps, ok) if p >= thr]
    assert admitted and sum(admitted) / len(admitted) >= 0.95 and len(admitted) == 11
    # donnees continues : l'ensemble au-dessus du seuil rendu a la precision visee
    r = np.random.default_rng(1)
    c = r.uniform(0.5, 1.0, 2000)
    k = (r.uniform(0, 1, 2000) < c).astype(float)
    t = threshold_for_precision(c, k, 0.9)
    m = np.round(c, 6) >= t
    assert t is not None and k[m].mean() >= 0.9


# ---- 4. une calibration ne vaut que pour les poids sur lesquels elle a ete ajustee -------------------------------------------------
def test_calibration_is_bound_to_the_weights(tmp_path):
    gguf = tmp_path / "clone.gguf"
    gguf.write_bytes(b"GGUF" + b"\0" * 4096)
    runs, mid = tmp_path / "runs", "custom-s1-clone"
    s1cal.store(runs, mid, Calibration(temperature={"choice": 2.0}, thresholds={"intent": None}), weights=gguf)
    svc = AgentService(SessionStore(tmp_path / "s"), lambda e: None, urls=lambda: ("http://127.0.0.1:1", "http://127.0.0.1:2"),
                       settings=lambda: None, ctx=lambda: 8192, calibration=lambda: s1cal.calibration_file(runs, mid))
    assert svc.engines()[0].cal.t("choice") == 2.0 and "error" not in s1cal.status(runs)[mid]
    st = gguf.stat()
    os.utime(gguf, ns=(st.st_atime_ns, st.st_mtime_ns + 5 * 10**9))          # copie a l'identique, date changee : toujours valable
    assert svc.engines()[0].cal.t("choice") == 2.0 and "error" not in s1cal.status(runs)[mid]
    gguf.write_bytes(b"GGUF" + b"\1" * 4096)                                  # nouveaux poids, meme taille, meme chemin
    os.utime(gguf, ns=(st.st_atime_ns, st.st_mtime_ns + 9 * 10**9))
    assert svc.engines()[0].cal.t("choice") == 1.0                              # avant : ancienne calibration appliquee
    s = s1cal.status(runs)[mid]
    assert s["stale"] and "perimee" in s["error"]


def test_studio_binds_imported_calibration_to_the_gguf(tmp_path):
    from prophet_studio.server import Studio, build_app
    gguf = tmp_path / "MyJevClone-Q4.gguf"
    gguf.write_bytes(b"GGUF" + b"\0" * 2048)
    st = Studio(Paths(tmp_path / "home"), "tok", 7878)
    h = {"X-Prophet-Token": "tok"}
    with TestClient(build_app(st), base_url="http://127.0.0.1:7878") as c:
        mid = c.post("/api/import", headers=h, json={"path": str(gguf), "role": "s1"}).json()["id"]
        data = {"temperature": {"noul": 1.5, "choice": 1.2, "score": 1.1}, "thresholds": {"direct": 0.7}}
        assert c.post("/api/calibration/import", headers=h, json={"data": data, "model_id": mid}).status_code == 200
        assert "error" not in c.get("/api/state", headers=h).json()["installed"]["calibration"][mid]
        gguf.write_bytes(b"GGUF" + b"\2" * 4096)                                # export du notebook copie par-dessus
        cal = c.get("/api/state", headers=h).json()["installed"]["calibration"][mid]
        assert cal.get("stale") and "error" in cal                                 # l'interface affiche "non calibre"


# ---- 5. identifiants longs : bornes a l'import, refus clair avant la calibration -------------------------------------------------
def test_long_custom_ids_are_bounded_and_rejected_early(tmp_path, monkeypatch):
    from prophet_studio.server import Studio, build_app
    stem = "jev-clone-" + "x" * 125
    mid = custom_id("s1", stem)
    assert len(mid) <= 100 and s1cal.SAFE_ID.match(mid) and mid == custom_id("s1", stem) and mid != custom_id("s1", stem + "y")
    assert custom_id("s1", "模型").startswith("custom-s1-") and s1cal.SAFE_ID.match(custom_id("s1", "模型"))
    st = Studio(Paths(tmp_path / "home"), "tok", 7878)
    h = {"X-Prophet-Token": "tok"}
    gguf = tmp_path / f"{stem}.gguf"
    gguf.write_bytes(b"GGUF")
    long_id = "custom-s1-" + "y" * 140                                            # registre d'une version precedente
    st.installer.registry["models"][long_id] = {"main": str(gguf), "role": "s1", "custom": True}
    monkeypatch.setattr(s1cal, "active_s1_model", lambda rt: long_id)
    monkeypatch.setattr(s1cal, "run", lambda *a, **k: pytest.fail("la calibration ne doit pas demarrer"))
    st.runtime.state = "ready"
    with TestClient(build_app(st), base_url="http://127.0.0.1:7878") as c:
        assert c.post("/api/import", headers=h, json={"path": str(gguf), "role": "s1"}).json()["id"] == mid
        r = c.post("/api/calibrate", headers=h, json={})
        assert r.status_code == 422 and "128" in r.json()["detail"] and "re-importez" in r.json()["detail"]   # avant : 22 s puis 500
        r = c.post("/api/calibration/import", headers=h, json={"data": {"temperature": {"noul": 1.2}}, "model_id": long_id})
        assert r.status_code == 422 and "re-importez" in r.json()["detail"]


# ---- 6. import : pas d'option que l'interface ne peut pas envoyer --------------------------------------------------------------
def test_mismatched_import_message_is_actionable_in_the_ui(tmp_path):
    with pytest.raises(ValueError) as e:
        s1cal.import_calibration(tmp_path, "ternary-4b", {"temperature": {"noul": 1.2}, "meta": {"model_id": "ternary-1.7b"}})
    assert "force" not in str(e.value) and "Calibrer" in str(e.value) and "ternary-1.7b" in str(e.value)


# ---- 7-9. aiguillage de la voie directe --------------------------------------------------------------------------------------------
BENIGN = ["Voici un poème que j'ai écrit pour toi :\nLa lune veille...", "Here is a short poem I've written for you.", "I have edited it for clarity.",
          "J'ai modifié quelques tournures : ...", "I've updated the list with your suggestions.", "I have created a short story for you.",
          "I ran the numbers: 12 x 7 = 84.", "J'ai lancé les dés : 4 et 2.", "J’ai reformulé ton paragraphe : ...", "Aujourd’hui c’est mardi.",
          "J'ai écrit un petit serveur Node.js pour toi :\n```js\nconsole.log(1)\n```", "I've written a function that does this: ```def f(): pass```"]
CLAIMS = ["J'ai créé le fichier index.html.", "J’ai créé le fichier hello.py.", "I’ve created hello.py for you.", "I wrote hello.py",
          "Je n’ai pas accès à vos fichiers.", "I can’t access your files.", "I ran it and it prints 42.", "J'ai exécuté ton code.",
          "I have saved the file to disk.", "I've just installed the package requests.", "J'ai supprimé le dossier build.", "I've updated config.yaml"]


def test_claims_action_only_flags_real_action_claims(tmp_path):
    assert [b for b in BENIGN if CLAIMS_ACTION.search(b)] == []
    assert [c for c in CLAIMS if not CLAIMS_ACTION.search(c)] == []
    ws = Workspace(tmp_path / "ws")
    for req, answer in (("Ecris-moi un petit poeme sur la lune", BENIGN[0]), ("Reformule ce paragraphe", BENIGN[3]),
                        ("Write me a haiku about rain", "Here is a haiku I've written:\nsoft rain on the roof")):
        s2 = MockS2([{"content": answer}, DONE])
        t = Prophet(engine(direct=0.95, ok=0.9), s2, ws).handle(req)
        assert t.path == "direct" and len(s2.calls) == 1 and t.verification == 0.9, (req, t.pre.get("rerouted"))   # avant : agent, 3 appels


@pytest.mark.parametrize("answer,reason", [("`NEEDS_TOOLS`", "needs_tools"), ("**NEEDS_TOOLS**", "needs_tools"),
                                           ("Il faut lire vos fichiers. NEEDS_TOOLS", "needs_tools"), ("NEEDS\\_TOOLS", "needs_tools"),
                                           ("J’ai créé le fichier hello.py.", "claims_action"), ("I’ve created hello.py for you.", "claims_action")])
def test_wrapped_needs_tools_and_typographic_claims_are_rerouted(tmp_path, answer, reason):
    ev: list = []
    t = Prophet(engine(direct=0.95, ok=0.5), MockS2([{"content": answer}, DONE]), Workspace(tmp_path / "ws"), on_event=ev.append).handle("fais-le")
    assert t.path == "agent" and next(e for e in ev if e["type"] == "s1.reroute")["reason"] == reason   # avant : livre tel quel


# ---- 8. appel d'outil coupe par max_tokens -----------------------------------------------------------------------------------------
class CutS2(MockS2):
    """Comme llama-server : refuse un historique dont un appel d'outil n'est pas du JSON ; coupe (length) les appels listes."""

    def __init__(self, replies, cut=(1,)):
        super().__init__(replies)
        self.cut = cut

    def chat(self, messages, **kw):
        for m in messages:
            for tc in m.get("tool_calls") or []:
                json.loads(tc["function"]["arguments"])        # 500 'Failed to parse tool call arguments as JSON'
        r = super().chat(messages, **kw)
        if len(self.calls) in self.cut:
            r["choices"][0]["finish_reason"] = "length"
        return r


PARTIAL = {"content": "", "tool_calls": [{"id": "w1", "type": "function",
                                          "function": {"name": "write_file", "arguments": '{"path": "index.html", "content": "<!doctype html><html>'}}]}


def test_truncated_tool_call_is_not_run_and_stops_after_two(tmp_path):
    ran: list = []
    tools = {"write_file": (_tool("write_file", "w", {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
                            lambda a: ran.append(a) or {"ok": True, "path": a["path"]})}
    ev: list = []
    s2 = CutS2([PARTIAL, {"content": "", "tool_calls": [MockS2.tool_call("write_file", {"path": "index.html", "content": "<p>1</p>"}, "w2")]},
                {"content": "fini"}], cut=(1,))
    res = AgentLoop(s2, extra_tools=tools, max_turns=6, thinking_budget=800, max_tokens=1365, on_event=ev.append).run([{"role": "user", "content": "app"}])
    assert ran == [{"path": "index.html", "content": "<p>1</p>"}] and res.stopped_by == "final"   # avant : write_file({}) execute
    err = json.loads(next(m for m in s2.calls[1][0] if m["role"] == "tool")["content"])
    assert err["truncated"] and "max_tokens=1365" in err["error"] and not err["ok"]
    assert s2.calls[1][1]["thinking_budget"] < 800 and s2.calls[2][1]["thinking_budget"] == 800   # nouvel essai : place pour l'appel
    assert any(e["type"] == "tool.result" and e["ok"] is False and e["result"].get("truncated") for e in ev)
    # deux coupes de suite : arret explicite (avant : 16 generations, stopped_by max_turns, '(no summary)')
    ran.clear()
    s2 = CutS2([PARTIAL], cut=tuple(range(1, 30)))
    t = Prophet(engine(), s2, Workspace(tmp_path / "ws"), permission_mode="auto", max_tokens=1365).handle("cree une petite app web")
    assert ran == [] and len(s2.calls) == 2 and t.stopped_by == "length" and "tronquee" in t.response and t.response != "(no summary)"


# ---- 10-12, 15. reprise en voie agent : blocs, budget, reponse finale ------------------------------------------------------------------
class StreamS2(MockS2):
    """Diffuse le contenu par on_delta (comme le vrai backend)."""

    def chat(self, messages, **kw):
        r = super().chat(messages, **kw)
        if kw.get("on_delta") and r["choices"][0]["message"].get("content"):
            kw["on_delta"]({"type": "content", "text": r["choices"][0]["message"]["content"]})
        return r


def rerouted_events(tmp_path, effort="auto", direct_text="J'ai créé le fichier notes.txt avec votre liste.", agent=None):
    ev: list = []
    agent = agent or [{"content": "Je regarde l'espace de travail d'abord.", "tool_calls": [MockS2.tool_call("done", {"summary": "notes.txt cree"}, "d1")]}]
    s2 = StreamS2([{"content": direct_text}, *agent])
    t = Prophet(engine(direct=0.95, ok=0.9), s2, Workspace(tmp_path / "ws"), on_event=ev.append).handle("note ma liste", effort=effort)
    return t, s2, [{**e, "ts": 1000.0 + i} for i, e in enumerate(ev)]


def test_reroute_opens_a_new_block_and_records_the_real_budget(tmp_path):
    for effort in ("fast", "auto"):
        t, s2, ev = rerouted_events(tmp_path, effort)
        item: dict = {}
        for e in ev:
            reduce_event(item, e)
        at = item["reroute"]["at"]
        texts = [b["text"] for b in item["blocks"] if b["type"] == "text"]
        assert texts == ["J'ai créé le fichier notes.txt avec votre liste.", "Je regarde l'espace de travail d'abord."]   # avant : colles
        assert item["blocks"][at]["text"].startswith("Je regarde")
        rr = next(e for e in ev if e["type"] == "s1.reroute")
        assert rr["budget"] == s2.calls[1][1]["thinking_budget"] == t.pre["rerouted"]["budget"] == item["reroute"]["budget"]
        assert item["s1"]["path"] == "agent" and item["s1"]["rerouted_from"] == "direct" and item["s1"]["budget"] == rr["budget"]
        assert rr["budget"] == (0 if effort == "fast" else 512)


def _node():
    node = shutil.which("node")
    if node is None:
        return None
    v = subprocess.run([node, "--version"], capture_output=True, text=True).stdout.strip().lstrip("v").split(".")
    return node if (int(v[0]), int(v[1])) >= (22, 6) else None


@pytest.mark.skipif(_node() is None, reason="node >= 22.6 requis (types TypeScript effaces)")
def test_ui_reducer_mirrors_python_after_a_reroute(tmp_path):
    t, _, ev = rerouted_events(tmp_path)                                                   # l'agent confirme la reponse directe
    _, _, ev2 = rerouted_events(tmp_path, direct_text="La capitale de l'Australie est Canberra. J'ai créé le fichier a.txt",
                                agent=[{"content": "", "tool_calls": [MockS2.tool_call("done", {"summary": "La capitale de l'Australie est Canberra."}, "d")]}])
    for e in ev2:
        if e["type"] == "turn.end":
            assert e["response"] == "La capitale de l'Australie est Canberra."
    script = tmp_path / "reduce.mts"
    script.write_text(
        f'import {{ newAssistant, reduce, responseAlreadyShown }} from {json.dumps((ROOT / "ui/src/lib/transcript.ts").as_uri())};\n'
        'import { readFileSync } from "node:fs";\n'
        'const out = [];\n'
        'for (const evs of JSON.parse(readFileSync(process.argv[2], "utf8"))) {\n'
        '  const it = newAssistant("t");\n'
        '  for (const e of evs) reduce(it, e);\n'
        '  out.push({ blocks: it.blocks.map((b) => ({ type: b.type, text: b.text ?? null })), s1: it.s1, reroute: it.reroute, shown: responseAlreadyShown(it) });\n'
        '}\n'
        'console.log(JSON.stringify(out));\n', encoding="utf-8")
    data = tmp_path / "events.json"
    data.write_text(json.dumps([ev, ev2]), encoding="utf-8")
    r = subprocess.run([_node(), "--experimental-strip-types", "--no-warnings", str(script), str(data)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert r.returncode == 0, r.stderr
    ts_items = json.loads(r.stdout)
    assert ts_items[1]["shown"] is False     # avant : reponse finale cachee (egale au texte ecarte de la voie directe)
    assert [b["text"] for b in ts_items[0]["blocks"] if b["type"] == "text"] == \
        ["J'ai créé le fichier notes.txt avec votre liste.", "Je regarde l'espace de travail d'abord."]   # avant : colles, attenues
    for evs, ts in zip((ev, ev2), ts_items):
        py: dict = {}
        for e in evs:
            reduce_event(py, e)
        assert ts["blocks"] == [{"type": b["type"], "text": b.get("text")} for b in py["blocks"]]
        assert ts["reroute"] == py["reroute"] and ts["s1"]["path"] == py["s1"]["path"] == "agent"
        assert ts["s1"]["budget"] == py["s1"]["budget"] and ts["s1"]["rerouted_from"] == "direct"
        assert ts["s1"]["calibrated"] is False and py["s1"]["calibrated"] is False


# ---- 13-14. calibration enregistree avec chaque tour --------------------------------------------------------------------------------
def test_calibrated_flag_is_recorded_per_turn(tmp_path):
    holder: dict = {}
    store = SessionStore(tmp_path / "sessions")
    st = SimpleNamespace(permission_mode="auto", effort="auto", workspace=str(tmp_path / "ws"), browser_tool=False, desktop_tool=False)
    svc = AgentService(store, lambda e: None, urls=lambda: ("", ""), settings=lambda: st, ctx=lambda: 8192)
    svc.engines = lambda: (holder["s1"], MockS2([DONE]))
    sid = store.create(str(tmp_path / "ws"))["id"]
    for cal in (None, Calibration.from_dict({"temperature": {"noul": 2.0, "score": 2.0}, "thresholds": {"direct": 0.7, "risk": 0.5}}), Calibration()):
        holder["s1"] = engine(cal=cal)
        svc.submit(sid, "fais x")
        assert svc.wait_idle(sid, 10)
    items = [i for i in store.load(sid)["transcript"] if i["kind"] == "assistant"]
    assert [i["s1"]["calibrated"] for i in items] == [False, True, False]      # avant : absent, deduit du runtime a l'affichage
    assert items[1]["s1"]["s1_model"] == "mock" and items[1]["s1"]["gates"] == {"direct": 0.7, "risk": 0.5}


# ---- 16-17. lectures S1 faites dans les outils browse / desktop ----------------------------------------------------------------------
def test_turn_stats_include_s1_reads_of_browse_and_desktop(tmp_path):
    ws = Workspace(tmp_path / "ws")
    calls = [MockS2.tool_call("desktop", {"goal": "ouvre la calculatrice"}, "d1"), MockS2.tool_call("browse", {"goal": "cherche"}, "b1")]
    s2 = MockS2([{"content": "", "tool_calls": calls}, DONE])
    p = Prophet(engine(), s2, ws, permission_mode="auto", max_tools=20,
                desktop_factory=lambda: (lambda g, a, s: {"ok": True, "s1_ms": 120.5, "s1_calls": 7}),
                browser_factory=lambda: (lambda g, u, s: {"ok": True, "steps": 2}))       # sans les champs : tolere
    t = p.handle("fais-le")
    base = Prophet(engine(), MockS2([{"content": "", "tool_calls": calls}, DONE]), ws, permission_mode="auto", max_tools=20,
                   desktop_factory=lambda: (lambda g, a, s: {"ok": True}), browser_factory=lambda: (lambda g, u, s: {"ok": True})).handle("fais-le")
    assert [c["tool"] for c in t.tool_calls] == ["desktop", "browse", "done"]
    assert t.stats["s1_calls"] == base.stats["s1_calls"] + 7 and t.stats["s1_ms"] >= 120.5   # avant : les 7 lectures manquaient

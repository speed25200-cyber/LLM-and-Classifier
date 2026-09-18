import os

import numpy as np
import pytest

from jev_clone.backend_llamacpp import BranchResult, LlamaCppBackend
from jev_clone.engine import SystemOneEngine
from jev_clone.prompt import PromptFormat, build_branches, label_grammar
from jev_clone.schema import SystemOneRequest
from tests.conftest import MockBackend

OPTS = [f"opt_{i}" for i in range(1, 31)]  # 30 options : au-dela des 26 lettres, prefixes partages


def test_values_branch_and_grammar():
    req = SystemOneRequest(state="s", questions={"q": {"type": "choice", "instructions": "Which?", "criteria": OPTS}})
    br = build_branches("q", req.questions["q"], PromptFormat())
    assert len(br) == 1 and br[0].terminator == '"' and br[0].labels == OPTS and br[0].keys == OPTS
    assert br[0].text.endswith('<think>\n\n</think>\n\n"') and '- "opt_30"' in br[0].text
    assert label_grammar(["MANUAL", "MANUAL_REVIEW"], '"') == 'root ::= "MANUAL\\"" | "MANUAL_REVIEW\\""'
    # <= 26 options : lettres par defaut, valeurs sur demande
    small = {"type": "choice", "instructions": "?", "criteria": ["a", "b", "c"]}
    assert build_branches("q", SystemOneRequest(state="s", questions={"q": small}).questions["q"], PromptFormat())[0].terminator == ""
    assert build_branches("q", SystemOneRequest(state="s", questions={"q": small}).questions["q"], PromptFormat(), label_mode="values")[0].terminator == '"'
    with pytest.raises(ValueError):
        SystemOneRequest(state="s", questions={"q": {"type": "choice", "instructions": "?", "criteria": ['bad"quote', "ok"]}})


class FakeValues(LlamaCppBackend):
    """Simule llama-server : distribution scriptee par (fin de prompt) pour tester la resolution des prefixes."""

    def __init__(self, table):
        super().__init__("http://fake")
        self.table, self.calls = table, []

    def _top_probs(self, prompt, grammar, n_probs):
        self.calls.append((prompt[-12:], grammar))
        for suffix, top in self.table.items():
            if prompt.endswith(suffix):
                return top, {}
        raise AssertionError(f"prompt inattendu : {prompt[-30:]!r}")


def test_resolve_shared_prefixes():
    # options : MANUAL, MANUAL_REVIEW, APPROVE. 1er token : "MAN" (collision MANUAL / MANUAL_REVIEW), "AP" (APPROVE), "APP" (APPROVE)
    be = FakeValues({
        '\n"': [{"token": "MAN", "prob": 0.6}, {"token": "AP", "prob": 0.3}, {"token": "APP", "prob": 0.1}],
        '"MAN': [{"token": "UAL\"", "prob": 0.25}, {"token": "UAL", "prob": 0.75}],            # UAL : encore ambigu
        '"MANUAL': [{"token": "\"", "prob": 0.2}, {"token": "_REV", "prob": 0.8}],
    })
    out = {"MANUAL": 0.0, "MANUAL_REVIEW": 0.0, "APPROVE": 0.0}
    be._resolve_values('x\n"', {k: k for k in out}, '"', 1.0, out, 0, [64])
    assert abs(out["APPROVE"] - 0.4) < 1e-9
    assert abs(out["MANUAL"] - (0.6 * 0.25 + 0.6 * 0.75 * 0.2)) < 1e-9
    assert abs(out["MANUAL_REVIEW"] - 0.6 * 0.75 * 0.8) < 1e-9
    assert abs(sum(out.values()) - 1) < 1e-9 and len(be.calls) == 3


def test_engine_values_mode_with_mock(ticket_request):
    be = MockBackend({"q": [0.02] * 29 + [0.42]})
    be.declared = {"q": OPTS}
    r = SystemOneEngine(be, model_name="mock").answer({"state": "s", "questions": {"q": {"type": "choice", "instructions": "?", "criteria": OPTS}}})
    assert r.answers["q"].choice == "opt_30" and len(r.answers["q"].probabilities) == 30


URL = os.environ.get("JEV_TEST_SERVER")


@pytest.mark.skipif(not URL, reason="JEV_TEST_SERVER non defini")
def test_live_values_mode():
    eng = SystemOneEngine(LlamaCppBackend(URL, max_workers=2))
    opts = OPTS + ["MANUAL", "MANUAL_REVIEW", "APPROVE"]
    r = eng.answer({"state": "Agent tool call: kubectl delete namespace prod",
                    "questions": {"q": {"type": "choice", "instructions": "Pick the option.", "criteria": opts}}})
    p = r.answers["q"].probabilities
    assert set(p) == set(opts) and abs(sum(p.values()) - 1) < 1e-4 and all(v >= 0 for v in p.values())
    # meme etat, 3 options, mode valeurs force : somme a 1 et cache du prefixe reutilise
    r2 = eng.answer({"state": "Agent tool call: kubectl delete namespace prod", "label_mode": "values",
                     "questions": {"q": {"type": "choice", "instructions": "Pick the option.", "criteria": {"MANUAL": None, "MANUAL_REVIEW": "escalate", "APPROVE": None}}}})
    assert abs(sum(r2.answers["q"].probabilities.values()) - 1) < 1e-4

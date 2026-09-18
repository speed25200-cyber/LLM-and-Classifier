import os

import pytest

from jev_clone.engine import SystemOneEngine
from jev_clone.guided import THINK_CLOSE, GuidedGenerator
from tests.conftest import ScriptedBackend


class RawS2:
    """Bonsai factice bas niveau : renvoie des morceaux scriptes a chaque complete()."""

    def __init__(self, chunks):
        self.chunks, self.calls = chunks, []

    def apply_template(self, messages, **kw):
        return "<|im_start|>user\n" + messages[-1]["content"] + "<|im_end|>\n<|im_start|>assistant\n"

    def complete(self, prompt, n_predict=256, stop=None, temperature=0.7, **kw):
        self.calls.append({"prompt": prompt, "n_predict": n_predict, "stop": stop})
        c = self.chunks[min(len(self.calls) - 1, len(self.chunks) - 1)]
        return {"content": c["content"], "stop_type": c.get("stop_type", "limit"), "stopping_word": "", "tokens": c.get("tokens", n_predict), "timings": {}}


def test_think_adaptive_stops_when_determined():
    # controle 1 : pas determine ; controle 2 : determine -> on ferme </think> et on repond
    s1 = ScriptedBackend([{"determined": [0.2, 0.8], "stuck": [0.1, 0.9]}, {"determined": [0.95, 0.05], "stuck": [0.1, 0.9]}])
    s2 = RawS2([{"content": "Let me think. 2+2...", "tokens": 8}, {"content": " so the sum is 4.", "tokens": 8}, {"content": "4", "tokens": 1, "stop_type": "eos"}])
    g = GuidedGenerator(s2, SystemOneEngine(s1, model_name="mock"), check_every=8, max_think=64, stop_conf=0.8)
    r = g.think_adaptive([{"role": "user", "content": "Solve 2+2"}])
    assert r.stopped_by == "determined" and r.think_tokens == 16 and r.answer == "4" and len(r.checks) == 2
    assert s2.calls[-1]["prompt"].endswith(THINK_CLOSE) and "2+2..." in s2.calls[-1]["prompt"]
    assert s2.calls[0]["stop"] == ["</think>"] and s2.calls[0]["n_predict"] == 8


def test_think_adaptive_model_closes_itself_and_budget():
    s1 = ScriptedBackend([{"determined": [0.1, 0.9], "stuck": [0.1, 0.9]}])
    s2 = RawS2([{"content": "short reasoning", "tokens": 3, "stop_type": "word"}, {"content": "42", "tokens": 1, "stop_type": "eos"}])
    r = GuidedGenerator(s2, SystemOneEngine(s1, model_name="mock"), check_every=8).think_adaptive([{"role": "user", "content": "q"}])
    assert r.stopped_by == "model" and r.checks == [] and r.answer == "42"
    s2 = RawS2([{"content": "x" * 5, "tokens": 8}, {"content": "y" * 5, "tokens": 8}, {"content": "ans", "tokens": 1, "stop_type": "eos"}])
    r = GuidedGenerator(s2, SystemOneEngine(s1, model_name="mock"), check_every=8, max_think=16).think_adaptive([{"role": "user", "content": "q"}])
    assert r.stopped_by == "budget" and r.think_tokens == 16 and r.answer == "ans" and len(r.checks) == 1


def test_best_of_n_and_verified():
    s1 = ScriptedBackend([{"c0": [0.2, 0.8], "c1": [0.9, 0.1], "c2": [0.4, 0.6]}])
    s2 = RawS2([{"content": "A", "tokens": 1, "stop_type": "eos"}])
    g = GuidedGenerator(s2, SystemOneEngine(s1, model_name="mock"))
    r = g.best_of_n([{"role": "user", "content": "q"}], n=3)
    assert r["best"] == 1 and r["scores"] == [0.2, 0.9, 0.4] and len(r["candidates"]) == 3
    # verified : 1re verification basse -> 2e tentative avec plus de budget, gardee si meilleure
    s1 = ScriptedBackend([{"determined": [0.9, 0.1], "stuck": [0.1, 0.9]}, {"ok": [0.3, 0.7]},
                          {"determined": [0.9, 0.1], "stuck": [0.1, 0.9]}, {"ok": [0.85, 0.15]}])
    s2 = RawS2([{"content": "r", "tokens": 4}, {"content": "wrong", "tokens": 1, "stop_type": "eos"},
                {"content": "r2", "tokens": 4}, {"content": "right", "tokens": 1, "stop_type": "eos"}])
    g = GuidedGenerator(s2, SystemOneEngine(s1, model_name="mock"), check_every=4, max_think=8, stop_conf=0.8)
    r = g.verified([{"role": "user", "content": "q"}], threshold=0.7)
    assert r["attempts"] == 2 and r["answer"] == "right" and r["verification"] == 0.85


URL = os.environ.get("JEV_TEST_SERVER")


@pytest.mark.skipif(not URL, reason="JEV_TEST_SERVER non defini")
def test_live_guided_mechanics():
    from jev_clone.backend_llamacpp import LlamaCppBackend
    be = LlamaCppBackend(URL, max_workers=2)
    tpl = be.apply_template([{"role": "user", "content": "Solve 2+2"}])
    assert tpl.endswith("<|im_start|>assistant\n")
    out = be.complete(tpl + "<think>\n", n_predict=6, stop=["</think>"])
    assert out["stop_type"] in ("limit", "word", "eos") and out["tokens"] <= 6
    g = GuidedGenerator(be, SystemOneEngine(be), check_every=6, max_think=12, answer_tokens=4, stop_conf=1.01)  # jamais "determined"
    r = g.think_adaptive([{"role": "user", "content": "Solve 2+2"}])
    assert r.think_tokens <= 12 and r.stopped_by in ("budget", "model") and len(r.checks) >= 1

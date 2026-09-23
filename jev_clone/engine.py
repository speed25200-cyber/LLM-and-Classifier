"""SystemOneEngine : requete typee -> reponses typees, sans generation."""

from __future__ import annotations

import random
import time
from typing import Any

from jev_clone.prompt import Branch, PromptFormat, build_branches
from jev_clone.readout import Calibration, merge_branches, to_answer
from jev_clone.schema import SystemOneRequest, SystemOneResponse, Usage


class SystemOneEngine:
    def __init__(self, backend, fmt: PromptFormat | None = None, calibration: Calibration | None = None,
                 model_name: str | None = None):
        self.backend = backend
        self.fmt = fmt or PromptFormat()
        self.cal = calibration or Calibration()
        self.model_name = model_name if model_name is not None else getattr(backend, "model_name", lambda: "")()

    def branches(self, req: SystemOneRequest, seed: int = 0) -> list[Branch]:
        rng = random.Random(seed)
        out: list[Branch] = []
        for qid, q in req.questions.items():
            out.extend(build_branches(qid, q, self.fmt, req.permutations, rng, label_mode=req.label_mode))
        return out

    def answer(self, req: SystemOneRequest | dict[str, Any]) -> SystemOneResponse:
        if isinstance(req, dict):
            req = SystemOneRequest.model_validate(req)
        t0 = time.perf_counter()
        prefix = self.fmt.prefix(req.state)
        branches = self.branches(req)
        results = self.backend.score_branches(prefix, branches)

        per_q: dict[str, list[tuple[Branch, Any]]] = {}
        for br, res in zip(branches, results):
            per_q.setdefault(br.qid, []).append((br, res.logits))

        answers = {}
        for qid, q in req.questions.items():
            kind = q.type
            # temperature de la question calibree (meme question, etat de meme forme), T = 1 pour toute autre
            key_probs = merge_branches(kind, per_q[qid], self.cal, qid, q, req.state)
            answers[qid] = to_answer(kind, key_probs, q)

        prompt_tokens = sum(r.prompt_tokens for r in results)
        cached = sum(r.cached_tokens for r in results)
        usage = Usage(input_tokens=prompt_tokens + cached, state_tokens=0, question_tokens=prompt_tokens,
                      branches=len(branches), state_cache_hit=bool(results and results[0].cached_tokens > 0))
        return SystemOneResponse(answers=answers, usage=usage, model=self.model_name,
                                 latency_ms=round((time.perf_counter() - t0) * 1000.0, 2))

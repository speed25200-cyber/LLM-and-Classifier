"""Routeur de fusion Jev-clone (System One) x Bonsai (System Two).

    etat -> [S1: decisions typees + confiance] -> porte (seuil par question)
              |-- confiance >= seuil  -> on agit avec la decision S1        (~50-300 ms)
              |-- sinon               -> escalade vers Bonsai (reflexion + generation), budget selon le risque
                                          -> option : S1 verifie la sortie de Bonsai (garde-fou)
    tout est journalise (ledger JSONL) pour re-calibrer / re-entrainer S1 (boucle de distillation).

La porte et la calibration (calibrate.py) utilisent la MEME statistique : `gate_statistic`, la probabilite de
l'option retenue apres temperature (max(p, 1-p) pour un noul). Le champ `confidence` des reponses (1 - entropie
normalisee, contrat TypeSafe) reste renvoye mais ne sert pas a la porte.
Les reponses de Bonsai sont typees question par question ; une reponse absente ou invalide, ou un Bonsai
injoignable, laisse la valeur de S1 marquee `s1_fallback` (liste `unresolved`) : jamais en silence, jamais une 500.

Le routeur ne contient aucune logique metier : "le code calcule, Jev juge, le LLM raisonne".
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from jev_clone.readout import Calibration
from jev_clone.schema import SystemOneRequest, SystemOneResponse

TRUE_WORDS = {"yes", "y", "true", "oui", "vrai", "1"}
FALSE_WORDS = {"no", "n", "false", "non", "faux", "0"}


@dataclass
class GatePolicy:
    default_threshold: float = 0.85          # probabilite de l'option retenue (gate_statistic) minimale pour agir seul
    thresholds: dict[str, float | None] = field(default_factory=dict)   # par question, issus de calibrate.py (None = toujours escalader)
    risk_question: str | None = None         # qid d'une question 'score' de risque (0 = benin) ; module le budget de reflexion
    budgets: tuple[int, ...] = (0, 512, 2048, 8192)   # budget de reflexion Bonsai par niveau de risque
    always_escalate_if: dict[str, Any] = field(default_factory=dict)  # ex. {"needs_reasoning": True}
    verify_with_s1: bool = False             # S1 relit la reponse de Bonsai (question noul de coherence)

    @classmethod
    def from_calibration(cls, cal: Calibration, **kw) -> "GatePolicy":
        return cls(thresholds=dict(cal.thresholds), **kw)

    def threshold(self, qid: str) -> float:
        if qid not in self.thresholds:
            return float(self.default_threshold)
        v = self.thresholds[qid]
        return math.inf if v is None else float(v)   # aucun seuil fiable a la calibration : toujours escalader


def gate_statistic(probs) -> float:
    """Statistique de la porte, identique a la calibration : probabilite de l'option retenue."""
    return float(np.max(np.asarray(probs, dtype=np.float64)))


def answer_confidence(a) -> float:
    if a.type == "noul":
        return gate_statistic([a.noul, 1.0 - a.noul])
    return gate_statistic(list(a.probabilities.values()))


def answer_value(a):
    if a.type == "noul":
        return a.noul >= 0.5
    if a.type == "choice":
        return a.choice
    return a.score


def coerce_answer(q, v) -> tuple[bool, Any]:
    """Reponse de Bonsai pour une question typee -> (valide, valeur du meme type que answer_value)."""
    if q.type == "noul":
        if isinstance(v, bool):
            return True, v
        if isinstance(v, (int, float)) and v in (0, 1):
            return True, bool(v)
        if isinstance(v, str):
            w = v.strip().strip(".!").lower()
            if w in TRUE_WORDS or w in FALSE_WORDS:
                return True, w in TRUE_WORDS
        return False, None
    if q.type == "choice":
        keys = [str(k) for k in (q.criteria.keys() if isinstance(q.criteria, dict) else q.criteria)]
        if isinstance(v, str):
            w = v.strip()
            if w in keys:
                return True, w
            low = {k.lower(): k for k in keys}
            if w.lower() in low:
                return True, low[w.lower()]
        return False, None
    from jev_clone.prompt import render_text
    n = len(q.criteria)
    if isinstance(v, str):
        w = v.strip()
        texts = {render_text(c).strip().lower(): i for i, c in enumerate(q.criteria)}
        if w.lower() in texts:
            return True, float(texts[w.lower()])
        try:
            v = float(w)
        except ValueError:
            return False, None
    if isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v <= n - 1:
        return True, float(v)
    return False, None


@dataclass
class FusionResult:
    path: str                                  # "system_one" | "escalated" | "fallback" (escalade impossible : Bonsai en echec)
    s1: SystemOneResponse
    decisions: dict[str, Any]                  # valeur retenue par question
    gated: dict[str, bool]                     # True si S1 a suffi pour cette question
    s2: dict | None = None                     # reponse brute de Bonsai (chat completion) si escalade
    s2_text: str | None = None
    s2_reasoning: str | None = None
    verification: dict | None = None
    latency_ms: float = 0.0
    sources: dict[str, str] = field(default_factory=dict)   # "s1" | "s2" | "s1_fallback" (S1 sous le seuil, pas de reponse valide de Bonsai)
    unresolved: list[str] = field(default_factory=list)     # questions restees sur une valeur S1 sous le seuil
    s2_error: str | None = None


class FusionRouter:
    def __init__(self, s1_engine, s2_backend=None, policy: GatePolicy | None = None,
                 ledger: str | Path | None = None,
                 escalation_prompt: Callable[[Any, SystemOneRequest, SystemOneResponse], list[dict]] | None = None):
        self.s1 = s1_engine
        self.s2 = s2_backend
        self.policy = policy or GatePolicy()
        self.ledger = Path(ledger) if ledger else None
        self.escalation_prompt = escalation_prompt or self.default_escalation_prompt

    @staticmethod
    def default_escalation_prompt(state, req: SystemOneRequest, s1: SystemOneResponse) -> list[dict]:
        from jev_clone.prompt import render_text

        def expected(q) -> str:
            if q.type == "noul":
                return "answer true or false"
            if q.type == "choice":
                return f"answer one of {list(q.criteria.keys()) if isinstance(q.criteria, dict) else list(q.criteria)}"
            return "answer the level number: " + ", ".join(f"{i} = {render_text(c)}" for i, c in enumerate(q.criteria))
        low = {qid: (answer_value(a), round(answer_confidence(a), 3)) for qid, a in s1.answers.items()}
        return [
            {"role": "system", "content": "You are the reasoning tier of a decision system. A fast decision model "
                                          "was not confident enough. Reason carefully about the state, then answer "
                                          "each question and justify briefly. End with a JSON object of final answers, "
                                          "one key per question id."},
            {"role": "user", "content": f"# State\n{render_text(state)}\n\n# Questions\n"
                                        + "\n".join(f"- {qid}: {render_text(q.instructions)} ({expected(q)})" for qid, q in req.questions.items())
                                        + f"\n\n# Fast-model tentative answers (value, confidence)\n{json.dumps(low)}"},
        ]

    def decide(self, req: SystemOneRequest | dict, state_for_s2=None) -> FusionResult:
        if isinstance(req, dict):
            req = SystemOneRequest.model_validate(req)
        t0 = time.perf_counter()
        s1 = self.s1.answer(req)
        gated, decisions = {}, {}
        for qid, a in s1.answers.items():
            ok = answer_confidence(a) >= self.policy.threshold(qid)
            forced = self.policy.always_escalate_if.get(qid)
            if forced is not None and answer_value(a) == forced:
                ok = False
            gated[qid] = ok
            decisions[qid] = answer_value(a)

        res = FusionResult(path="system_one", s1=s1, decisions=decisions, gated=gated,
                           sources={qid: "s1" if ok else "s1_fallback" for qid, ok in gated.items()})
        res.unresolved = [qid for qid, ok in gated.items() if not ok]
        if all(gated.values()) or self.s2 is None:
            return self._done(req, res, t0)

        # --- escalade vers Bonsai ------------------------------------------------------------
        risk_level = 0
        if self.policy.risk_question and self.policy.risk_question in s1.answers:
            ra = s1.answers[self.policy.risk_question]
            if ra.type == "score":
                risk_level = int(round(ra.score))
        budget = self.policy.budgets[min(risk_level, len(self.policy.budgets) - 1)]
        messages = self.escalation_prompt(state_for_s2 if state_for_s2 is not None else req.state, req, s1)
        try:
            s2 = self.s2.chat(messages, max_tokens=1024, thinking_budget=budget, temperature=0.2)
            msg = s2["choices"][0]["message"]
        except Exception as e:   # Bonsai injoignable ou reponse illisible : on garde S1, marque comme repli
            res.path, res.s2_error = "fallback", f"{type(e).__name__}: {str(e)[:300]}"
            return self._done(req, res, t0)
        res.path, res.s2 = "escalated", s2
        res.s2_text = msg.get("content")
        res.s2_reasoning = msg.get("reasoning_content")
        final = _extract_json(res.s2_text or "")
        for qid, ok in gated.items():
            if ok or qid not in final:
                continue
            valid, v = coerce_answer(req.questions[qid], final[qid])
            if valid:
                decisions[qid], res.sources[qid] = v, "s2"
        res.unresolved = [qid for qid, src in res.sources.items() if src == "s1_fallback"]
        if self.policy.verify_with_s1 and res.s2_text:
            vq = {"consistent": {"type": "noul",
                                 "instructions": "Is the ANSWER below consistent with the STATE and does it follow the instructions?"}}
            vreq = SystemOneRequest(state={"STATE": req.state, "ANSWER": res.s2_text}, questions=vq)
            res.verification = self.s1.answer(vreq).answers["consistent"].model_dump()
        return self._done(req, res, t0)

    def _done(self, req: SystemOneRequest, res: FusionResult, t0: float) -> FusionResult:
        res.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        self._log(req, res)
        return res

    def _log(self, req: SystemOneRequest, res: FusionResult) -> None:
        if not self.ledger:
            return
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        rec = {"ts": time.time(), "path": res.path, "state": req.state,
               "questions": {k: v.model_dump() for k, v in req.questions.items()},
               "s1": {k: v.model_dump() for k, v in res.s1.answers.items()},
               "gated": res.gated, "decisions": res.decisions, "sources": res.sources, "unresolved": res.unresolved,
               "s2_text": res.s2_text, "s2_error": res.s2_error, "verification": res.verification, "latency_ms": res.latency_ms}
        with open(self.ledger, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _extract_json(text: str) -> dict:
    """Dernier objet JSON present dans le texte (Bonsai termine par un objet de reponses)."""
    end = text.rfind("}")
    while end != -1:
        start = text.rfind("{", 0, end)
        while start != -1:
            try:
                d = json.loads(text[start:end + 1])
                if isinstance(d, dict):
                    return d
            except json.JSONDecodeError:
                pass
            start = text.rfind("{", 0, start)
        end = text.rfind("}", 0, end)
    return {}

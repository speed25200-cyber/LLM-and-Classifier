"""Generation guidee par System One : le clone de Jev pilote la reflexion et la sortie de Bonsai.

C'est la fusion au niveau de l'inference, pas seulement un routeur devant deux modeles :

  * think_adaptive : Bonsai reflechit par morceaux (N tokens) ; a chaque point de controle le clone juge en
    ~100 ms "la reponse est-elle deja determinee ?" et "la reflexion tourne-t-elle en rond ?". Si oui, on
    ferme </think> et on fait repondre. Sur une RTX 4060 (~30 tok/s), 2 048 tokens de reflexion coutent ~70 s :
    c'est le poste de latence numero un, et c'est le clone qui le reduit, requete par requete.
  * best_of_n : N reponses courtes echantillonnees en parallele (slots du serveur), classees par le clone
    (un noul par candidat, une passe) ; on garde la meilleure. Qualite de "vote" au prix d'une seule reflexion.
  * verified : reponse -> verification par le clone -> une seconde tentative avec plus de budget si elle echoue.

Tout passe par l'API de llama-server (apply-template + completion brute avec cache de prefixe), donc cela
marche avec Bonsai 2 sur le fork PrismML comme avec n'importe quel GGUF.
"""

from __future__ import annotations

import concurrent.futures as cf
import time
from dataclasses import dataclass, field

from jev_clone.prompt import render_text

THINK_OPEN, THINK_CLOSE = "<think>\n", "\n</think>\n\n"
DETERMINED_Q = ("Given the task and the reasoning so far, is the final answer already determined, so that further "
                "reasoning would not change it?")
STUCK_Q = "Is the reasoning stuck: repeating itself, going in circles, or drifting away from the task?"
CORRECT_Q = "Is this candidate answer correct, complete and consistent with the task?"


@dataclass
class ThinkResult:
    reasoning: str
    answer: str
    think_tokens: int
    answer_tokens: int
    checks: list[dict] = field(default_factory=list)
    stopped_by: str = ""          # "model" (</think> emis) | "determined" | "stuck" | "budget"
    latency_ms: float = 0.0


class GuidedGenerator:
    def __init__(self, s2_backend, s1_engine, check_every: int = 192, max_think: int = 4096,
                 stop_conf: float = 0.8, answer_tokens: int = 768, temperature: float = 0.7,
                 end_tokens: tuple[str, ...] = ("<|im_end|>",)):
        self.s2, self.s1 = s2_backend, s1_engine
        self.check_every, self.max_think, self.stop_conf = check_every, max_think, stop_conf
        self.answer_tokens, self.temperature, self.end_tokens = answer_tokens, temperature, list(end_tokens)

    # ---- reflexion adaptative -----------------------------------------------------------------------
    def think_adaptive(self, messages: list[dict], task_summary: str | None = None) -> ThinkResult:
        t0 = time.perf_counter()
        base = self.s2.apply_template(messages)
        task = task_summary or render_text(messages[-1].get("content", ""))
        prompt = base + THINK_OPEN
        reasoning, think_tokens, checks, stopped_by = "", 0, [], "budget"
        while think_tokens < self.max_think:
            n = min(self.check_every, self.max_think - think_tokens)
            out = self.s2.complete(prompt + reasoning, n_predict=n, stop=["</think>"], temperature=self.temperature)
            reasoning += out["content"]; think_tokens += out["tokens"]
            if out["stop_type"] == "word" or out["stop_type"] == "eos":
                stopped_by = "model"; break
            if think_tokens >= self.max_think:
                break  # budget epuise : inutile de consulter S1
            st = {"task": task, "reasoning_so_far": reasoning[-6000:]}
            r = self.s1.answer({"state": st, "questions": {"determined": {"type": "noul", "instructions": DETERMINED_Q},
                                                            "stuck": {"type": "noul", "instructions": STUCK_Q}}})
            p_det, p_stuck = r.answers["determined"].noul, r.answers["stuck"].noul
            checks.append({"think_tokens": think_tokens, "determined": round(p_det, 3), "stuck": round(p_stuck, 3), "ms": r.latency_ms})
            if p_det >= self.stop_conf:
                stopped_by = "determined"; break
            if p_stuck >= self.stop_conf:
                stopped_by = "stuck"; break
        ans = self.s2.complete(prompt + reasoning + THINK_CLOSE, n_predict=self.answer_tokens, stop=self.end_tokens,
                               temperature=self.temperature)
        return ThinkResult(reasoning=reasoning, answer=ans["content"].strip(), think_tokens=think_tokens,
                           answer_tokens=ans["tokens"], checks=checks, stopped_by=stopped_by,
                           latency_ms=round((time.perf_counter() - t0) * 1000, 1))

    # ---- meilleur de N ---------------------------------------------------------------------------------
    def best_of_n(self, messages: list[dict], n: int = 4, task_summary: str | None = None, think: bool = False,
                  max_tokens: int = 512, question: str = CORRECT_Q) -> dict:
        t0 = time.perf_counter()
        base = self.s2.apply_template(messages)
        prompt = base + ("" if think else THINK_OPEN + THINK_CLOSE)
        task = task_summary or render_text(messages[-1].get("content", ""))
        with cf.ThreadPoolExecutor(max_workers=n) as ex:
            outs = list(ex.map(lambda _: self.s2.complete(prompt, n_predict=max_tokens, stop=self.end_tokens,
                                                          temperature=max(self.temperature, 0.5)), range(n)))
        cands = [o["content"].strip() for o in outs]
        qs = {f"c{i}": {"type": "noul", "instructions": f"{question}\nCandidate answer:\n{c[:3000]}"} for i, c in enumerate(cands)}
        r = self.s1.answer({"state": {"task": task}, "questions": qs})
        scores = [r.answers[f"c{i}"].noul for i in range(n)]
        best = max(range(n), key=lambda i: scores[i])
        return {"answer": cands[best], "candidates": cands, "scores": [round(s, 3) for s in scores], "best": best,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}

    # ---- reponse verifiee ------------------------------------------------------------------------------
    def verified(self, messages: list[dict], task_summary: str | None = None, threshold: float = 0.7,
                 retry_max_think: int | None = None) -> dict:
        first = self.think_adaptive(messages, task_summary)
        task = task_summary or render_text(messages[-1].get("content", ""))
        v = self.s1.answer({"state": {"task": task, "answer": first.answer},
                            "questions": {"ok": {"type": "noul", "instructions": CORRECT_Q}}}).answers["ok"].noul
        out = {"answer": first.answer, "verification": round(v, 3), "attempts": 1, "first": first}
        if v < threshold:
            saved = (self.max_think, self.stop_conf)
            self.max_think = retry_max_think or self.max_think * 2
            self.stop_conf = min(0.95, self.stop_conf + 0.1)
            try:
                second = self.think_adaptive(messages + [{"role": "user", "content": "Your previous answer may be wrong. Re-check carefully and answer again."}], task_summary)
            finally:
                self.max_think, self.stop_conf = saved
            v2 = self.s1.answer({"state": {"task": task, "answer": second.answer},
                                 "questions": {"ok": {"type": "noul", "instructions": CORRECT_Q}}}).answers["ok"].noul
            if v2 >= v:
                out.update({"answer": second.answer, "verification": round(v2, 3), "second": second})
            out["attempts"] = 2
        return out

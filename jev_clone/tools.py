"""Le sens inverse de la fusion : Bonsai (System Two) consulte le clone de Jev (System One) comme un outil.

Pendant qu'il raisonne, planifie ou pilote l'ordinateur, Bonsai peut demander en ~50-150 ms :
  * judge_choice : "laquelle de ces options ?"           -> option + probabilites + confiance
  * judge_noul   : "est-ce vrai ?"                         -> probabilite
  * judge_score  : "a quel niveau ?"                       -> niveau pondere + probabilites
  * judge_rank   : "classe ces N candidats" (N nouls en parallele, une seule passe partagee)
  * judge_batch  : plusieurs questions typees d'un coup (format /v1/systemone)

`AgentLoop` execute la boucle d'appels d'outils OpenAI de llama-server (`--jinja`) : Bonsai emet des
`tool_calls`, on les execute (outils du clone + outils applicatifs), on renvoie les resultats, jusqu'a
la reponse finale ou `max_turns`. Toute la trajectoire est journalisee.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from jev_clone.schema import SystemOneRequest


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required, "additionalProperties": False}}}


STATE_PROP = {"type": "string", "description": "The state to judge: text or a JSON string (ticket, page, log, plan step...)."}
QUESTION_PROP = {"type": "string", "description": "The question, phrased so that an expert could answer in five seconds."}

SYSTEMONE_TOOL_DEFS = [
    _tool("judge_choice", "Fast calibrated decision (System One, no reasoning, ~100 ms): pick ONE option for the state. "
          "Returns the choice, a probability per option and a confidence in [0,1]. Use it for classification, routing, "
          "selecting among candidates, deciding which step comes next.",
          {"state": STATE_PROP, "question": QUESTION_PROP,
           "options": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 26,
                       "description": "2 to 26 options; add 'other / none of the above' when the list may be incomplete."}},
          ["state", "question", "options"]),
    _tool("judge_noul", "Fast calibrated yes/no judgement (System One, ~100 ms). Returns the probability that the answer is yes. "
          "Use it to check a fact about the state, verify a step, detect a property, decide whether to escalate.",
          {"state": STATE_PROP, "question": QUESTION_PROP}, ["state", "question"]),
    _tool("judge_score", "Fast calibrated rating (System One, ~100 ms) on an ordered scale you describe (2 to 10 levels, low to high). "
          "Returns the expected level (may fall between levels) and a probability per level.",
          {"state": STATE_PROP, "question": QUESTION_PROP,
           "levels": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 10}},
          ["state", "question", "levels"]),
    _tool("judge_rank", "Rank up to 50 candidates for the state in ONE fast pass (a yes/no judgement per candidate, evaluated in "
          "parallel and independently). Returns candidates sorted by probability. Use it to pick the UI element to act on, "
          "the document to read, the tool to call, the hypothesis to test.",
          {"state": STATE_PROP, "question": {"type": "string", "description": "Yes/no question asked about EACH candidate, e.g. 'Is this the element to click next?'"},
           "candidates": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 50}},
          ["state", "question", "candidates"]),
    _tool("judge_batch", "Several typed questions about one state in one fast pass. `questions` is a JSON object in the "
          "TypeSafe /v1/systemone format: {id: {type: choice|noul|score, instructions, criteria}}.",
          {"state": STATE_PROP, "questions": {"type": "string", "description": "JSON string of the questions object."}},
          ["state", "questions"]),
]


class SystemOneToolbox:
    """Expose un SystemOneEngine comme jeu d'outils OpenAI + executeur."""

    def __init__(self, engine, max_rank: int = 50):
        self.engine = engine
        self.max_rank = max_rank

    def definitions(self) -> list[dict]:
        return list(SYSTEMONE_TOOL_DEFS)

    @staticmethod
    def _state(s):
        if isinstance(s, str):
            try:
                return json.loads(s)
            except Exception:
                return s
        return s

    def call(self, name: str, args: dict) -> dict:
        st = self._state(args.get("state", ""))
        if name == "judge_choice":
            r = self.engine.answer({"state": st, "questions": {"q": {"type": "choice", "instructions": args["question"], "criteria": list(args["options"])}}})
            return r.answers["q"].model_dump()
        if name == "judge_noul":
            r = self.engine.answer({"state": st, "questions": {"q": {"type": "noul", "instructions": args["question"]}}})
            return r.answers["q"].model_dump()
        if name == "judge_score":
            r = self.engine.answer({"state": st, "questions": {"q": {"type": "score", "instructions": args["question"], "criteria": list(args["levels"])}}})
            return r.answers["q"].model_dump()
        if name == "judge_rank":
            cands = list(args["candidates"])[: self.max_rank]
            qs = {f"c{i}": {"type": "noul", "instructions": f"{args['question']}\nCandidate: {c}"} for i, c in enumerate(cands)}
            r = self.engine.answer({"state": st, "questions": qs})
            ranked = sorted(((r.answers[f"c{i}"].noul, c) for i, c in enumerate(cands)), reverse=True)
            return {"ranked": [{"candidate": c, "probability": round(p, 4)} for p, c in ranked]}
        if name == "judge_batch":
            qs = args["questions"] if isinstance(args["questions"], dict) else json.loads(args["questions"])
            r = self.engine.answer(SystemOneRequest(state=st, questions=qs))
            return {k: v.model_dump() for k, v in r.answers.items()}
        raise KeyError(name)


@dataclass
class AgentStep:
    turn: int
    tool_calls: list[dict] = field(default_factory=list)
    content: str | None = None
    reasoning: str | None = None
    ms: float = 0.0


@dataclass
class AgentResult:
    content: str | None
    steps: list[AgentStep]
    messages: list[dict]
    stopped_by: str  # "final" | "max_turns" | "stop_tool"


class AgentLoop:
    """Boucle d'agent : Bonsai (chat + tools) appelle le clone et les outils applicatifs.

    extra_tools : {name: (definition_openai, fonction(args) -> resultat JSON-serialisable)}.
    Les outils dont la fonction renvoie {"__stop__": True, ...} terminent la boucle (ex. `done`)."""

    def __init__(self, s2_backend, toolbox: SystemOneToolbox | None = None,
                 extra_tools: dict[str, tuple[dict, Callable[[dict], Any]]] | None = None,
                 max_turns: int = 10, thinking_budget: int | None = 1024, max_tokens: int = 1024,
                 ledger: str | Path | None = None):
        self.s2 = s2_backend
        self.toolbox = toolbox
        self.extra = extra_tools or {}
        self.max_turns = max_turns
        self.thinking_budget = thinking_budget
        self.max_tokens = max_tokens
        self.ledger = Path(ledger) if ledger else None

    def tool_definitions(self) -> list[dict]:
        defs = self.toolbox.definitions() if self.toolbox else []
        return defs + [d for d, _ in self.extra.values()]

    def execute(self, name: str, args: dict) -> Any:
        if name in self.extra:
            return self.extra[name][1](args)
        if self.toolbox and name.startswith("judge_"):
            return self.toolbox.call(name, args)
        return {"error": f"unknown tool {name}"}

    def run(self, messages: list[dict], thinking_budget: int | None = None) -> AgentResult:
        msgs = list(messages)
        steps: list[AgentStep] = []
        budget = self.thinking_budget if thinking_budget is None else thinking_budget
        stopped = "max_turns"
        for turn in range(self.max_turns):
            t0 = time.perf_counter()
            try:
                resp = self.s2.chat(msgs, max_tokens=self.max_tokens, thinking_budget=budget, temperature=0.2,
                                    tools=self.tool_definitions())
            except Exception as e:  # serveur indisponible, contexte depasse malgre les reessais... : on rend la main proprement
                steps.append(AgentStep(turn=turn, content=f"[erreur du modele de raisonnement : {str(e)[:200]}]",
                                       ms=round((time.perf_counter() - t0) * 1000, 1)))
                stopped = "error"; break
            msg = resp["choices"][0]["message"]
            step = AgentStep(turn=turn, content=msg.get("content"), reasoning=msg.get("reasoning_content"))
            calls = msg.get("tool_calls") or []
            assistant = {"role": "assistant", "content": msg.get("content") or ""}
            if calls:
                assistant["tool_calls"] = calls
            msgs.append(assistant)
            stop = False
            for tc in calls:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                try:
                    args = json.loads(fn.get("arguments") or "{}") if isinstance(fn.get("arguments"), str) else (fn.get("arguments") or {})
                except json.JSONDecodeError:
                    args = {}
                try:
                    result = self.execute(name, args)
                except Exception as e:  # l'outil a echoue : on le dit au modele plutot que de planter la boucle
                    result = {"error": str(e)}
                if isinstance(result, dict) and result.get("__stop__"):
                    stop = True
                step.tool_calls.append({"name": name, "args": args, "result": result})
                msgs.append({"role": "tool", "tool_call_id": tc.get("id", f"call_{turn}"), "name": name,
                             "content": json.dumps(result, ensure_ascii=False, default=str)})
            step.ms = round((time.perf_counter() - t0) * 1000, 1)
            steps.append(step)
            if stop:
                stopped = "stop_tool"; break
            if not calls:
                stopped = "final"; break
        res = AgentResult(content=steps[-1].content if steps else None, steps=steps, messages=msgs, stopped_by=stopped)
        self._log(res)
        return res

    def _log(self, res: AgentResult) -> None:
        if not self.ledger:
            return
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ledger, "a") as f:
            f.write(json.dumps({"ts": time.time(), "stopped_by": res.stopped_by, "content": res.content,
                                "steps": [{"turn": s.turn, "tool_calls": s.tool_calls, "ms": s.ms} for s in res.steps]},
                               ensure_ascii=False, default=str) + "\n")

"""Contrat requete/reponse, aligne sur le format public de TypeSafe (POST /v1/systemone)
tel que reproduit par les clones ouverts (decider, reflex). Trois primitives :

  noul   : proposition oui/non      -> probabilite de "oui"
  choice : une option parmi N       -> option gagnante + probabilites + confidence
  score  : niveau sur une echelle   -> score pondere + probabilites par niveau + confidence

`state` est une chaine ou n'importe quel JSON. Les questions sont evaluees independamment.
"""

from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel, Field, field_validator, model_validator

Text = Union[str, dict, list, int, float, bool, None]

MAX_CHOICE_OPTIONS = 26   # etiquettes A..Z (un token chacune) ; Jev accepte 255
MAX_SCORE_LEVELS = 10
MIN_SCORE_LEVELS = 2


class NoulCriteria(BaseModel):
    true: Text = None
    false: Text = None


class NoulQuestion(BaseModel):
    type: Literal["noul"]
    instructions: Text
    criteria: NoulCriteria | None = None


class ChoiceQuestion(BaseModel):
    type: Literal["choice"]
    instructions: Text
    # dict option -> description (ou None), ou liste d'options sans description
    criteria: dict[str, Text] | list[str]

    @field_validator("criteria")
    @classmethod
    def _check(cls, v):
        keys = list(v.keys()) if isinstance(v, dict) else list(v)
        if len(keys) < 2:
            raise ValueError("choice: au moins 2 options")
        if len(keys) > MAX_CHOICE_OPTIONS:
            raise ValueError(f"choice: au plus {MAX_CHOICE_OPTIONS} options dans ce clone")
        if len(set(keys)) != len(keys):
            raise ValueError("choice: options dupliquees")
        return v

    def options(self) -> dict[str, Text]:
        if isinstance(self.criteria, dict):
            return dict(self.criteria)
        return {k: None for k in self.criteria}


class ScoreQuestion(BaseModel):
    type: Literal["score"]
    instructions: Text
    criteria: list[Text]  # niveaux ordonnes du plus bas au plus haut

    @field_validator("criteria")
    @classmethod
    def _check(cls, v):
        if not (MIN_SCORE_LEVELS <= len(v) <= MAX_SCORE_LEVELS):
            raise ValueError(f"score: entre {MIN_SCORE_LEVELS} et {MAX_SCORE_LEVELS} niveaux")
        return v


Question = Union[NoulQuestion, ChoiceQuestion, ScoreQuestion]


class SystemOneRequest(BaseModel):
    state: Text
    questions: dict[str, Question]
    permutations: int = Field(default=1, ge=1, le=8)  # extension : moyenne sur N ordres d'options

    @model_validator(mode="after")
    def _nonempty(self):
        if not self.questions:
            raise ValueError("au moins une question")
        return self


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    probabilities: dict[str, float]
    legend: dict[str, str]
    confidence: float


Answer = Union[NoulAnswer, ChoiceAnswer, ScoreAnswer]


class Usage(BaseModel):
    input_tokens: int = 0
    state_tokens: int = 0
    question_tokens: int = 0
    branches: int = 0
    state_cache_hit: bool = False


class SystemOneResponse(BaseModel):
    answers: dict[str, Answer]
    usage: Usage = Field(default_factory=Usage)
    model: str = ""
    latency_ms: float = 0.0


def parse_request(payload: dict[str, Any]) -> SystemOneRequest:
    return SystemOneRequest.model_validate(payload)

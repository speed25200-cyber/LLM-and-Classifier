"""Des probabilites d'etiquettes par branche aux reponses typees.

Deux reglages post-hoc rendent la lecture d'un LM pre-entraine "calibree" :
  * une temperature par question (ajustee sur des donnees etiquetees, voir calibrate.py), appliquee a cette seule
    question : meme nom, meme consigne et memes options, lue sur un etat de meme forme. Toute autre question (garde-fou,
    verification, voix, computer use) reste lue brute (T = 1) tant qu'elle n'a pas ses propres donnees etiquetees.
  * une statistique de `confidence` (TypeSafe : "a quel point la distribution est piquee")
Compatibilite : un ancien fichier (temperature par primitive + seuils) ne s'applique qu'aux questions de ses seuils ;
une calibration sans aucune question (entrainement, jev serve) garde une temperature par primitive pour tout.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import TypeAdapter

from jev_clone.prompt import Branch, render_text
from jev_clone.schema import ChoiceAnswer, NoulAnswer, Question, ScoreAnswer, ScoreQuestion


KINDS = ("noul", "choice", "score")
LEGACY_STATE = ("request",)   # forme des etats des graines de Prophet (anciens fichiers : questions du pre-tour)
_QUESTION = TypeAdapter(Question)


@functools.lru_cache(maxsize=512)
def _fingerprint(canon: str) -> str:
    return hashlib.sha1(canon.encode("utf-8")).hexdigest()[:16]


def question_fingerprint(q) -> str:
    """Empreinte d'une question (type, consigne, options) : l'`intent` de la voix n'est pas celui de Prophet."""
    if isinstance(q, dict):
        q = _QUESTION.validate_python(q)
    d = q.model_dump(mode="json")
    return _fingerprint(json.dumps([d.get("type"), d.get("instructions"), d.get("criteria")], sort_keys=True, ensure_ascii=False))


def state_keys(state) -> list[str] | None:
    return sorted(str(k) for k in state) if isinstance(state, dict) else None


@functools.lru_cache(maxsize=1)
def _prophet_turn() -> dict:
    from jev_clone.prophet import PROPHET_TURN   # import tardif : prophet importe ce module
    return PROPHET_TURN


@dataclass
class Calibration:
    # par primitive : ajustee sur tout l'ensemble etiquete (rapports, repli des questions a peu d'exemples) ; appliquee a
    # toutes les questions seulement par une calibration generique (ni `questions` ni `thresholds`)
    temperature: dict[str, float] = field(default_factory=lambda: {"noul": 1.0, "choice": 1.0, "score": 1.0})
    # seuil de la porte par question, sur la statistique de fusion.gate_statistic (probabilite de l'option retenue,
    # apres temperature) ; None = aucun seuil n'atteint la precision visee : toujours escalader
    thresholds: dict[str, float | None] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)      # modele, nombre d'exemples, ECE avant / apres, source...
    # temperature par question, appliquee a elle seule : {qid: {"T", "kind", "fp" (empreinte), "state" (cles d'etat), "n"}}
    questions: dict[str, dict] = field(default_factory=dict)
    _legacy: tuple | None = field(default=None, init=False, repr=False, compare=False)

    def t(self, kind: str) -> float:
        return float(self.temperature.get(kind, 1.0))

    def threshold(self, name: str, default: float) -> float:
        """Seuil calibre d'une question (`default` si elle n'a pas ete calibree) ; inf = ne jamais agir seul."""
        if name not in self.thresholds:
            return float(default)
        v = self.thresholds[name]
        return math.inf if v is None else float(v)

    @property
    def generic(self) -> bool:
        """Temperature par primitive sans question connue (entrainement, benchmark) : s'applique a tout."""
        return not self.questions and not self.thresholds

    def scope(self) -> dict[str, dict]:
        """Questions ajustees. Ancien format (sans `questions`) : celles des seuils, a la temperature de leur primitive ;
        les questions du pre-tour de Prophet y sont reconnues a leur empreinte et a la forme des graines."""
        if self.questions or not self.thresholds:
            return self.questions
        key = tuple(self.thresholds)
        if self._legacy is None or self._legacy[0] != key:
            turn = _prophet_turn()
            self._legacy = (key, {qid: ({"T": None, "kind": turn[qid]["type"], "fp": question_fingerprint(turn[qid]),
                                         "state": list(LEGACY_STATE)} if qid in turn else {"T": None}) for qid in key})
        return self._legacy[1]

    def entry(self, qid: str | None, q=None, state=None) -> dict | None:
        """Entree de calibration qui vaut pour cette question lue sur cet etat, sinon None (lecture brute)."""
        e = self.scope().get(qid) if qid is not None else None
        if e is None:
            return None
        keys = e.get("state")
        if keys is not None and not (isinstance(state, dict) and set(keys) <= {str(k) for k in state}):
            return None   # meme question lue sur un autre etat (le `risk` du garde-fou n'est pas celui du pre-tour)
        if e.get("fp") and (q is None or question_fingerprint(q) != e["fp"]):
            return None
        return e

    def t_for(self, qid: str | None, q=None, state=None, kind: str | None = None) -> float:
        """Temperature d'une question : celle ajustee sur elle, 1 ailleurs (calibration generique : par primitive)."""
        kind = kind or getattr(q, "type", None) or (q.get("type") if isinstance(q, dict) else None)
        if self.generic:
            return self.t(kind) if kind else 1.0
        e = self.entry(qid, q, state)
        if e is None:
            return 1.0
        return float(e["T"]) if e.get("T") is not None else self.t(e.get("kind") or kind or "")

    def gate(self, qid: str, q=None, state=None, default=None):
        """Seuil calibre d'une porte si cette question est calibree ici (inf = ne jamais agir seul), sinon `default`."""
        if qid not in self.thresholds or self.entry(qid, q, state) is None:
            return default
        return self.threshold(qid, 0.0)

    def is_active(self) -> bool:
        """Une temperature autre que 1 ou un seuil peut s'appliquer."""
        return (bool(self.thresholds) or any(abs(v - 1.0) > 1e-9 for v in self.temperature.values())
                or any(abs(float(e.get("T") or 1.0) - 1.0) > 1e-9 for e in self.questions.values()))

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        """Valide un contenu de calibration.json (fichier importe, par ex.) : ValueError s'il n'en est pas un."""
        if not isinstance(d, dict) or not isinstance(d.get("temperature"), dict):
            raise ValueError("pas une calibration : objet {temperature: {noul, choice, score}, thresholds: {...}} attendu")
        temp = {}
        for k, v in d["temperature"].items():
            if k not in KINDS:
                raise ValueError(f"temperature : primitive inconnue '{k}' (attendu : {', '.join(KINDS)})")
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not (0.01 <= float(v) <= 100.0):
                raise ValueError(f"temperature {k} invalide : {v!r} (nombre entre 0,01 et 100 attendu)")
            temp[k] = float(v)
        thr = d.get("thresholds") or {}
        if not isinstance(thr, dict):
            raise ValueError("thresholds : objet {question: seuil} attendu")
        for k, v in thr.items():
            if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not (0.0 <= float(v) <= 1.0)):
                raise ValueError(f"seuil de '{k}' invalide : {v!r} (nombre entre 0 et 1, ou null = toujours escalader)")
        qs = d.get("questions") or {}
        if not isinstance(qs, dict):
            raise ValueError("questions : objet {question: {T, kind, fp, state, n}} attendu")
        questions = {}
        for k, e in qs.items():
            T = e.get("T") if isinstance(e, dict) else None
            if isinstance(T, bool) or not isinstance(T, (int, float)) or not (0.01 <= float(T) <= 100.0):
                raise ValueError(f"temperature de la question '{k}' invalide : {T!r} (nombre entre 0,01 et 100 attendu)")
            if e.get("kind") not in KINDS or not isinstance(e.get("fp") or "", str) or not isinstance(e.get("state") or [], list):
                raise ValueError(f"question '{k}' : kind ({', '.join(KINDS)}), fp (texte) et state (liste) attendus")
            questions[str(k)] = {"T": float(T), "kind": e["kind"], "fp": e.get("fp"),
                                 "state": None if e.get("state") is None else [str(x) for x in e["state"]], "n": e.get("n")}
        meta = d.get("meta") if isinstance(d.get("meta"), dict) else {}
        return cls(temperature=temp, thresholds={str(k): (None if v is None else float(v)) for k, v in thr.items()}, meta=dict(meta),
                   questions=questions)

    def to_dict(self) -> dict:
        return {"temperature": self.temperature, "thresholds": self.thresholds, **({"questions": self.questions} if self.questions else {}),
                **({"meta": self.meta} if self.meta else {})}

    @classmethod
    def load(cls, path: str | Path | None) -> "Calibration":
        if path is None or not Path(path).exists():
            return cls()
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(path).with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        tmp.replace(path)


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / max(temperature, 1e-6)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def probs_to_logits(p: np.ndarray) -> np.ndarray:
    """Log-probabilites restreintes : la temperature agit alors comme p^(1/T) renormalise."""
    return np.log(np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0))


def confidence(p: np.ndarray) -> float:
    """1 - entropie normalisee : 1.0 = toute la masse sur une option, 0.0 = uniforme.
    Statistique par defaut ; la distribution complete est toujours renvoyee."""
    n = len(p)
    if n <= 1:
        return 1.0
    p = np.clip(p, 1e-12, 1.0)
    h = -(p * np.log(p)).sum()
    return float(max(0.0, min(1.0, 1.0 - h / math.log(n))))


def merge_branches(kind: str, results: list[tuple[Branch, np.ndarray]], cal: Calibration | float, qid: str | None = None,
                   q=None, state=None) -> dict[Any, float]:
    """Moyenne des probabilites (apres temperature) sur les branches permutees d'une meme question.
    cal : temperature, ou calibration (sa temperature pour cette question lue sur cet etat : Calibration.t_for)."""
    T = float(cal) if isinstance(cal, (int, float)) else cal.t_for(qid, q, state, kind)
    acc: dict[Any, float] = {}
    for br, logits in results:
        probs = softmax(logits, T)
        for k, pr in zip(br.keys, probs):
            acc[k] = acc.get(k, 0.0) + float(pr)
    total = sum(acc.values())
    return {k: v / total for k, v in acc.items()}


def to_answer(kind: str, key_probs: dict[Any, float], q=None):
    if kind == "noul":
        return NoulAnswer(noul=round(key_probs[True], 6))
    if kind == "choice":
        probs = {str(k): round(v, 6) for k, v in key_probs.items()}
        best = max(probs, key=probs.get)
        return ChoiceAnswer(choice=best, probabilities=probs,
                            confidence=round(confidence(np.array(list(probs.values()))), 6))
    assert kind == "score" and isinstance(q, ScoreQuestion)
    levels = sorted(key_probs.keys())
    p = np.array([key_probs[i] for i in levels])
    return ScoreAnswer(
        score=round(float((p * np.array(levels, dtype=float)).sum()), 6),
        legend={str(i): render_text(q.criteria[i]) for i in levels},
        probabilities={str(i): round(float(key_probs[i]), 6) for i in levels},
        confidence=round(confidence(p), 6),
    )

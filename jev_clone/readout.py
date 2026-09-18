"""Des probabilites d'etiquettes par branche aux reponses typees.

Deux reglages post-hoc rendent la lecture d'un LM pre-entraine "calibree" :
  * une temperature par primitive (ajustee sur des donnees etiquetees, voir calibrate.py)
  * une statistique de `confidence` (TypeSafe : "a quel point la distribution est piquee")
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from jev_clone.prompt import Branch, render_text
from jev_clone.schema import ChoiceAnswer, NoulAnswer, ScoreAnswer, ScoreQuestion


@dataclass
class Calibration:
    temperature: dict[str, float] = field(default_factory=lambda: {"noul": 1.0, "choice": 1.0, "score": 1.0})
    thresholds: dict[str, float] = field(default_factory=dict)   # seuil de confiance par question (optionnel)

    def t(self, kind: str) -> float:
        return float(self.temperature.get(kind, 1.0))

    @classmethod
    def load(cls, path: str | Path | None) -> "Calibration":
        if path is None or not Path(path).exists():
            return cls()
        with open(path) as f:
            d = json.load(f)
        return cls(temperature=d.get("temperature", {}), thresholds=d.get("thresholds", {}))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"temperature": self.temperature, "thresholds": self.thresholds}, f, indent=2)


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


def merge_branches(kind: str, results: list[tuple[Branch, np.ndarray]], cal: Calibration) -> dict[Any, float]:
    """Moyenne des probabilites (apres temperature) sur les branches permutees d'une meme question."""
    acc: dict[Any, float] = {}
    for br, logits in results:
        probs = softmax(logits, cal.t(kind))
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

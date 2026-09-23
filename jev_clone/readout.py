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


KINDS = ("noul", "choice", "score")


@dataclass
class Calibration:
    temperature: dict[str, float] = field(default_factory=lambda: {"noul": 1.0, "choice": 1.0, "score": 1.0})
    # seuil de la porte par question, sur la statistique de fusion.gate_statistic (probabilite de l'option retenue,
    # apres temperature) ; None = aucun seuil n'atteint la precision visee : toujours escalader
    thresholds: dict[str, float | None] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)      # modele, nombre d'exemples, ECE avant / apres, source...

    def t(self, kind: str) -> float:
        return float(self.temperature.get(kind, 1.0))

    def threshold(self, name: str, default: float) -> float:
        """Seuil calibre d'une question (`default` si elle n'a pas ete calibree) ; inf = ne jamais agir seul."""
        if name not in self.thresholds:
            return float(default)
        v = self.thresholds[name]
        return math.inf if v is None else float(v)

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
        meta = d.get("meta") if isinstance(d.get("meta"), dict) else {}
        return cls(temperature=temp, thresholds={str(k): (None if v is None else float(v)) for k, v in thr.items()}, meta=dict(meta))

    def to_dict(self) -> dict:
        return {"temperature": self.temperature, "thresholds": self.thresholds, **({"meta": self.meta} if self.meta else {})}

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

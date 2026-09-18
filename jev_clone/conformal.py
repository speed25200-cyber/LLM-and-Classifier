"""Garanties formelles sur les decisions : prediction conforme et porte a risque controle.

Jev fournit des probabilites *calibrees* (en moyenne, 0,9 veut dire 90 % de reussite) mais aucune garantie
finie-echantillon. Ici, a partir d'un jeu de calibration etiquete (n >= 100), on obtient :

  * **ensembles de prediction conformes (LAC)** : pour un niveau alpha, l'ensemble C(x) = {options k : p_k >= 1 - q}
    contient la bonne reponse avec probabilite >= 1 - alpha (garantie marginale, sans hypothese sur le modele).
    Un ensemble de taille 1 = decision sure ; taille > 1 = ambiguite explicite -> escalade ou choix parmi C(x).
  * **porte a risque controle** : le plus petit seuil de confiance tau tel que, sur les cas ou conf >= tau,
    l'erreur est <= alpha avec probabilite >= 1 - delta (borne binomiale exacte de Clopper-Pearson, test
    sequentiel a seuils decroissants dans l'esprit de Learn-then-Test). Au-dessus de tau, System One agit seul ;
    en dessous, Bonsai.

Reference : Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction" (2021) ; Angelopoulos et al.,
"Learn then Test" (2021). Pas de dependance au-dela de numpy.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _lac_scores(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return 1.0 - probs[np.arange(len(labels)), labels]


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Quantile conforme : ceil((n+1)(1-alpha))/n-ieme plus petit score (1.0 si n trop petit)."""
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return 1.0
    return float(np.sort(scores)[k - 1])


@dataclass
class ConformalCalibrator:
    alpha: float = 0.1
    qhat: dict[str, float] = field(default_factory=dict)   # par primitive : noul / choice / score
    n: dict[str, int] = field(default_factory=dict)

    def fit(self, kind: str, probs: np.ndarray, labels: np.ndarray) -> float:
        """probs [N, K] (colonnes inutilisees a 0), labels [N] index de la bonne option."""
        q = conformal_quantile(_lac_scores(np.asarray(probs, float), np.asarray(labels)), self.alpha)
        self.qhat[kind] = q
        self.n[kind] = int(len(labels))
        return q

    def prediction_set(self, kind: str, probs: dict[str, float]) -> list[str]:
        q = self.qhat.get(kind)
        if q is None:
            return list(probs)
        keep = [k for k, p in probs.items() if p >= 1.0 - q]
        if not keep:  # garde-fou : jamais vide
            keep = [max(probs, key=probs.get)]
        return keep

    def coverage(self, kind: str, probs: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
        """(couverture empirique, taille moyenne des ensembles) sur un jeu de test."""
        q = self.qhat[kind]
        sets = np.asarray(probs, float) >= 1.0 - q
        labels = np.asarray(labels)
        cov = float(sets[np.arange(len(labels)), labels].mean())
        return cov, float(sets.sum(1).mean())

    def to_dict(self) -> dict:
        return {"alpha": self.alpha, "qhat": self.qhat, "n": self.n}

    @classmethod
    def from_dict(cls, d: dict | None) -> "ConformalCalibrator":
        if not d:
            return cls()
        return cls(alpha=float(d.get("alpha", 0.1)), qhat=dict(d.get("qhat", {})), n=dict(d.get("n", {})))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path | None) -> "ConformalCalibrator":
        if path is None or not Path(path).exists():
            return cls()
        with open(path) as f:
            return cls.from_dict(json.load(f))


def hoeffding_upper(err_hat: float, n: int, delta: float) -> float:
    """Borne superieure (prob. >= 1 - delta) sur l'erreur vraie a partir de l'erreur empirique sur n cas."""
    if n <= 0:
        return 1.0
    return min(1.0, err_hat + math.sqrt(math.log(1.0 / delta) / (2.0 * n)))


def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(Bin(n, p) <= k), calcule en log-espace (stable jusqu'a n ~ 10^5)."""
    if p <= 0:
        return 1.0
    if p >= 1:
        return 0.0 if k < n else 1.0
    lp, lq = math.log(p), math.log(1 - p)
    total = 0.0
    for i in range(k + 1):
        total += math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq)
    return min(1.0, total)


def binomial_upper(k: int, n: int, delta: float) -> float:
    """Borne superieure de Clopper-Pearson (niveau 1 - delta) sur la probabilite d'erreur apres k erreurs sur n."""
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    lo, hi = k / n, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _binom_cdf(k, n, mid) > delta:
            lo = mid
        else:
            hi = mid
    return hi


def risk_controlled_threshold(conf: np.ndarray, correct: np.ndarray, alpha: float = 0.05, delta: float = 0.1) -> dict:
    """Plus petit seuil tau (= plus grande couverture) tel que l'erreur sur {conf >= tau} soit <= alpha avec
    probabilite >= 1 - delta. Seuils testes du plus strict au plus permissif ; le test commence au premier n ou
    zero erreur suffirait a passer (regle fixee a l'avance, independante des donnees) et s'arrete au premier
    echec : test sequentiel a niveau delta fixe, valide sans correction multiple."""
    conf = np.asarray(conf, float)
    correct = np.asarray(correct, float)
    order = np.argsort(-conf)
    c, ok = conf[order], correct[order]
    n_start = 1
    while n_start <= len(c) and binomial_upper(0, n_start, delta) > alpha:
        n_start += 1
    best = None
    for i in range(n_start - 1, len(c)):
        if i + 1 < len(c) and c[i + 1] == c[i]:
            continue  # on ne coupe pas au milieu d'ex aequo
        n = i + 1
        k = int(round(n - ok[:n].sum()))
        ub = binomial_upper(k, n, delta)
        if ub <= alpha:
            best = {"threshold": float(c[i]), "n_selected": n, "coverage": n / len(c),
                    "empirical_error": k / n, "error_upper_bound": float(ub)}
        else:
            break
    return best or {"threshold": None, "n_selected": 0, "coverage": 0.0, "empirical_error": None, "error_upper_bound": None,
                    "note": f"aucun seuil ne garantit une erreur <= {alpha} a {1-delta:.0%} avec n={len(c)} : escalade systematique"}

"""Calibration post-hoc et metriques (ECE, Brier, NLL, precision selective) a partir d'un JSONL
d'exemples etiquetes : {"state": ..., "questions": {...}, "labels": {qid: reponse}}.

    python -m jev_clone.calibrate --server http://127.0.0.1:8081 --data data/val.jsonl --out runs/calibration.json

RLCD-lite : Jev est entraine par "Reinforcement Learning for Calibrated Decisions" avec une regle de
score propre (log / Brier). Quand la sortie du modele EST la distribution, l'esperance de la recompense
est differentiable : l'objectif RL se reduit a minimiser NLL ou Brier (voir training/). Ici on n'ajuste
qu'une temperature par primitive, ce qui suffit souvent a rendre les pourcentages "honnetes".
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass

import numpy as np

from jev_clone.readout import softmax


@dataclass
class Report:
    n: int
    accuracy: float
    nll: float
    brier: float
    ece: float
    mean_confidence: float
    bins: list[dict]

    def __str__(self) -> str:
        s = (f"n={self.n}  acc={self.accuracy:.3f}  NLL={self.nll:.3f}  Brier={self.brier:.3f}  "
             f"ECE={self.ece:.3f}  conf.moy={self.mean_confidence:.3f}\n")
        s += "  bin        n   conf    acc\n"
        for b in self.bins:
            if b["n"]:
                s += f"  {b['lo']:.2f}-{b['hi']:.2f} {b['n']:5d}  {b['conf']:.3f}  {b['acc']:.3f}\n"
        return s


def report(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> Report:
    """probs [N, K] (les colonnes inutilisees a 0), labels [N] index de la bonne option."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    n = len(labels)
    pred = probs.argmax(1)
    conf = probs.max(1)
    correct = (pred == labels).astype(float)
    p_true = np.clip(probs[np.arange(n), labels], 1e-12, 1)
    onehot = np.zeros_like(probs); onehot[np.arange(n), labels] = 1
    brier = float(((probs - onehot) ** 2).sum(1).mean())
    edges = np.linspace(0, 1, n_bins + 1)
    ece, bins = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        k = int(m.sum())
        if k:
            c, a = float(conf[m].mean()), float(correct[m].mean())
            ece += k / n * abs(c - a)
        else:
            c = a = 0.0
        bins.append({"lo": float(lo), "hi": float(hi), "n": k, "conf": c, "acc": a})
    return Report(n=n, accuracy=float(correct.mean()), nll=float(-np.log(p_true).mean()), brier=brier,
                  ece=float(ece), mean_confidence=float(conf.mean()), bins=bins)


def fit_temperature(logits: np.ndarray, labels: np.ndarray, grid=None) -> float:
    """Temperature qui minimise la NLL (recherche sur grille log-espacee puis affinage)."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    grid = grid if grid is not None else np.exp(np.linspace(math.log(0.2), math.log(20.0), 120))

    def nll(T):
        p = np.stack([softmax(l, T) for l in logits])
        return float(-np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, 1)).mean())

    best = min(grid, key=nll)
    fine = np.linspace(best * 0.8, best * 1.25, 60)
    return float(min(fine, key=nll))


def threshold_for_precision(conf: np.ndarray, correct: np.ndarray, target: float = 0.95) -> float | None:
    """Plus petit seuil de confiance tel que la precision des cas au-dessus du seuil >= target.
    Sert de "porte" (gate) : au-dessus -> System One agit seul, en dessous -> escalade vers Bonsai."""
    order = np.argsort(-conf)
    c, ok = conf[order], correct[order]
    cum_acc = np.cumsum(ok) / np.arange(1, len(ok) + 1)
    valid = np.where(cum_acc >= target)[0]
    if len(valid) == 0:
        return None
    return float(c[valid[-1]])


def _index_of(kind: str, q, label) -> int:
    if kind == "noul":
        return 0 if bool(label) else 1
    if kind == "choice":
        keys = list(q.criteria.keys()) if isinstance(q.criteria, dict) else list(q.criteria)
        return keys.index(label)
    return int(label)


def collect(engine, examples: list[dict]):
    """Lit chaque exemple avec l'engine (temperature 1) et renvoie (logits par kind, labels par kind, conf/correct par qid)."""
    from jev_clone.readout import probs_to_logits
    from jev_clone.schema import SystemOneRequest

    per_kind: dict[str, tuple[list, list]] = {"noul": ([], []), "choice": ([], []), "score": ([], [])}
    per_qid: dict[str, tuple[list, list]] = {}
    for ex in examples:
        req = SystemOneRequest(state=ex["state"], questions=ex["questions"])
        resp = engine.answer(req)
        for qid, q in req.questions.items():
            if qid not in ex.get("labels", {}):
                continue
            a = resp.answers[qid]
            if q.type == "noul":
                p = np.array([a.noul, 1 - a.noul])
            else:
                p = np.array(list(a.probabilities.values()))
            gold = _index_of(q.type, q, ex["labels"][qid])
            per_kind[q.type][0].append(probs_to_logits(p)); per_kind[q.type][1].append(gold)
            per_qid.setdefault(qid, ([], []))
            per_qid[qid][0].append(float(p.max())); per_qid[qid][1].append(float(p.argmax() == gold))
    return per_kind, per_qid


def pad(rows: list[np.ndarray]) -> np.ndarray:
    K = max(len(r) for r in rows)
    out = np.full((len(rows), K), -1e9)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Calibration du clone Jev (temperature + seuils)")
    ap.add_argument("--server", default="http://127.0.0.1:8081")
    ap.add_argument("--data", required=True, help="JSONL {state, questions, labels}")
    ap.add_argument("--out", default="runs/calibration.json")
    ap.add_argument("--target-precision", type=float, default=0.95)
    args = ap.parse_args(argv)

    from jev_clone.backend_llamacpp import LlamaCppBackend
    from jev_clone.engine import SystemOneEngine
    from jev_clone.readout import Calibration

    engine = SystemOneEngine(LlamaCppBackend(args.server))
    with open(args.data) as f:
        examples = [json.loads(l) for l in f if l.strip()]
    per_kind, per_qid = collect(engine, examples)

    cal = Calibration()
    for kind, (rows, labels) in per_kind.items():
        if not rows:
            continue
        L, y = pad(rows), np.array(labels)
        print(f"== {kind} : avant ==\n{report(np.stack([softmax(l) for l in L]), y)}")
        T = fit_temperature(L, y)
        cal.temperature[kind] = T
        print(f"== {kind} : apres T={T:.3f} ==\n{report(np.stack([softmax(l, T) for l in L]), y)}")
    for qid, (conf, ok) in per_qid.items():
        thr = threshold_for_precision(np.array(conf), np.array(ok), args.target_precision)
        if thr is not None:
            cal.thresholds[qid] = round(thr, 4)
    cal.save(args.out)
    print(f"calibration ecrite dans {args.out}: {cal}")


if __name__ == "__main__":
    main()

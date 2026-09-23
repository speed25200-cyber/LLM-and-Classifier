"""Calibration post-hoc et metriques (ECE, Brier, NLL, precision selective) a partir d'un JSONL
d'exemples etiquetes : {"state": ..., "labels": {qid: reponse}, "questions": {...} (facultatif)}.
Sans "questions", c'est le schema pre-tour de Prophet (jev_clone.prophet.PROPHET_TURN), restreint aux questions
etiquetees : le format des graines livrees (jev_clone/seeds/prophet_seeds.jsonl, copie de training/seeds/).

    # classifieur de Prophet Studio : fichier par modele, applique par Studio a ce S1 seulement (ou bouton "Calibrer")
    python -m jev_clone.calibrate --server http://127.0.0.1:7881 --studio-model ternary-1.7b
    # jeu de donnees quelconque -> fichier libre (jev serve : JEV_CALIBRATION)
    python -m jev_clone.calibrate --server http://127.0.0.1:8081 --data data/val.jsonl --out runs/calibration.json

RLCD-lite : Jev est entraine par "Reinforcement Learning for Calibrated Decisions" avec une regle de
score propre (log / Brier). Quand la sortie du modele EST la distribution, l'esperance de la recompense
est differentiable : l'objectif RL se reduit a minimiser NLL ou Brier (voir training/). Ici on n'ajuste
qu'une temperature par question etiquetee, ce qui suffit souvent a rendre les pourcentages "honnetes". Elle ne vaut que
pour cette question (empreinte : type, consigne, options) lue sur un etat de meme forme : le garde-fou, la voix ou le
computer use, jamais etiquetes ici, restent lus bruts (T = 1) et gardent le sens de leurs seuils fixes.
Les seuils par question sont calcules APRES la temperature, sur la statistique de la porte (fusion.gate_statistic :
probabilite de l'option retenue) ; une question qu'aucun seuil ne rend assez precise recoit `null` : toujours escalader.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from jev_clone.fusion import gate_statistic
from jev_clone.readout import Calibration, probs_to_logits, question_fingerprint, softmax, state_keys

SEEDS = Path(__file__).parent / "seeds" / "prophet_seeds.jsonl"   # graines etiquetees de Prophet, livrees avec le paquet


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

    def summary(self) -> dict:
        return {"n": self.n, **{k: round(getattr(self, k), 4) for k in ("accuracy", "nll", "brier", "ece", "mean_confidence")}}


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
    Sert de "porte" (gate) : au-dessus -> System One agit seul, en dessous -> escalade vers Bonsai.
    Les ex aequo passent ensemble la porte : la precision n'est evaluee qu'en fin de groupe (a la resolution des reponses,
    6 decimales), et le seuil rendu admet exactement l'ensemble evalue (arrondi lisible seulement s'il n'en ajoute aucun)."""
    c0 = np.round(np.asarray(conf, dtype=np.float64), 6)
    if c0.size == 0:
        return None
    order = np.argsort(-c0, kind="stable")
    c, ok = c0[order], np.asarray(correct, dtype=np.float64)[order]
    cum_acc = np.cumsum(ok) / np.arange(1, len(ok) + 1)
    end = np.r_[c[:-1] > c[1:], True]            # {conf >= c[i]} = prefixe [0..i] seulement en fin de groupe
    valid = np.where(end & (cum_acc >= target))[0]
    if len(valid) == 0:
        return None
    i = int(valid[-1])
    thr, below = float(c[i]), (float(c[i + 1]) if i + 1 < len(c) else -math.inf)
    t = math.floor(thr * 1e4) / 1e4              # lisible, et marge pour les reponses relues a l'execution
    return t if t > below else round((thr + below) / 2, 7)   # valeurs a moins de 1e-4 : un point strictement entre les deux


def _index_of(kind: str, q, label) -> int:
    if kind == "noul":
        return 0 if bool(label) else 1
    if kind == "choice":
        keys = list(q.criteria.keys()) if isinstance(q.criteria, dict) else list(q.criteria)
        return keys.index(label)
    return int(label)


def load_examples(path: str | Path | None = None) -> list[dict]:
    """JSONL d'exemples etiquetes ; par defaut les graines de Prophet livrees avec le paquet."""
    with open(path or SEEDS, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def questions_for(ex: dict) -> dict:
    """Questions d'un exemple : les siennes, sinon celles du schema pre-tour de Prophet qui sont etiquetees."""
    if ex.get("questions"):
        return ex["questions"]
    from jev_clone.prophet import PROPHET_TURN
    qs = {k: PROPHET_TURN[k] for k in ex.get("labels", {}) if k in PROPHET_TURN}
    if not qs:
        raise ValueError("exemple sans 'questions' ni etiquette du schema pre-tour de Prophet "
                         f"({', '.join(PROPHET_TURN)})")
    return qs


def _probs(q, a) -> np.ndarray:
    """Distribution d'une reponse dans l'ordre des options declarees (celui des index d'etiquettes)."""
    if q.type == "noul":
        return np.array([a.noul, 1.0 - a.noul])
    if q.type == "choice":
        return np.array([a.probabilities[str(k)] for k in q.options()])
    return np.array([a.probabilities[str(i)] for i in range(len(q.criteria))])


def collect(engine, examples: list[dict], on_progress=None):
    """Lit chaque exemple avec l'engine (sans calibration : temperature 1).
    Renvoie per_kind {kind: (logits, index justes)} et per_qid {qid: (kind, logits, index justes, empreinte, cles d'etat)} :
    l'empreinte de la question et les cles d'etat communes a ses exemples bornent ou sa temperature s'appliquera."""
    from jev_clone.schema import SystemOneRequest

    cal = getattr(engine, "cal", None)
    if cal is not None and (any(abs(cal.t(k) - 1.0) > 1e-9 for k in ("noul", "choice", "score"))
                            or any(abs(float(e.get("T") or 1.0) - 1.0) > 1e-9 for e in cal.questions.values())):
        raise ValueError("la calibration se mesure sur des lectures brutes : engine sans calibration (T = 1) attendu")
    per_kind: dict[str, tuple[list, list]] = {"noul": ([], []), "choice": ([], []), "score": ([], [])}
    per_qid: dict[str, list] = {}
    for i, ex in enumerate(examples):
        req = SystemOneRequest(state=ex["state"], questions=questions_for(ex))
        resp = engine.answer(req)
        keys = state_keys(req.state)
        for qid, q in req.questions.items():
            if qid not in ex.get("labels", {}):
                continue
            p = _probs(q, resp.answers[qid])
            try:
                gold = _index_of(q.type, q, ex["labels"][qid])
            except (ValueError, TypeError):
                continue                      # etiquette hors des options : ignoree
            if not 0 <= gold < len(p):
                continue
            lg = probs_to_logits(p)
            per_kind[q.type][0].append(lg); per_kind[q.type][1].append(gold)
            fp = question_fingerprint(q)
            e = per_qid.setdefault(qid, [q.type, [], [], fp, keys])
            e[1].append(lg); e[2].append(gold)
            if e[3] is not None and e[3] != fp:
                e[3] = None                   # consigne variable sous un meme nom (rerank p0, p1...) : portee par nom seulement
            e[4] = None if e[4] is None or keys is None else sorted(set(e[4]) & set(keys))
        if on_progress is not None:
            on_progress(i + 1, len(examples))
    return per_kind, {k: tuple(v) for k, v in per_qid.items()}


def pad(rows: list[np.ndarray]) -> np.ndarray:
    K = max(len(r) for r in rows)
    out = np.full((len(rows), K), -1e9)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out


def fit(per_kind: dict, per_qid: dict, target_precision: float = 0.95, min_per_question: int = 30) -> tuple[Calibration, dict]:
    """Temperature par question (NLL ; repli sur celle de sa primitive sous `min_per_question` exemples), appliquee a cette
    seule question, puis seuil par question sur gate_statistic des probabilites APRES temperature. La temperature par
    primitive (tout l'ensemble) reste pour les rapports et les questions a peu d'exemples."""
    cal, reports = Calibration(), {}
    for kind, (rows, labels) in per_kind.items():
        if not rows:
            continue
        cal.temperature[kind] = round(fit_temperature(pad(rows), np.array(labels)), 4)
    after: dict[str, tuple[list, list]] = {}
    for qid, entry in per_qid.items():
        kind, rows, labels, *scope = entry
        fp, keys = (list(scope) + [None, None])[:2]
        T = round(fit_temperature(pad(rows), np.array(labels)), 4) if len(rows) >= min_per_question else cal.t(kind)
        P = [softmax(l, T) for l in rows]
        conf = np.array([gate_statistic(p) for p in P])
        ok = np.array([float(p.argmax() == g) for p, g in zip(P, labels)])
        cal.questions[qid] = {"T": T, "kind": kind, "fp": fp, "state": keys, "n": len(rows)}
        cal.thresholds[qid] = threshold_for_precision(conf, ok, target_precision)   # admet exactement l'ensemble evalue
        a = after.setdefault(kind, ([], []))
        a[0].extend(P); a[1].extend(labels)
    for kind, (rows, labels) in per_kind.items():
        if rows:
            P, y = after.get(kind, ([softmax(l, cal.t(kind)) for l in rows], labels))
            reports[kind] = {"before": report(np.stack([softmax(l) for l in pad(rows)]), np.array(labels)),
                             "after": report(pad_probs(P), np.array(y))}
    return cal, reports


def pad_probs(rows: list[np.ndarray]) -> np.ndarray:
    K = max(len(r) for r in rows)
    out = np.zeros((len(rows), K))
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out


def calibrate(engine, examples: list[dict], target_precision: float = 0.95, on_progress=None, **meta) -> tuple[Calibration, dict]:
    """Collecte + ajustement ; `meta` (modele, source...) est enregistre avec le resultat."""
    per_kind, per_qid = collect(engine, examples, on_progress)
    if not any(rows for rows, _ in per_kind.values()):
        raise ValueError("aucun exemple etiquete exploitable")
    cal, reports = fit(per_kind, per_qid, target_precision)
    cal.meta = {**{k: v for k, v in meta.items() if v is not None}, "n": len(examples), "ts": round(time.time(), 1),
                "target_precision": target_precision, "statistic": "top1",
                "report": {k: {w: r.summary() for w, r in v.items()} for k, v in reports.items()}}
    return cal, reports


def main(argv=None, engine=None):
    ap = argparse.ArgumentParser(description="Calibration du clone Jev (temperature + seuils)")
    ap.add_argument("--server", default="http://127.0.0.1:8081")
    ap.add_argument("--data", default=None, help="JSONL {state, labels, questions?} (defaut : graines de Prophet livrees)")
    ap.add_argument("--out", default=None, help="defaut : runs/calibration.json, ou le fichier de Studio avec --studio-model")
    ap.add_argument("--studio-model", default=None, help="id du classifieur dans Prophet Studio : ecrit <donnees>/runs/calibration/<id>.json")
    ap.add_argument("--target-precision", type=float, default=0.95)
    args = ap.parse_args(argv)

    out, weights = args.out, None
    if out is None and args.studio_model:
        from prophet_studio.calibration import calibration_file, registered_weights
        from prophet_studio.config import Paths
        paths = Paths()
        out = calibration_file(paths.runs, args.studio_model)
        if out is None:
            ap.error(f"--studio-model : identifiant invalide {args.studio_model!r}")
        weights = registered_weights(paths.installed, args.studio_model)   # la calibration vaut pour ces poids-la
    out = out or "runs/calibration.json"
    if engine is None:
        from jev_clone.backend_llamacpp import LlamaCppBackend
        from jev_clone.engine import SystemOneEngine
        engine = SystemOneEngine(LlamaCppBackend(args.server, max_workers=4))
    cal, reports = calibrate(engine, load_examples(args.data), args.target_precision, source="cli",
                             model_id=args.studio_model, data=str(args.data or SEEDS))
    if weights is not None:
        from prophet_studio.calibration import weights_id
        cal.meta["weights"] = weights_id(weights)
    for kind, r in reports.items():
        print(f"== {kind} : avant ==\n{r['before']}")
        print(f"== {kind} : apres (temperature par question) ==\n{r['after']}")
    cal.save(out)
    print(f"calibration ecrite dans {out}: temperatures={ {q: e['T'] for q, e in cal.questions.items()} } seuils={cal.thresholds}")
    return cal


if __name__ == "__main__":
    main()

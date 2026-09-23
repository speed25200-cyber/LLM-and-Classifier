"""Mesure d'un classifieur (S1) servi par llama-server sur un JSONL etiquete, par famille et par question.

    # clone entraine (notebook) ou S1 de Studio (port du serveur S1 affiche dans Modeles > Runtime)
    python training/eval_clone.py --server http://127.0.0.1:8081 --data data/prophet_val.jsonl --json runs/eval.json

Par (famille, question) : n, exactitude, ECE, Brier (lecture brute, T = 1 : on compare des modeles, pas des calibrations).
Garde-fou (famille guard) : verdict reel de jev_clone.guard -> taux de confirmations demandees sur les actions benignes
(gene pour l'utilisateur) et taux d'actions risquees laissees passer (le chiffre qui doit rester a 0).
Comparer le S1 d'origine et le clone entraine : lancer deux fois, un serveur puis l'autre.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from jev_clone.calibrate import _index_of, _probs, pad_probs, report
from jev_clone.guard import verdict
from jev_clone.presets import RISKY_TOOL_CLASSES
from jev_clone.schema import SystemOneRequest


def evaluate(engine, rows: list[dict], on_progress=None) -> dict:
    per: dict = defaultdict(lambda: ([], []))
    guard = {"benign": 0, "benign_confirm": 0, "risky": 0, "risky_missed": 0}
    for i, ex in enumerate(rows):
        req = SystemOneRequest(state=ex["state"], questions=ex["questions"])
        resp = engine.answer(req)
        fam, labels = ex.get("family", "?"), ex.get("labels") or {}
        for qid, q in req.questions.items():
            if qid not in labels:
                continue
            try:
                gold = _index_of(q.type, q, labels[qid])
            except (ValueError, TypeError):
                continue
            p = _probs(q, resp.answers[qid])
            k = (fam, "t_*" if qid.startswith("t_") else qid)
            per[k][0].append(p); per[k][1].append(gold)
        if {"tool_risk", "risk", "policy_violation"} <= set(resp.answers) and "tool_risk" in labels:
            v = verdict(resp.answers)
            if labels["tool_risk"] in RISKY_TOOL_CLASSES:
                guard["risky"] += 1; guard["risky_missed"] += not v["needs_confirmation"]
            else:
                guard["benign"] += 1; guard["benign_confirm"] += v["needs_confirmation"]
        if on_progress is not None:
            on_progress(i + 1, len(rows))
    out = {}
    for (fam, qid), (P, y) in sorted(per.items()):
        r = report(pad_probs(P), np.array(y))
        out[f"{fam}/{qid}"] = {"n": r.n, "accuracy": round(r.accuracy, 4), "ece": round(r.ece, 4), "brier": round(r.brier, 4)}
    if guard["benign"] or guard["risky"]:
        out["guard/verdict"] = {**guard, "benign_confirm_rate": round(guard["benign_confirm"] / guard["benign"], 4) if guard["benign"] else None,
                                "risky_miss_rate": round(guard["risky_missed"] / guard["risky"], 4) if guard["risky"] else None}
    return out


def main(argv=None, engine=None):
    ap = argparse.ArgumentParser(description="Mesure d'un classifieur S1 par famille (exactitude, ECE, verdicts du garde-fou)")
    ap.add_argument("--server", default="http://127.0.0.1:8081")
    ap.add_argument("--data", required=True, help="JSONL etiquete (make_synthetic_prophet.py --val, graines, trajectoires...)")
    ap.add_argument("--limit", type=int, default=0, help="nombre maximal d'exemples (0 = tous)")
    ap.add_argument("--json", help="ecrire le resultat dans ce fichier")
    args = ap.parse_args(argv)
    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    rows = rows[: args.limit] if args.limit else rows
    if engine is None:
        from jev_clone.backend_llamacpp import LlamaCppBackend
        from jev_clone.engine import SystemOneEngine
        engine = SystemOneEngine(LlamaCppBackend(args.server, max_workers=4, timeout=120))
    res = evaluate(engine, rows, lambda i, n: print(f"{i}/{n}", file=sys.stderr) if i % 200 == 0 else None)
    for k, v in res.items():
        print(f"{k:32s} " + "  ".join(f"{a}={b}" for a, b in v.items()))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


if __name__ == "__main__":
    main()

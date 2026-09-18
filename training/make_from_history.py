"""L'historique de vos decisions -> exemples d'entrainement / calibration pour le clone.

"Des annees d'humains (ou d'agents) prenant exactement la meme decision, avec l'issue attachee" : c'est la
meilleure source d'entrainement possible. Entree : CSV ou JSONL avec une colonne d'etat (texte ou JSON) et
une ou plusieurs colonnes de decision ; pour chaque decision, un schema (type + options) dans un JSON.

    python training/make_from_history.py --in tickets.csv --state-col text \
        --schema schema.json --out data/history.jsonl --val data/history_val.jsonl --val-frac 0.1

schema.json : {"team": {"type": "choice", "instructions": "...", "criteria": {"billing": "...", ...}, "col": "team"},
               "refund": {"type": "noul", "instructions": "...", "col": "refund_flag"},
               "priority": {"type": "score", "instructions": "...", "criteria": ["low", "mid", "high"], "col": "priority"}}
Les valeurs de colonne sont mises en correspondance : choice -> nom d'option (insensible a la casse),
noul -> vrai/faux/1/0/yes/no, score -> index de niveau (entier) ou nom de niveau.
Option : --outcome-col garde seulement les lignes dont l'issue confirme la decision (ex. "resolved").
"""

from __future__ import annotations

import argparse
import csv
import json
import random


def parse_value(kind: str, q: dict, raw):
    if raw is None:
        return None
    s = str(raw).strip()
    if kind == "noul":
        return s.lower() in ("1", "true", "yes", "oui", "y", "t", "vrai")
    if kind == "choice":
        keys = list(q["criteria"].keys()) if isinstance(q["criteria"], dict) else list(q["criteria"])
        for k in keys:
            if k.lower() == s.lower():
                return k
        return None
    levels = q["criteria"]
    if s.isdigit() and int(s) < len(levels):
        return int(s)
    for i, lv in enumerate(levels):
        if str(lv).lower() == s.lower():
            return i
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--state-col", required=True, help="colonne contenant l'etat (texte ou JSON) ; plusieurs colonnes separees par des virgules -> objet JSON")
    ap.add_argument("--schema", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--outcome-col")
    ap.add_argument("--outcome-ok", default="resolved,ok,correct,accepted,true,1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    schema = json.load(open(args.schema))
    questions = {qid: {k: v for k, v in q.items() if k != "col"} for qid, q in schema.items()}
    cols = [c.strip() for c in args.state_col.split(",")]
    ok_values = {v.strip().lower() for v in args.outcome_ok.split(",")}

    if args.inp.endswith(".jsonl"):
        rows = [json.loads(l) for l in open(args.inp) if l.strip()]
    else:
        with open(args.inp, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    examples, skipped = [], 0
    for r in rows:
        if args.outcome_col and str(r.get(args.outcome_col, "")).strip().lower() not in ok_values:
            skipped += 1; continue
        state = r[cols[0]] if len(cols) == 1 else {c: r.get(c) for c in cols}
        labels = {}
        for qid, q in schema.items():
            v = parse_value(q["type"], q, r.get(q["col"]))
            if v is not None:
                labels[qid] = v
        if not labels or not state:
            skipped += 1; continue
        examples.append({"state": state, "questions": questions, "labels": labels})

    random.Random(args.seed).shuffle(examples)
    n_val = int(len(examples) * args.val_frac) if args.val else 0
    with open(args.out, "w") as f:
        for ex in examples[n_val:]:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    if args.val:
        with open(args.val, "w") as f:
            for ex in examples[:n_val]:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"{len(examples) - n_val} exemples -> {args.out}" + (f" ; {n_val} -> {args.val}" if args.val else "") + f" ; {skipped} lignes ignorees")


if __name__ == "__main__":
    main()

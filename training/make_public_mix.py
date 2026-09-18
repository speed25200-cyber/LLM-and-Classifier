"""Convertit quelques jeux publics de decision en JSONL {state, questions, labels} pour le clone Jev
(recette de decider / system-one-gemma, reduite). A executer la ou Hugging Face est accessible (Colab).

    python training/make_public_mix.py --out data/public_train.jsonl --val data/public_val.jsonl --per-task 8000

Taches : banking77 (choice 77 -> sous-echantillonne a 10 options avec la bonne + "other"), ag_news (choice 4),
go_emotions (noul par emotion), MMLU (choice 4, connaissances), yelp (score 1-5) si disponible.
Chaque exemple garde toujours la bonne option ; les options sont melangees ; un "other" est ajoute
aleatoirement pour apprendre l'abstention.
"""

from __future__ import annotations

import argparse
import json
import random

from datasets import load_dataset


def choice_example(text, instructions, options, gold, rng, max_options=10, abstain_p=0.3):
    opts = list(options)
    if len(opts) > max_options:
        others = [o for o in opts if o != gold]
        opts = rng.sample(others, max_options - 1) + [gold]
    rng.shuffle(opts)
    if rng.random() < abstain_p:
        opts.append("other / none of the above")
    return {"state": text, "questions": {"q": {"type": "choice", "instructions": instructions, "criteria": opts}},
            "labels": {"q": gold}}


def build(per_task: int, seed: int):
    rng = random.Random(seed)
    rows = []

    def take(ds, n):
        idx = list(range(len(ds))); rng.shuffle(idx)
        return [ds[i] for i in idx[:n]]

    try:  # banking77 : intentions bancaires (77 classes)
        ds = load_dataset("banking77", split="train"); names = ds.features["label"].names
        for ex in take(ds, per_task):
            rows.append(choice_example(ex["text"], "What is the customer's intent?",
                                       [n.replace("_", " ") for n in names], names[ex["label"]].replace("_", " "), rng))
    except Exception as e:
        print("banking77 ignore:", e)

    try:  # ag_news : 4 themes
        ds = load_dataset("ag_news", split="train"); names = ds.features["label"].names
        for ex in take(ds, per_task):
            rows.append(choice_example(ex["text"], "What is the topic of this news article?", names, names[ex["label"]], rng, abstain_p=0.1))
    except Exception as e:
        print("ag_news ignore:", e)

    try:  # go_emotions : noul par emotion
        ds = load_dataset("go_emotions", "simplified", split="train"); names = ds.features["labels"].feature.names
        for ex in take(ds, per_task):
            present = {names[i] for i in ex["labels"]}
            emo = rng.choice(names)
            rows.append({"state": ex["text"], "questions": {"q": {"type": "noul", "instructions": f"Does the text express the emotion '{emo}'?"}},
                         "labels": {"q": emo in present}})
    except Exception as e:
        print("go_emotions ignore:", e)

    try:  # MMLU : QCM de connaissances (4 options)
        ds = load_dataset("cais/mmlu", "all", split="auxiliary_train" if per_task > 2000 else "validation")
        for ex in take(ds, per_task):
            opts = list(ex["choices"]); gold = opts[int(ex["answer"])]
            rows.append(choice_example(ex["question"], "Which answer is correct?", opts, gold, rng, abstain_p=0.0))
    except Exception as e:
        print("mmlu ignore:", e)

    try:  # yelp : notation 1-5 -> score a 5 niveaux
        ds = load_dataset("yelp_review_full", split="train")
        levels = ["1 star: terrible", "2 stars: poor", "3 stars: average", "4 stars: good", "5 stars: excellent"]
        for ex in take(ds, per_task // 2):
            rows.append({"state": ex["text"][:2000], "questions": {"q": {"type": "score", "instructions": "How would the reviewer rate this business?", "criteria": levels}},
                         "labels": {"q": int(ex["label"])}})
    except Exception as e:
        print("yelp ignore:", e)

    rng.shuffle(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/public_train.jsonl")
    ap.add_argument("--val", default="data/public_val.jsonl")
    ap.add_argument("--per-task", type=int, default=8000)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rows = build(args.per_task, args.seed)
    n_val = int(len(rows) * args.val_frac)
    import os; os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.val, "w") as f:
        for r in rows[:n_val]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.out, "w") as f:
        for r in rows[n_val:]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows) - n_val} exemples -> {args.out} ; {n_val} -> {args.val}")


if __name__ == "__main__":
    main()

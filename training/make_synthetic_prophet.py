"""Zero donnee -> jeu d'entrainement des jugements de Prophet : les graines manuscrites sont etendues par Bonsai.

    python training/make_synthetic_prophet.py --bonsai http://127.0.0.1:8080 --seeds training/seeds/prophet_seeds.jsonl \
        --per-seed 20 --out data/prophet_states.jsonl
    python -m jev_clone.distill --bonsai http://127.0.0.1:8080 --in data/prophet_states.jsonl --out data/prophet_labeled.jsonl \
        --mode soft --think-if-below 0.85

Pour chaque graine, Bonsai produit `--per-seed` demandes nouvelles de meme nature (formulation, langue FR/EN,
domaine, longueur varies, avec des cas ambigus et des cas pieges), en JSON contraint. Les etiquettes
d'observation de la graine (intent, language) et le risque sont herites ; `direct`, `clarify`,
`needs_reasoning` sont laisses a la distillation. Les questions sont celles de `jev_clone.prophet.PROPHET_TURN`.
"""

from __future__ import annotations

import argparse
import json
import random
import sys

from jev_clone.prophet import PROPHET_TURN

GEN_SCHEMA = {"type": "object", "properties": {"requests": {"type": "array", "items": {"type": "string"}, "minItems": 1}},
              "required": ["requests"], "additionalProperties": False}


def expand(backend, seed: dict, n: int, budget: int = 0) -> list[str]:
    labels = seed["labels"]
    prompt = (f"You generate realistic requests a user could type to Prophet, a general local agent that can write code, run "
              f"commands, use the web and keep notes.\nSeed request: {json.dumps(seed['state']['request'], ensure_ascii=False)}\n"
              f"Kind: {labels['intent']}" + (f"\nStack: {labels.get('language')}" if labels.get("language") else "")
              + f"\nRisk level 0-3: {labels.get('risk', 0)}\n\nWrite {n} NEW requests of the same kind and similar risk, varied in "
              f"wording, length, domain and language (mix French and English), some terse, some detailed, a few ambiguous. Return JSON.")
    d = backend.chat([{"role": "user", "content": prompt}], max_tokens=2048, thinking_budget=budget, temperature=0.9,
                     extra={"response_format": {"type": "json_schema", "json_schema": {"name": "requests", "schema": GEN_SCHEMA}}})
    try:
        reqs = json.loads(d["choices"][0]["message"].get("content") or "{}").get("requests", [])
    except json.JSONDecodeError:
        reqs = []
    return [r.strip() for r in reqs if isinstance(r, str) and r.strip()][:n]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bonsai", default="http://127.0.0.1:8080")
    ap.add_argument("--seeds", default="training/seeds/prophet_seeds.jsonl")
    ap.add_argument("--per-seed", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=0)
    args = ap.parse_args(argv)

    from jev_clone.backend_llamacpp import LlamaCppBackend
    backend = LlamaCppBackend(args.bonsai, max_workers=1, timeout=600)
    seeds = [json.loads(l) for l in open(args.seeds) if l.strip()]
    rows = [{"state": s["state"], "questions": PROPHET_TURN, "labels": s["labels"], "source": "seed"} for s in seeds]
    for i, s in enumerate(seeds):
        for r in expand(backend, s, args.per_seed, args.budget):
            labels = {"intent": s["labels"]["intent"], "risk": s["labels"].get("risk", 0)}
            if s["labels"].get("language"):
                labels["language"] = s["labels"]["language"]
            rows.append({"state": {"request": r}, "questions": PROPHET_TURN, "labels": labels, "source": "synthetic"})
        print(f"{i+1}/{len(seeds)} graines etendues, {len(rows)} demandes", file=sys.stderr)
    random.Random(args.seed).shuffle(rows)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} demandes -> {args.out} (completer les etiquettes avec jev_clone.distill)", file=sys.stderr)


if __name__ == "__main__":
    main()

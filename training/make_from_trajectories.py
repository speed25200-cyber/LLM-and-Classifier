"""Trajectoires du computer use (navigateur et bureau) -> exemples d'entrainement de la politique rapide (DAgger).

Chaque pas ou Bonsai a repris la main devient un exemple etiquete par Bonsai (source "bonsai") : action, element cible,
slot a saisir, objectif atteint, sur l'etat qu'il a vu. Les pas rapides verifies avec succes sont gardes en
auto-etiquetage (source "self" ; --no-self pour les exclure, --self-weight pour les sous-echantillonner).
Sortie au format de training/train_lora_rlcd.py : une ligne JSON {"state", "questions", "labels", "source"}.

Journaux ecrits par Prophet Studio (dossier de donnees, voir Reglages > Donnees) :
    <donnees>/runs/trajectories.jsonl          outil navigateur (browse)
    <donnees>/runs/desktop_trajectories.jsonl  controle du bureau (desktop)

    python training/make_from_trajectories.py --in runs/trajectories.jsonl runs/desktop_trajectories.jsonl \
        --out data/computer_use.jsonl --val data/computer_use_val.jsonl --val-frac 0.1
"""

from __future__ import annotations

import argparse
import json
import random

from jev_clone.computer_use import trajectory_to_examples


def convert(paths: list[str], self_examples: bool = True, self_weight: float = 1.0, seed: int = 0) -> tuple[list[dict], dict]:
    """Lit les journaux (lignes illisibles ignorees) ; rend les exemples et des compteurs."""
    rng = random.Random(seed)
    out: list[dict] = []
    stats = {"trajectories": 0, "bad_lines": 0, "bonsai": 0, "self": 0}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    traj = json.loads(line)
                    exs = trajectory_to_examples(traj)
                except Exception:
                    stats["bad_lines"] += 1
                    continue
                stats["trajectories"] += 1
                for ex in exs:
                    if ex["source"] == "self" and (not self_examples or rng.random() >= self_weight):
                        continue
                    stats[ex["source"]] = stats.get(ex["source"], 0) + 1
                    out.append(ex)
    return out, stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", nargs="+", required=True, help="un ou plusieurs journaux trajectories.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--no-self", action="store_true", help="n'utiliser que les etiquettes de Bonsai")
    ap.add_argument("--self-weight", type=float, default=1.0, help="part des pas rapides auto-etiquetes a garder (0..1)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    examples, stats = convert(args.inp, not args.no_self, args.self_weight, args.seed)
    random.Random(args.seed).shuffle(examples)
    n_val = int(len(examples) * args.val_frac) if args.val else 0
    with open(args.out, "w", encoding="utf-8") as f:
        for ex in examples[n_val:]:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    if args.val:
        with open(args.val, "w", encoding="utf-8") as f:
            for ex in examples[:n_val]:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"{len(examples) - n_val} exemples -> {args.out}" + (f" ; {n_val} -> {args.val}" if args.val else "")
          + f" ; {stats['trajectories']} trajectoires, {stats['bonsai']} etiquettes Bonsai, {stats['self']} auto-etiquetees,"
          f" {stats['bad_lines']} lignes ignorees")
    return stats


if __name__ == "__main__":
    main()

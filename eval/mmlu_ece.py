"""Calibration sur MMLU (1 200 items, comme le chiffre annonce par TypeSafe : ECE 0,031) : accuracy, ECE, Brier,
puis temperature ajustee sur une moitie et mesuree sur l'autre (protocole de reflex). A lancer la ou Hugging Face
est accessible (Colab), contre un llama-server du clone.

    python -m eval.mmlu_ece --server http://127.0.0.1:8081 --n 1200 --out runs/mmlu_calibration.json
"""

from __future__ import annotations

import argparse
import json
import random

import numpy as np

from jev_clone.calibrate import fit_temperature, pad, report
from jev_clone.readout import Calibration, probs_to_logits, softmax


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8081")
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--permutations", type=int, default=1)
    ap.add_argument("--out", default="runs/mmlu_calibration.json")
    args = ap.parse_args(argv)

    from datasets import load_dataset
    from jev_clone.backend_llamacpp import LlamaCppBackend
    from jev_clone.engine import SystemOneEngine

    ds = load_dataset("cais/mmlu", "all", split="test")
    idx = list(range(len(ds))); random.Random(args.seed).shuffle(idx); idx = idx[: args.n]
    engine = SystemOneEngine(LlamaCppBackend(args.server, max_workers=4))
    L, y, subj = [], [], []
    for k, i in enumerate(idx):
        ex = ds[i]
        opts = {f"opt{j}": c for j, c in enumerate(ex["choices"])}
        r = engine.answer({"state": ex["question"], "permutations": args.permutations,
                           "questions": {"q": {"type": "choice", "instructions": "Which answer is correct?", "criteria": opts}}})
        p = np.array([r.answers["q"].probabilities[f"opt{j}"] for j in range(len(ex["choices"]))])
        L.append(probs_to_logits(p)); y.append(int(ex["answer"])); subj.append(ex.get("subject", ""))
        if (k + 1) % 100 == 0:
            print(f"{k+1}/{len(idx)}")
    L, y = pad(L), np.array(y)
    half = len(y) // 2
    print("== brut (tout) ==\n", report(np.stack([softmax(l) for l in L]), y))
    T = fit_temperature(L[:half], y[:half])
    print(f"== temperature T={T:.3f} ajustee sur la 1re moitie, mesuree sur la 2e ==\n",
          report(np.stack([softmax(l, T) for l in L[half:]]), y[half:]))
    print("== reference : Jev annonce ECE 0,031 sur 1 200 items ; reflex Qwen3.5-4B : 0,039 apres temperature ==")
    Calibration(temperature={"noul": T, "choice": T, "score": T}).save(args.out)


if __name__ == "__main__":
    main()

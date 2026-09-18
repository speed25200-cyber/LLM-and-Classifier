"""CLI minimal.

    jev decide --server http://127.0.0.1:8081 --state "Ma carte a ete debitee deux fois" \
        --choice team=billing,technical,sales --noul angry="Le client est-il en colere ?" \
        --score urgency="peut attendre|aujourd'hui|bloquant"
    jev serve --port 8008
    jev bench --server http://127.0.0.1:8081 --n 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time


def _questions_from_args(args) -> dict:
    qs = {}
    for spec in args.choice or []:
        qid, opts = spec.split("=", 1)
        qs[qid] = {"type": "choice", "instructions": args.instructions.get(qid, qid), "criteria": opts.split(",")}
    for spec in args.noul or []:
        qid, text = spec.split("=", 1)
        qs[qid] = {"type": "noul", "instructions": text}
    for spec in args.score or []:
        qid, levels = spec.split("=", 1)
        qs[qid] = {"type": "score", "instructions": args.instructions.get(qid, qid), "criteria": levels.split("|")}
    return qs


def main(argv=None):
    ap = argparse.ArgumentParser(prog="jev")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("decide")
    d.add_argument("--server", default="http://127.0.0.1:8081")
    d.add_argument("--state", required=True)
    d.add_argument("--choice", action="append")
    d.add_argument("--noul", action="append")
    d.add_argument("--score", action="append")
    d.add_argument("--instruction", action="append", default=[], help="qid=texte de la question")
    d.add_argument("--calibration")
    d.add_argument("--permutations", type=int, default=1)

    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8008)

    b = sub.add_parser("bench")
    b.add_argument("--server", default="http://127.0.0.1:8081")
    b.add_argument("--n", type=int, default=20)

    args = ap.parse_args(argv)
    if args.cmd == "serve":
        import uvicorn
        uvicorn.run("jev_clone.server:app", host=args.host, port=args.port)
        return

    from jev_clone.backend_llamacpp import LlamaCppBackend
    from jev_clone.engine import SystemOneEngine
    from jev_clone.readout import Calibration

    if args.cmd == "decide":
        args.instructions = dict(kv.split("=", 1) for kv in args.instruction)
        engine = SystemOneEngine(LlamaCppBackend(args.server), calibration=Calibration.load(args.calibration))
        req = {"state": args.state, "questions": _questions_from_args(args), "permutations": args.permutations}
        print(json.dumps(engine.answer(req).model_dump(), ensure_ascii=False, indent=2))
        return

    if args.cmd == "bench":
        engine = SystemOneEngine(LlamaCppBackend(args.server))
        req = {"state": "Customer says: my payouts failed three times this week and nobody replied to my emails.",
               "questions": {"team": {"type": "choice", "instructions": "Which team should handle this?",
                                      "criteria": {"payments": "payouts, refunds", "account": "login, 2FA", "other": None}},
                             "escalate": {"type": "noul", "instructions": "Should this be escalated to a manager?"},
                             "urgency": {"type": "score", "instructions": "How urgent is this?",
                                         "criteria": ["can wait a week", "handle today", "blocked right now"]}}}
        engine.answer(req)  # echauffement : remplit le cache du prefixe
        lat = []
        for _ in range(args.n):
            t0 = time.perf_counter(); engine.answer(req); lat.append((time.perf_counter() - t0) * 1000)
        lat.sort()
        print(f"n={args.n}  p50={lat[len(lat)//2]:.1f} ms  p95={lat[int(len(lat)*0.95)-1]:.1f} ms  "
              f"min={lat[0]:.1f} ms  (3 questions, etat ~40 tokens, cache chaud)")


if __name__ == "__main__":
    sys.exit(main())

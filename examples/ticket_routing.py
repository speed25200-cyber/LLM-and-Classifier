"""Routage de tickets : System One decide, Bonsai n'est appele que si la confiance est insuffisante.

    export JEV_S1_URL=http://127.0.0.1:8081 JEV_S2_URL=http://127.0.0.1:8080
    python examples/ticket_routing.py
"""
import json
import os

from jev_clone.backend_llamacpp import LlamaCppBackend
from jev_clone.engine import SystemOneEngine
from jev_clone.fusion import FusionRouter, GatePolicy
from jev_clone.readout import load_calibration

S1 = os.environ.get("JEV_S1_URL", "http://127.0.0.1:8081")
S2 = os.environ.get("JEV_S2_URL")   # None = pas d'escalade (System One seul)

QUESTIONS = {
    "team": {"type": "choice", "instructions": "Which team should handle this ticket?",
             "criteria": {"billing": "charges, invoices, refunds", "technical": "bugs, outages, errors",
                          "sales": "pricing, upgrades, quotes", "other": "anything else"}},
    "refund_requested": {"type": "noul", "instructions": "Does the customer explicitly ask for a refund?"},
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                    "criteria": ["calm", "annoyed", "very frustrated / threatening to leave"]},
    "needs_human": {"type": "noul", "instructions": "Does this ticket require a human agent rather than an automated reply?"},
}
TICKETS = [
    "Hi, I was charged twice for order A-104 this month. Please refund the duplicate charge.",
    "The export button crashes in Safari but works fine in Chrome. Not urgent.",
    "This is the third time I write and nobody answers. Cancel everything, I'm done with you.",
    "Do you have a discount if we move 40 seats to the annual plan?",
]

engine = SystemOneEngine(LlamaCppBackend(S1), calibration=load_calibration(os.environ.get("JEV_CALIBRATION"), agent=False))
router = FusionRouter(engine, LlamaCppBackend(S2, max_workers=1) if S2 else None,
                      GatePolicy(default_threshold=0.6, risk_question="frustration",
                                 always_escalate_if={"needs_human": True}),
                      ledger="runs/ledger.jsonl")

for t in TICKETS:
    r = router.decide({"state": {"ticket": t}, "questions": QUESTIONS}, state_for_s2={"ticket": t})
    print(f"\n{t}\n  -> {r.path} ({r.latency_ms} ms)  decisions={json.dumps(r.decisions)}")
    for qid, a in r.s1.answers.items():
        print(f"     {qid:17s} {a.model_dump(exclude={'legend'})}")
    if r.s2_text:
        print("     Bonsai:", r.s2_text[:300].replace("\n", " "))

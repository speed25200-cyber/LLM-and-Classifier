"""Boucle d'agent temps reel (style demo Doom de Jev) : un etat de jeu structure -> une action typee
a chaque pas, en lisant les probabilites (aucune generation). Bonsai n'intervient que sur les etats
ou System One doute (escalade "lente", ici en simulation).

    JEV_S1_URL=http://127.0.0.1:8081 python examples/agent_loop.py --steps 30 --hz 10
"""
import argparse
import os
import random
import time

from jev_clone.backend_llamacpp import LlamaCppBackend
from jev_clone.engine import SystemOneEngine

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--hz", type=float, default=10.0)
args = ap.parse_args()

engine = SystemOneEngine(LlamaCppBackend(os.environ.get("JEV_S1_URL", "http://127.0.0.1:8081"), max_workers=2))
ACTIONS = {"shoot": "fire at the nearest enemy", "move_forward": "advance", "turn_left": None,
           "turn_right": None, "retreat": "back away from danger", "pick_up": "grab the nearby item"}

state = {"health": 73, "ammo": 4, "enemies": [{"type": "imp", "distance": 150, "angle": 12}], "items": []}
rng = random.Random(0)
lat = []
for step in range(args.steps):
    t0 = time.perf_counter()
    resp = engine.answer({"state": state, "questions": {
        "action": {"type": "choice", "instructions": "Best next action for the player right now?", "criteria": ACTIONS},
        "danger": {"type": "score", "instructions": "How dangerous is the situation?",
                   "criteria": ["safe", "threat nearby", "critical"]},
    }})
    ms = (time.perf_counter() - t0) * 1000
    lat.append(ms)
    a = resp.answers["action"]
    escalate = a.confidence < 0.3  # ici on ne fait que le signaler : brancher FusionRouter pour appeler Bonsai
    print(f"step {step:2d} {ms:6.1f} ms  action={a.choice:12s} conf={a.confidence:.2f} "
          f"danger={resp.answers['danger'].score:.2f}{'  [escalade S2]' if escalate else ''}")
    # simulation grossiere de l'environnement
    state["enemies"][0]["distance"] = max(10, state["enemies"][0]["distance"] - rng.randint(0, 40))
    state["health"] = max(1, state["health"] - rng.randint(0, 8))
    if rng.random() < 0.2:
        state["items"] = ["medikit at distance 30"]
    time.sleep(max(0.0, 1.0 / args.hz - ms / 1000))
lat.sort()
print(f"\np50={lat[len(lat)//2]:.1f} ms  p95={lat[int(len(lat)*0.95)-1]:.1f} ms  -> {1000/lat[len(lat)//2]:.1f} decisions/s possibles")

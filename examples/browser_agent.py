"""Agent navigateur a deux vitesses : le clone decide chaque pas, Bonsai reprend la main en cas de doute.

    export JEV_S1_URL=http://127.0.0.1:8081 JEV_S2_URL=http://127.0.0.1:8080
    python examples/browser_agent.py --url https://example.com --goal "Open the 'More information' link" --headed
    python examples/browser_agent.py --url https://duckduckgo.com --goal "Search for Bonsai 2 27B" --slot query="Bonsai 2 27B"
"""
import argparse
import json
import os

from jev_clone.backend_llamacpp import LlamaCppBackend
from jev_clone.computer_use import BrowserSession, ComputerUseAgent, FastPolicy, SlowPolicy
from jev_clone.engine import SystemOneEngine
from jev_clone.readout import Calibration
from jev_clone.tools import SystemOneToolbox

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--goal", required=True)
ap.add_argument("--slot", action="append", default=[], help="cle=valeur que le clone peut saisir")
ap.add_argument("--headed", action="store_true")
ap.add_argument("--vision", action="store_true", help="joindre une capture a Bonsai lors des escalades (mmproj requis)")
ap.add_argument("--max-steps", type=int, default=20)
args = ap.parse_args()

s1 = SystemOneEngine(LlamaCppBackend(os.environ.get("JEV_S1_URL", "http://127.0.0.1:8081"), max_workers=4),
                     calibration=Calibration.load(os.environ.get("JEV_CALIBRATION")))
s2_url = os.environ.get("JEV_S2_URL")
session = BrowserSession(headless=not args.headed, screenshot=args.vision)
slow = SlowPolicy(LlamaCppBackend(s2_url, max_workers=1), session, SystemOneToolbox(s1), vision=args.vision) if s2_url else None
agent = ComputerUseAgent(session, FastPolicy(s1), slow, max_steps=args.max_steps)
try:
    out = agent.run(args.goal, url=args.url, slots=dict(kv.split("=", 1) for kv in args.slot))
    print(json.dumps({"status": out["status"], "steps": out["steps"], "history": out["history"]}, ensure_ascii=False, indent=2))
    fast = [r["fast"]["ms"] for r in out["records"]]
    if fast:
        print(f"decisions rapides : {len(fast)}, latence moyenne {sum(fast)/len(fast):.0f} ms ; escalades : "
              f"{sum(1 for r in out['records'] if str(r.get('path','')).startswith('escalated'))}")
finally:
    session.close()

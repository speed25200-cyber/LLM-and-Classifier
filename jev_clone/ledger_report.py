"""Ce que la fusion fait economiser : lecture du journal (`runs/ledger.jsonl`, ecrit par FusionRouter).

    python -m jev_clone.ledger_report runs/ledger.jsonl --s2-tokens 900 --s2-ms 12000 --s2-cost 0.002

Pour chaque item : chemin (system_one / escalated), latence, tokens. Le rapport donne le taux d'escalade,
les latences p50/p95 par chemin, et l'economie estimee par rapport a "tout passe par le LLM" (tokens, temps,
cout) selon vos hypotheses de cout unitaire du chemin lent.
"""

from __future__ import annotations

import argparse
import json
import statistics


def summarize(rows: list[dict], s2_tokens: float = 900.0, s2_ms: float = 12000.0, s2_cost: float = 0.0) -> dict:
    n = len(rows)
    fast = [r for r in rows if r.get("path") == "system_one"]
    slow = [r for r in rows if str(r.get("path", "")).startswith("escalated")]
    lat_fast = sorted(r.get("latency_ms", 0.0) for r in fast)
    lat_slow = sorted(r.get("latency_ms", 0.0) for r in slow)
    pct = lambda v, q: v[max(0, min(len(v) - 1, int(len(v) * q) - 1))] if v else float("nan")
    per_q: dict[str, dict] = {}
    for r in rows:
        for qid, ok in (r.get("gated") or {}).items():
            d = per_q.setdefault(qid, {"n": 0, "gated": 0})
            d["n"] += 1; d["gated"] += bool(ok)
    return {
        "items": n, "fast": len(fast), "escalated": len(slow),
        "escalation_rate": (len(slow) / n) if n else float("nan"),
        "fast_p50_ms": statistics.median(lat_fast) if lat_fast else float("nan"), "fast_p95_ms": pct(lat_fast, 0.95),
        "slow_p50_ms": statistics.median(lat_slow) if lat_slow else float("nan"), "slow_p95_ms": pct(lat_slow, 0.95),
        "gate_rate_per_question": {q: (d["gated"] / d["n"]) for q, d in per_q.items()},
        "saved_llm_calls": len(fast),
        "saved_tokens_est": len(fast) * s2_tokens,
        "saved_time_s_est": len(fast) * s2_ms / 1000.0,
        "saved_cost_est": len(fast) * s2_cost,
        "assumptions": {"s2_tokens_per_item": s2_tokens, "s2_ms_per_item": s2_ms, "s2_cost_per_item": s2_cost},
    }


def render(s: dict) -> str:
    out = [f"items={s['items']}  rapides={s['fast']}  escalades={s['escalated']}  taux d'escalade={s['escalation_rate']:.1%}",
           f"latence rapide p50={s['fast_p50_ms']:.0f} ms p95={s['fast_p95_ms']:.0f} ms ; escalade p50={s['slow_p50_ms']:.0f} ms p95={s['slow_p95_ms']:.0f} ms",
           f"appels LLM evites={s['saved_llm_calls']}  tokens evites~{s['saved_tokens_est']:.0f}  temps evite~{s['saved_time_s_est']:.0f} s  cout evite~{s['saved_cost_est']:.4f}",
           "part des decisions prises par System One, par question :"]
    for q, r in sorted(s["gate_rate_per_question"].items()):
        out.append(f"  {q:24s} {r:6.1%}")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("ledger")
    ap.add_argument("--s2-tokens", type=float, default=900.0, help="tokens moyens d'un traitement LLM complet")
    ap.add_argument("--s2-ms", type=float, default=12000.0, help="latence moyenne d'un traitement LLM complet (ms)")
    ap.add_argument("--s2-cost", type=float, default=0.0, help="cout unitaire d'un traitement LLM complet (ex. API)")
    args = ap.parse_args(argv)
    rows = [json.loads(l) for l in open(args.ledger) if l.strip()]
    print(render(summarize(rows, args.s2_tokens, args.s2_ms, args.s2_cost)))


if __name__ == "__main__":
    main()

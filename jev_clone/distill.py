"""Distillation "System Two -> System One" : Bonsai 2 27B etiquette vos etats pour entrainer /
calibrer le petit clone Jev.

Deux modes d'etiquetage, combinables :
  * soft  : lecture System One sur Bonsai lui-meme (grammaire, une passe) -> distribution "enseignant"
            (rapide : pas de generation ; c'est le meme code que le clone, pointe sur le serveur Bonsai)
  * think : Bonsai raisonne (budget de reflexion) puis repond en JSON strict (json_schema) -> etiquette dure
            de meilleure qualite pour les cas ambigus (lent : generation)

    python -m jev_clone.distill --bonsai http://127.0.0.1:8080 --in data/states.jsonl --out data/labeled.jsonl \
        --mode soft --think-if-below 0.85

Entree : JSONL {"state": ..., "questions": {...}}  (labels optionnels, conserves)
Sortie : JSONL {"state", "questions", "labels", "teacher_probs", "teacher_mode"}
"""

from __future__ import annotations

import argparse
import json
import sys

from jev_clone.backend_llamacpp import LlamaCppBackend
from jev_clone.engine import SystemOneEngine
from jev_clone.prompt import render_text
from jev_clone.schema import SystemOneRequest


def _json_schema_for(req: SystemOneRequest) -> dict:
    props = {}
    for qid, q in req.questions.items():
        if q.type == "noul":
            props[qid] = {"type": "boolean"}
        elif q.type == "choice":
            keys = list(q.criteria.keys()) if isinstance(q.criteria, dict) else list(q.criteria)
            props[qid] = {"type": "string", "enum": keys}
        else:
            props[qid] = {"type": "integer", "minimum": 0, "maximum": len(q.criteria) - 1}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def think_labels(backend: LlamaCppBackend, req: SystemOneRequest, budget: int = 2048, max_tokens: int = 512) -> dict:
    """Etiquettes dures par raisonnement + sortie JSON contrainte (llama-server: response_format json_schema)."""
    qtxt = []
    for qid, q in req.questions.items():
        if q.type == "noul":
            qtxt.append(f"- {qid} (boolean): {render_text(q.instructions)}")
        elif q.type == "choice":
            opts = q.options()
            qtxt.append(f"- {qid} (one of {list(opts)}): {render_text(q.instructions)} "
                        + "; ".join(f"{k}: {render_text(v)}" for k, v in opts.items() if v))
        else:
            lv = "; ".join(f"{i}: {render_text(c)}" for i, c in enumerate(q.criteria))
            qtxt.append(f"- {qid} (integer level 0..{len(q.criteria)-1}): {render_text(q.instructions)} levels: {lv}")
    messages = [
        {"role": "system", "content": "You are an expert annotator. Answer every question about the state. Output only JSON."},
        {"role": "user", "content": f"# State\n{render_text(req.state)}\n\n# Questions\n" + "\n".join(qtxt)
                                    + "\n\nReturn a JSON object with exactly these keys."},
    ]
    schema = _json_schema_for(req)
    d = backend.chat(messages, max_tokens=max_tokens, thinking_budget=budget, temperature=0.0,
                     extra={"response_format": {"type": "json_schema", "json_schema": {"name": "labels", "schema": schema}}})
    content = d["choices"][0]["message"].get("content") or "{}"
    return json.loads(content)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Etiquetage par Bonsai (enseignant) pour le clone Jev")
    ap.add_argument("--bonsai", default="http://127.0.0.1:8080")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["soft", "think", "both"], default="soft")
    ap.add_argument("--think-if-below", type=float, default=None,
                    help="en mode soft : escalade vers 'think' si la confiance max < seuil")
    ap.add_argument("--budget", type=int, default=2048, help="budget de reflexion (tokens) en mode think")
    args = ap.parse_args(argv)

    backend = LlamaCppBackend(args.bonsai, max_workers=2)
    engine = SystemOneEngine(backend)
    n = 0
    with open(args.inp) as fi, open(args.out, "w") as fo:
        for line in fi:
            if not line.strip():
                continue
            ex = json.loads(line)
            req = SystemOneRequest(state=ex["state"], questions=ex["questions"])
            out = dict(ex)
            labels = dict(ex.get("labels", {}))
            mode = args.mode
            if mode in ("soft", "both"):
                resp = engine.answer(req)
                tp, low = {}, False
                for qid, a in resp.answers.items():
                    if a.type == "noul":
                        tp[qid] = {"true": a.noul, "false": 1 - a.noul}
                        labels.setdefault(qid, a.noul >= 0.5)
                        conf = max(a.noul, 1 - a.noul)
                    else:
                        tp[qid] = a.probabilities
                        labels.setdefault(qid, a.choice if a.type == "choice" else int(round(a.score)))
                        conf = max(a.probabilities.values())
                    if args.think_if_below is not None and conf < args.think_if_below:
                        low = True
                out["teacher_probs"] = tp
                out["teacher_mode"] = "soft"
                if low and mode == "soft":
                    mode = "think"
            if mode in ("think", "both"):
                hard = think_labels(backend, req, budget=args.budget)
                labels.update(hard)
                out["teacher_mode"] = "think" if args.mode != "both" else "both"
            out["labels"] = labels
            fo.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
            if n % 20 == 0:
                print(f"{n} exemples etiquetes", file=sys.stderr)
    print(f"termine : {n} exemples -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

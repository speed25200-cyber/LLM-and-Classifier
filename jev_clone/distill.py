"""Distillation "System Two -> System One" : Bonsai 2 27B etiquette vos etats pour entrainer /
calibrer le petit clone Jev.

Deux modes d'etiquetage, combinables :
  * soft  : lecture System One sur Bonsai lui-meme (grammaire, une passe) -> distribution "enseignant"
            (rapide : pas de generation ; c'est le meme code que le clone, pointe sur le serveur Bonsai)
  * think : Bonsai raisonne (budget de reflexion) puis repond en JSON strict (json_schema) -> etiquette dure
            de meilleure qualite pour les cas ambigus (lent : generation)

    python -m jev_clone.distill --teacher http://127.0.0.1:8080 --in data/states.jsonl --out data/labeled.jsonl \
        --mode soft --think-if-below 0.85 --workers 4 --resume

Entree : JSONL {"state": ..., "questions": {...}}  (labels optionnels)
Sortie : JSONL {"state", "questions", "labels", "teacher_probs", "teacher_mode"} (+ "teacher_disagrees", "teacher_error")

Etiquettes existantes :
  * exemple sans cle "weak" (etats a vous) : en soft, les etiquettes presentes sont gardees et Bonsai complete les autres ;
    en think, la reponse de Bonsai remplace tout (comportement historique) ;
  * exemple genere par regles (training/make_synthetic_prophet.py, cle "weak") : ses etiquettes font foi, sauf celles
    listees dans "weak" (estimations) que Bonsai remplace ; `teacher_disagrees` liste les questions ou Bonsai (soft)
    pense autrement que la regle (a relire : regle fausse ou enseignant faible).
Les distributions `teacher_probs` servent au terme KL de training/train_lora_rlcd.py (--kl).
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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


def _valid(q, v) -> bool:
    """Une etiquette de Bonsai n'est gardee que si elle est dans le domaine de la question (serveur sans json_schema...)."""
    if q.type == "noul":
        return isinstance(v, bool)
    if q.type == "choice":
        return isinstance(v, str) and v in q.options()
    return isinstance(v, int) and not isinstance(v, bool) and 0 <= v < len(q.criteria)


def label_example(engine, backend, ex: dict, mode: str = "soft", think_if_below: float | None = None, budget: int = 2048) -> dict:
    """Un exemple etiquete par l'enseignant (voir la politique des etiquettes existantes en tete du module)."""
    req = SystemOneRequest(state=ex["state"], questions=ex["questions"])
    rules = "weak" in ex
    weak = set(ex.get("weak") or [])
    labels = {k: v for k, v in (ex.get("labels") or {}).items() if not (rules and k in weak)}
    fixed = set(labels) if rules else set()
    out = dict(ex)
    step, guess = mode, {}
    if mode in ("soft", "both"):
        try:
            answers = engine.answer(req).answers
        except Exception as e:   # un etat refuse (contexte depasse...) ne bloque pas la suite ni la reprise (--resume)
            answers, step, out["teacher_error"] = {}, "none", f"{type(e).__name__}: {str(e)[:200]}"
        tp, low = {}, False
        for qid, a in answers.items():
            if a.type == "noul":
                tp[qid] = {"true": a.noul, "false": 1 - a.noul}
                guess[qid], conf = a.noul >= 0.5, max(a.noul, 1 - a.noul)
            else:
                tp[qid] = a.probabilities
                guess[qid], conf = (a.choice if a.type == "choice" else int(round(a.score))), max(a.probabilities.values())
            labels.setdefault(qid, guess[qid])
            if think_if_below is not None and conf < think_if_below and qid not in fixed:
                low = True
        if answers:
            out["teacher_probs"], out["teacher_mode"] = tp, "soft"
        if low and mode == "soft":
            step = "think"
    if step in ("think", "both"):
        try:
            hard = think_labels(backend, req, budget=budget)
            hard = {k: v for k, v in hard.items() if k in req.questions and k not in fixed and _valid(req.questions[k], v)}
        except Exception as e:   # JSON tronque ou serveur sans json_schema : on garde la lecture soft
            hard, out["teacher_error"] = {}, f"{type(e).__name__}: {str(e)[:200]}"
        labels.update(hard)
        if hard:
            out["teacher_mode"] = "both" if mode == "both" else "think"
    if rules:   # enseignant muet sur une estimation (think en echec) : l'estimation de la regle reste
        labels.update({k: v for k, v in (ex.get("labels") or {}).items() if k in weak and k not in labels})
        out["teacher_disagrees"] = sorted(q for q in fixed if q in guess and guess[q] != labels[q])
    out["labels"] = labels
    return out


def label_rows(rows, backend, mode: str = "soft", think_if_below: float | None = None, budget: int = 2048, workers: int = 1,
               engine=None, chunk: int = 64):
    """Etiquette une suite d'exemples (ordre conserve, `workers` exemples en parallele) ; generateur."""
    engine = engine or SystemOneEngine(backend)
    fn = lambda ex: label_example(engine, backend, ex, mode, think_if_below, budget)   # noqa: E731
    if workers <= 1:
        yield from map(fn, rows)
        return
    buf: list = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for ex in rows:
            buf.append(ex)
            if len(buf) >= chunk:
                yield from pool.map(fn, buf)
                buf = []
        yield from pool.map(fn, buf)


def check_teacher(url: str, timeout: float = 5.0) -> None:
    """Arret clair si l'enseignant ne repond pas (au lieu d'une erreur au premier exemple, apres le chargement)."""
    import requests
    try:
        r = requests.get(f"{url.rstrip('/')}/health", timeout=timeout)
        r.raise_for_status()
    except Exception as e:
        raise SystemExit(f"enseignant injoignable sur {url} ({type(e).__name__}) : lancez Bonsai 2 27B "
                         f"(scripts/start_bonsai.sh, ou Prophet Studio : serveur S2) puis relancez") from None


def _done_lines(path: Path) -> int:
    """Lignes completes deja ecrites (une ligne coupee par un arret brutal est retiree)."""
    if not path.exists():
        return 0
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        data = data[: data.rfind(b"\n") + 1]
        path.write_bytes(data)
    return data.count(b"\n")


def main(argv=None, backend=None):
    ap = argparse.ArgumentParser(description="Etiquetage par Bonsai 2 27B (enseignant) pour le clone Jev")
    ap.add_argument("--teacher", "--bonsai", dest="teacher", default="http://127.0.0.1:8080", help="URL du llama-server de Bonsai")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["soft", "think", "both"], default="soft")
    ap.add_argument("--think-if-below", type=float, default=None,
                    help="en mode soft : escalade vers 'think' si la confiance max < seuil")
    ap.add_argument("--budget", type=int, default=2048, help="budget de reflexion (tokens) en mode think")
    ap.add_argument("--workers", type=int, default=1, help="exemples etiquetes en parallele (<= slots du serveur, -np)")
    ap.add_argument("--resume", action="store_true", help="reprendre apres les lignes deja ecrites dans --out")
    args = ap.parse_args(argv)

    if backend is None:
        check_teacher(args.teacher)
        backend = LlamaCppBackend(args.teacher, max_workers=2, timeout=600)
    out = Path(args.out)
    skip = _done_lines(out) if args.resume else 0
    n = 0
    with open(args.inp, encoding="utf-8") as fi, open(out, "a" if skip else "w", encoding="utf-8") as fo:
        rows = (json.loads(l) for i, l in enumerate(x for x in fi if x.strip()) if i >= skip)
        for ex in label_rows(rows, backend, args.mode, args.think_if_below, args.budget, args.workers):
            fo.write(json.dumps(ex, ensure_ascii=False) + "\n")
            fo.flush()
            n += 1
            if n % 50 == 0:
                print(f"{skip + n} exemples etiquetes", file=sys.stderr)
    print(f"termine : {n} exemples etiquetes" + (f" ({skip} deja faits)" if skip else "") + f" -> {args.out}", file=sys.stderr)
    return n


if __name__ == "__main__":
    main()

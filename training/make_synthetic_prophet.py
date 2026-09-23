"""Jeu d'entrainement du clone pour Prophet, au format EXACT des appels de Prophet / Studio, sans aucune donnee a fournir.

    # regles seules (aucun reseau) : pre-tour, pertinence des outils, verification, voix
    python training/make_synthetic_prophet.py --out data/prophet_train.jsonl --val data/prophet_val.jsonl \
        --calib data/prophet_calib.jsonl --n 20000
    # + garde-fou : vos exemples risques (destructive / privileged / exfiltration, violations de politique)
    python training/make_synthetic_prophet.py ... --guard-extra data/guard_risky.jsonl
    # + Bonsai 2 27B enseignant (S2) : distributions pour le terme KL, estimations remplacees
    python training/make_synthetic_prophet.py ... --teacher http://127.0.0.1:8080 --workers 4
    # + graines etendues par Bonsai (demandes nouvelles de meme nature ; ancien mode de ce script)
    python training/make_synthetic_prophet.py ... --expand http://127.0.0.1:8080 --per-seed 20

Familles (training/prophet_data.py) : turn (PROPHET_TURN), guard (tool_risk, risk, policy_violation), tools (t_<outil>),
verify (ok), voice (intent). Sorties : --out (entrainement), --val (gabarits jamais vus a l'entrainement), --calib (pre-tour
tenu a l'ecart, forme d'etat de Prophet : fichier pour `python -m jev_clone.calibrate --data`, importable dans Studio).
Les graines livrees (jev_clone/seeds) ne sont jamais mises dans l'entrainement : le bouton Calibrer de Studio les lit.

Garde-fou : ce script n'ecrit que des actions benignes (lecture, ecriture dans le projet). Un garde entraine sans exemples
risques apprendrait que tout est benin : sans --guard-extra, la famille guard est retiree (--guard-benign-only pour la
forcer). Format de --guard-extra, une ligne par action jugee (memes questions que jev_clone.guard.JUDGE_QUESTIONS) :
    {"state": {"user_request": "...", "proposed_action": "shell: ..."},
     "labels": {"tool_risk": "destructive", "risk": 3, "policy_violation": true}}
Sources : vos refus dans Studio (journal runs/ledger.jsonl), vos regles internes, une relecture humaine.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

if __package__ in (None, ""):   # python training/make_synthetic_prophet.py : la racine du depot porte le paquet training
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jev_clone.guard import JUDGE_QUESTIONS
from jev_clone.prophet import PROPHET_TURN
from jev_clone.schema import SystemOneRequest

GEN_SCHEMA = {"type": "object", "properties": {"requests": {"type": "array", "items": {"type": "string"}, "minItems": 1}},
              "required": ["requests"], "additionalProperties": False}
SEEDS = Path(__file__).parent / "seeds" / "prophet_seeds.jsonl"


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


def expanded_rows(backend, seeds: list[dict], per_seed: int, budget: int = 0, seed: int = 0) -> list[dict]:
    """Demandes nouvelles par graine, dans la forme d'etat de Prophet ; intent / language / risk herites, le reste estime
    (weak) ou laisse a l'enseignant."""
    from training.prophet_data import _files, _recent
    rng, rows = random.Random(seed), []
    for i, s in enumerate(seeds):
        for j, r in enumerate(expand(backend, s, per_seed, budget)):
            labels = {k: s["labels"][k] for k in ("intent", "language", "risk") if k in s["labels"]}
            state = {"request": r, "workspace_files": _files(rng, None), "recent_turns": _recent(rng)}
            rows.append({"state": state, "questions": PROPHET_TURN, "labels": labels, "source": "bonsai_expand", "family": "turn",
                         "group": f"expand:{i}", "stratum": labels.get("intent", "?"), "weak": []})
        print(f"{i + 1}/{len(seeds)} graines etendues, {len(rows)} demandes", file=sys.stderr)
    return rows


def load_guard_extra(path: str) -> list[dict]:
    """Exemples de garde fournis par l'utilisateur ; lignes invalides refusees avec leur numero."""
    rows = []
    for n, line in enumerate(open(path, encoding="utf-8"), 1):
        if not line.strip():
            continue
        ex = json.loads(line)
        st, lab = ex.get("state"), ex.get("labels") or {}
        if not isinstance(st, dict) or not {"user_request", "proposed_action"} <= set(st):
            raise SystemExit(f"{path}:{n} : state doit contenir user_request et proposed_action (forme de jev_clone.guard)")
        req = SystemOneRequest(state=st, questions=JUDGE_QUESTIONS)
        for q, v in lab.items():
            if q not in req.questions or not _in_domain(req.questions[q], v):
                raise SystemExit(f"{path}:{n} : etiquette {q}={v!r} hors des options de JUDGE_QUESTIONS")
        rows.append({"state": st, "questions": JUDGE_QUESTIONS, "labels": lab, "source": ex.get("source", "user"), "family": "guard",
                     "group": f"guard:extra:{n}", "stratum": lab.get("tool_risk", "?"), "weak": []})
    return rows


def _in_domain(q, v) -> bool:
    from jev_clone.distill import _valid
    return _valid(q, v)


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv=None, teacher_backend=None, expand_backend=None):
    from training import prophet_data as P
    ap = argparse.ArgumentParser(description="Donnees d'entrainement du clone pour Prophet (etat et questions exacts de Prophet)")
    ap.add_argument("--out", required=True, help="JSONL d'entrainement")
    ap.add_argument("--val", help="JSONL de validation (gabarits tenus a l'ecart)")
    ap.add_argument("--calib", help="JSONL du pre-tour tenu a l'ecart, pour jev_clone.calibrate --data")
    ap.add_argument("--n", type=int, default=20000, help="exemples generes par regles (uniques, au plus)")
    ap.add_argument("--families", default=",".join(P.FAMILIES), help="familles parmi " + ",".join(P.FAMILIES))
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--guard-extra", help="JSONL de vos exemples de garde (classes risquees) ; voir l'en-tete du script")
    ap.add_argument("--guard-benign-only", action="store_true", help="garder la famille guard sans exemples risques (deconseille)")
    ap.add_argument("--expand", "--bonsai", dest="expand", help="URL de Bonsai : etendre les graines livrees en demandes nouvelles")
    ap.add_argument("--seeds", default=str(SEEDS))
    ap.add_argument("--per-seed", type=int, default=20)
    ap.add_argument("--budget", type=int, default=0, help="budget de reflexion de l'extension")
    ap.add_argument("--teacher", help="URL de Bonsai 2 27B (S2) : etiquetage enseignant de l'entrainement (jev_clone.distill)")
    ap.add_argument("--mode", choices=["soft", "think", "both"], default="soft")
    ap.add_argument("--think-if-below", type=float, default=None)
    ap.add_argument("--teacher-budget", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--distill-val", action="store_true", help="etiqueter aussi la validation par l'enseignant")
    args = ap.parse_args(argv)

    fams = [f.strip() for f in args.families.split(",") if f.strip()]
    bad = [f for f in fams if f not in P.FAMILIES]
    if bad:
        ap.error(f"familles inconnues : {bad}")
    extra = load_guard_extra(args.guard_extra) if args.guard_extra else []
    if "guard" in fams and not extra and not args.guard_benign_only:
        print("garde-fou : aucun exemple risque (--guard-extra) -> famille guard retiree (un garde qui ne voit que du benin "
              "perd le sens du danger) ; --guard-benign-only pour la garder", file=sys.stderr)
        fams.remove("guard")
    rows = P.generate(args.n, args.seed, fams) if fams else []
    if args.expand or expand_backend is not None:
        if expand_backend is None:
            from jev_clone.backend_llamacpp import LlamaCppBackend
            from jev_clone.distill import check_teacher
            check_teacher(args.expand)
            expand_backend = LlamaCppBackend(args.expand, max_workers=1, timeout=600)
        seeds = [json.loads(l) for l in open(args.seeds, encoding="utf-8") if l.strip()]
        rows += expanded_rows(expand_backend, seeds, args.per_seed, args.budget, args.seed)
    rows += extra
    train, val = P.split(rows, args.val_frac if (args.val or args.calib) else 0.0, args.seed)
    rng = random.Random(args.seed)
    rng.shuffle(train)
    rng.shuffle(val)
    if args.teacher or teacher_backend is not None:
        from jev_clone.distill import check_teacher, label_rows
        if teacher_backend is None:
            from jev_clone.backend_llamacpp import LlamaCppBackend
            check_teacher(args.teacher)
            teacher_backend = LlamaCppBackend(args.teacher, max_workers=2, timeout=600)
        todo = [("entrainement", train)] + ([("validation", val)] if args.distill_val else [])
        for name, part in todo:
            out = []
            for i, ex in enumerate(label_rows(part, teacher_backend, args.mode, args.think_if_below, args.teacher_budget, args.workers), 1):
                out.append(ex)
                if i % 200 == 0:
                    print(f"enseignant : {i}/{len(part)} ({name})", file=sys.stderr)
            part[:] = out
    write_jsonl(args.out, train)
    if args.val:
        write_jsonl(args.val, val)
    if args.calib:   # pre-tour seulement : une temperature ajustee sur d'autres familles s'appliquerait au garde ou a la voix
        calib = [{**r, "labels": {k: v for k, v in r["labels"].items() if k not in r.get("weak", []) or "teacher_probs" in r}}
                 for r in val if r.get("family") == "turn"]
        write_jsonl(args.calib, calib)
    c = P.counts(train)
    print(f"{len(train)} exemples -> {args.out}" + (f" ; {len(val)} -> {args.val}" if args.val else "")
          + (f" ; {len(calib)} (pre-tour) -> {args.calib}" if args.calib else ""), file=sys.stderr)
    print("par famille / classe : " + json.dumps(c, ensure_ascii=False), file=sys.stderr)
    g = c.get("guard", {})
    risky = sum(v for k, v in g.items() if k not in ("readonly", "workspace_write"))
    if g and risky < 0.3 * sum(g.values()):
        print(f"attention : {risky} exemples de garde risques sur {sum(g.values())} (< 30 %) : ajoutez-en dans --guard-extra",
              file=sys.stderr)
    return {"train": len(train), "val": len(val), "counts": c}


if __name__ == "__main__":
    main()

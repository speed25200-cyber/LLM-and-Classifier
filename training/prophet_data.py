"""Generateurs a base de regles des jugements que Prophet demande au clone (System One), au format EXACT de ses appels.

Chaque famille reproduit l'etat et les questions que Prophet / Studio envoient au S1 a l'execution :
  * turn   : pre-tour (jev_clone.prophet.PROPHET_TURN) sur {"request", "workspace_files", "recent_turns"}
  * guard  : juge des outils (jev_clone.guard.JUDGE_QUESTIONS : tool_risk, risk, policy_violation) sur
             {"user_request", "proposed_action"} ; actions BENIGNES seulement (readonly, workspace_write). Les exemples
             risques viennent d'un fichier que vous fournissez (make_synthetic_prophet.py --guard-extra) : un garde entraine
             sans eux perdrait le sens du danger, d'ou le refus par defaut de make_synthetic_prophet.py.
  * tools  : pertinence des outils (un noul `t_<outil>` par outil du catalogue reel de Prophet) sur {"request", "recent_turns"}
  * verify : verification d'une reponse ("ok") sur {"request", "response", "files"}
  * voice  : commandes vocales (prophet_studio.voice.COMMANDS + "prompt") sur {"utterance", "context"}

Ligne produite (format de training/train_lora_rlcd.py) :
  {"state", "questions", "labels", "source": "rules", "family", "group", "stratum", "weak": [...]}
`group` = gabarit d'origine : le decoupage entrainement / validation se fait par gabarit (aucun gabarit de validation n'est
vu a l'entrainement). `weak` = questions dont l'etiquette est une estimation (l'enseignant la remplace, jev_clone.distill).
Un outil "limite" pour une demande n'est pas etiquete du tout.
"""

from __future__ import annotations

import hashlib
import json
import random
import tempfile
from collections import defaultdict

from jev_clone.guard import JUDGE_QUESTIONS
from jev_clone.prophet import CORE_TOOLS, PROPHET_TURN, SHELL_NAME, Prophet, Workspace
from training import prophet_templates as T

FAMILIES = ("turn", "guard", "tools", "verify", "voice")
WEIGHTS = {"turn": 0.34, "guard": 0.22, "tools": 0.16, "verify": 0.12, "voice": 0.16}
VOICE_CONTEXT = "voice input of a desktop AI assistant app"          # prophet_studio.voice.route_utterance
VOICE_INSTRUCTIONS = "The user spoke this utterance. Is it one of the app control commands, or a prompt for the assistant?"
VOICE_PROMPT_DESC = "a question, request or task for the assistant itself (NOT an instruction to control the app)"
VERIFY_Q = {"ok": {"type": "noul", "instructions": "Does the response (and the workspace state) satisfy the user's request?"}}   # Prophet._verify
SHELLS = {"windows": "PowerShell (Windows)", "linux": "bash"}


# ---- questions exactes ---------------------------------------------------------------------------------------------------
def tool_question(name: str, desc: str) -> dict:
    """Meme consigne que Prophet._select_tools."""
    return {"type": "noul", "instructions": f"Would the tool `{name}` ({desc[:120]}) plausibly be useful for this request?"}


def voice_questions() -> dict:
    """Meme question que prophet_studio.voice.route_utterance."""
    from prophet_studio.voice import COMMANDS
    criteria = {c: s["desc"] for c, s in COMMANDS.items()}
    criteria["prompt"] = VOICE_PROMPT_DESC
    return {"intent": {"type": "choice", "criteria": criteria, "instructions": VOICE_INSTRUCTIONS}}


_CATALOG: dict | None = None


def tool_catalog(shell: str | None = None) -> dict[str, str]:
    """Outils soumis au jugement de pertinence (catalogue reel de Prophet, navigateur et bureau compris) -> description.
    shell : "windows" / "linux" = description de run_command sur cette machine-la (defaut : machine courante)."""
    global _CATALOG
    if _CATALOG is None:
        with tempfile.TemporaryDirectory() as d:
            p = Prophet(None, None, Workspace(d), browser_factory=lambda: None, desktop_factory=lambda: None)
            _CATALOG = {n: t["function"]["description"] for n, (t, _) in p._catalog("", []).items() if n not in CORE_TOOLS}
    out = dict(_CATALOG)
    if shell in SHELLS:
        out["run_command"] = out["run_command"].replace(SHELL_NAME, SHELLS[shell])
    return out


# ---- outils communs -------------------------------------------------------------------------------------------------------
def _h(s: str) -> float:
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF


def fill(rng: random.Random, text: str) -> str:
    for k, v in T.POOLS.items():
        while "{" + k + "}" in text:
            text = text.replace("{" + k + "}", rng.choice(v), 1)
    return text


def _files(rng: random.Random, kind: str | None) -> list[str]:
    fs = list(T.FILESETS[kind or rng.choice(list(T.FILESETS))])
    return fs[: rng.randint(max(1, len(fs) - 3), len(fs))] if fs else []


def _recent(rng: random.Random, p: float = 0.35) -> list[dict]:
    """Derniers echanges sans rapport direct (Prophet en montre 4 au plus, 400 caracteres chacun)."""
    if rng.random() >= p:
        return []
    q = fill(rng, rng.choice(["Explique-moi {topic}", "Crée {app}", "Merci", "Lance les tests"]))
    return [{"role": "user", "content": q}, {"role": "assistant", "content": rng.choice(["C'est fait.", "Voilà.", "Tests : tout passe.", "Bien noté."])}]


def _vary(rng: random.Random, text: str) -> str:
    """Variantes de saisie : casse, ponctuation finale, formule de politesse."""
    t = text[0].upper() + text[1:] if rng.random() < 0.6 else text[0].lower() + text[1:]
    if rng.random() < 0.15 and not t.endswith("?"):
        t += rng.choice([" stp", " s'il te plaît", " please"])
    if rng.random() < 0.3 and t[-1] not in ".?!":
        t += rng.choice([".", "!", " ?"]) if not t.lower().startswith(("qu", "wh", "com", "est")) else " ?"
    return t


def _row(family: str, group: str, stratum, state: dict, questions: dict, labels: dict, weak: list[str] | None = None) -> dict:
    return {"state": state, "questions": questions, "labels": labels, "source": "rules", "family": family, "group": f"{family}:{group}",
            "stratum": str(stratum), "weak": list(weak or [])}


# ---- familles -------------------------------------------------------------------------------------------------------------
def gen_turn(rng: random.Random) -> dict:
    if rng.random() < 0.12:
        g, turns, reqs, lab, fk = rng.choice(T.FOLLOWUPS)
        i = rng.randrange(len(reqs))
        state = {"request": _vary(rng, reqs[i]), "workspace_files": _files(rng, fk) if fk else [],
                 "recent_turns": [{"role": r, "content": c} for r, c in turns]}
        return _row("turn", f"{g}:{i}", lab["intent"], state, PROPHET_TURN, dict(lab), ["needs_reasoning"])
    g, intent, lang, direct, clarify, reason, risk, fk, phrasings = rng.choice(T.TURN)
    i = rng.randrange(len(phrasings))
    state = {"request": _vary(rng, fill(rng, phrasings[i])), "workspace_files": _files(rng, fk), "recent_turns": _recent(rng)}
    labels = {"direct": direct, "clarify": clarify, "intent": intent, "language": lang, "needs_reasoning": reason, "risk": risk}
    return _row("turn", f"{g}:{i}", intent, state, PROPHET_TURN, labels, ["needs_reasoning"])


def describe(kind: str, payload) -> str:
    """Texte juge, tel que Prophet._catalog le construit pour chaque outil."""
    if kind == "shell":
        return f"shell: {payload}"
    if kind == "python":
        return f"python code:\n{payload}"
    if kind == "write":
        path, content = payload
        return f"write {path} ({len(content)} chars):\n{content}"
    if kind == "browse":
        return f"web browser: {payload}"
    return f"control the computer desktop: {payload}"


def gen_guard(rng: random.Random) -> dict:
    # ~40 % d'exemples risques integres (GUARD_RISKY) : un juge qui ne voit que du benin perd le sens du danger
    if T.GUARD_RISKY and rng.random() < 0.4:
        g, cls, risk, policy, requests, actions = rng.choice(T.GUARD_RISKY)
    else:
        (g, cls, risk, requests, actions), policy = rng.choice(T.GUARD_BENIGN), False
    i = rng.randrange(len(actions))
    state = {"user_request": _vary(rng, rng.choice(requests)), "proposed_action": describe(*actions[i])}
    return _row("guard", f"{g}:{i}", cls, state, JUDGE_QUESTIONS, {"tool_risk": cls, "risk": risk, "policy_violation": policy})


def gen_tools(rng: random.Random) -> dict:
    cat = tool_catalog(rng.choice(["windows", "linux"]))
    names = [n for n in cat if not (n in ("browse", "desktop") and rng.random() < 0.5)]   # outils optionnels dans Studio (Reglages)
    skill = rng.choice(T.SKILLS) if rng.random() < 0.25 else None
    if skill is not None and rng.random() < 0.5:   # demande que seule la competence couvre
        cname, req, rel, irr = f"skill_{skill[2]}", rng.choice(T.SKILL_REQUESTS[skill[2]]), set(), set()
    else:
        cname = rng.choice(list(T.TOOL_CATS))
        phr, rel, irr = T.TOOL_CATS[cname]
        i = rng.randrange(len(phr))
        req, cname = fill(rng, phr[i]), f"{cname}:{i}"
    qs = {f"t_{n}": tool_question(n, cat[n]) for n in names}
    labels = {f"t_{n}": True for n in names if n in rel}
    labels.update({f"t_{n}": False for n in names if n in irr})
    if skill is not None:
        qs[f"t_{skill[0]}"] = tool_question(skill[0], skill[1])
        labels[f"t_{skill[0]}"] = cname == f"skill_{skill[2]}"
    return _row("tools", cname, cname.split(":")[0], {"request": _vary(rng, req), "recent_turns": _recent(rng)}, qs, labels)


def gen_verify(rng: random.Random) -> dict:
    r = rng.random()
    if r < 0.2:   # calcul : juste ou faux
        pct, x = rng.choice([5, 10, 15, 17, 20, 25, 30]), rng.choice([80, 120, 250, 1200, 2350, 4000])
        good = rng.random() < 0.5
        val = pct * x / 100 if good else pct * x / 100 + rng.choice([-10, 7, 25, 100])
        req = rng.choice([f"Combien font {pct} % de {x} ?", f"What is {pct}% of {x}?"])
        return _row("verify", "percent", good, {"request": req, "response": f"{pct} % de {x} = {val:g}", "files": []}, VERIFY_Q, {"ok": good})
    if r < 0.35:  # culture generale : juste ou erreur classique
        country, (ok_city, bad_city) = rng.choice(list(T.CAPITALS.items()))
        good = rng.random() < 0.5
        return _row("verify", "capital", good, {"request": f"Quelle est la capitale de {country} ?",
                                                 "response": f"La capitale est {ok_city if good else bad_city}.", "files": []}, VERIFY_Q, {"ok": good})
    g, req, resp, files, ok = rng.choice(T.VERIFY)
    extra = [f for f in _files(rng, None) if f not in files] if files and rng.random() < 0.3 else []
    return _row("verify", g, ok, {"request": _vary(rng, req), "response": resp, "files": sorted(files + extra)}, VERIFY_Q, {"ok": ok})


def gen_voice(rng: random.Random) -> dict:
    from prophet_studio.voice import grammar_match
    for _ in range(20):
        cmd = rng.choice(list(T.VOICE))
        i = rng.randrange(len(T.VOICE[cmd]))
        u = _vary(rng, T.VOICE[cmd][i])
        if not grammar_match(u):   # une phrase de la grammaire exacte n'atteint jamais le classifieur
            return _row("voice", f"{cmd}:{i}", cmd, {"utterance": u, "context": VOICE_CONTEXT}, voice_questions(), {"intent": cmd})
    raise RuntimeError("aucune paraphrase hors grammaire")


GENERATORS = {"turn": gen_turn, "guard": gen_guard, "tools": gen_tools, "verify": gen_verify, "voice": gen_voice}


# ---- assemblage -----------------------------------------------------------------------------------------------------------
def key(row: dict) -> str:
    return json.dumps([row["state"], sorted(row["questions"])], sort_keys=True, ensure_ascii=False)


def generate(n: int, seed: int = 0, families=FAMILIES, weights: dict | None = None) -> list[dict]:
    """n exemples uniques (au plus : les gabarits sont finis), familles tirees selon leurs poids."""
    rng = random.Random(seed)
    w = {f: (weights or WEIGHTS).get(f, 0.0) for f in families}
    fams, ps = list(w), [w[f] for f in w]
    out, seen = [], set()
    for _ in range(n * 30):
        if len(out) >= n:
            break
        row = GENERATORS[rng.choices(fams, ps)[0]](rng)
        k = key(row)
        if k not in seen:
            seen.add(k); out.append(row)
    return out


def split(rows: list[dict], frac: float = 0.1, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Validation = gabarits entiers jamais vus a l'entrainement, pris dans chaque (famille, classe) qui en a au moins 3."""
    groups: dict[tuple, set] = defaultdict(set)
    for r in rows:
        groups[(r.get("family"), r.get("stratum"))].add(r["group"])
    val: set = set()
    for gs in groups.values():
        if len(gs) >= 3 and frac > 0:
            val |= set(sorted(gs, key=lambda g: _h(f"{seed}:{g}"))[: max(1, round(len(gs) * frac))])
    return [r for r in rows if r["group"] not in val], [r for r in rows if r["group"] in val]


def counts(rows: list[dict]) -> dict:
    c: dict = defaultdict(lambda: defaultdict(int))
    for r in rows:
        c[r.get("family", "?")][r.get("stratum", "?")] += 1
    return {f: dict(v) for f, v in c.items()}

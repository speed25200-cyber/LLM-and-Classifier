"""Sondes du duo reel (S1 = classifieur, S2 = modele de raisonnement), partagees par eval/measure_duo.py et
tests/test_live_duo.py. Rien d'autre que le depot : requests, numpy, jev_clone. Fonctionne sous Windows.

  * s1_latency  : latence d'une decision par type de question, a froid (etat Prophet neuf, nonce en tete) et a chaud
  * s1_readout  : lecture des etiquettes avec la grammaire de Prophet et sans (masse que le modele met de lui-meme)
  * s2_speed    : generation et prefill (timings du serveur), temps jusqu'au premier token (flux)
  * s2_thinking : thinking_budget_tokens respecte (budget 0 puis 512 : tokens de reflexion comptes par /tokenize)
  * run_turn_local : un tour Prophet complet en processus (memes reglages que prophet_studio.sessions.AgentService)
  * VramSampler : VRAM avant / pic / apres (nvidia-smi, s'il existe)
"""

from __future__ import annotations

import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Callable

import requests

from jev_clone.backend_llamacpp import LlamaCppBackend, raise_for_status
from jev_clone.engine import SystemOneEngine
from jev_clone.guard import JUDGE_QUESTIONS
from jev_clone.presets import META
from jev_clone.prompt import label_grammar
from jev_clone.prophet import CORE_TOOLS, PROPHET_TURN, Prophet, Workspace
from jev_clone.readout import load_calibration
from jev_clone.schema import SystemOneRequest

DIRECT_QUESTION = "Quelle est la capitale de l'Italie ? Reponds en un mot."
AGENT_TASK = ("Cree dans l'espace de travail un fichier mesure.txt contenant une seule ligne : duo ok. "
              "Utilise l'outil write_file, puis termine avec done.")
WORDS = ("export upload retry batch timeout client record logger queue worker schema cache token parser config report invoice user "
         "session image thumbnail search index router handler service fixture build deploy docker readme api migration backup "
         "payment refund webhook cron metrics dashboard auth login locale theme").split()


# ---- statistiques ---------------------------------------------------------------------------------------------------
def pct(xs: list[float], q: float) -> float | None:
    """Percentile au rang le plus proche (p50 = mediane)."""
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    if q == 0.5:
        return round(statistics.median(xs), 1)
    return round(xs[max(0, min(len(xs) - 1, math.ceil(q * len(xs)) - 1))], 1)


def summary(xs: list[float]) -> dict:
    xs = [x for x in xs if x is not None]
    return {"n": len(xs), "p50": pct(xs, 0.5), "p95": pct(xs, 0.95), "mean": round(statistics.fmean(xs), 1) if xs else None,
            "min": round(min(xs), 1) if xs else None, "max": round(max(xs), 1) if xs else None}


def med(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 1) if xs else None


# ---- etats de la forme de ceux de Prophet ------------------------------------------------------------------------------
def _words(rng: random.Random, n: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n))


def _path(rng: random.Random, i: int) -> str:
    return f"src/{rng.choice(WORDS)}/{rng.choice(WORDS)}_{i}.py"


def prophet_state(rng: random.Random) -> dict:
    """Etat d'un pre-tour Prophet (prophet.handle : demande bornee a 2500 caracteres, 60 fichiers, 4 tours de 400), tire au
    hasard, nonce en tete : ni le cache du slot ni --cache-ram ne le connaissent, le serveur le lit en entier (~2 k tokens)."""
    req = f"[{uuid.uuid4().hex}] " + " ".join(f"Update {_path(rng, i)} so that {_words(rng, 9)}." for i in range(40))
    turns = [{"role": ("user", "assistant")[i % 2], "content": (f"Turn {i}: " + _words(rng, 80))[:400]} for i in range(4)]
    return {"request": req[:2500], "workspace_files": [_path(rng, i) for i in range(60)], "recent_turns": turns}


def tools_state(rng: random.Random) -> dict:
    s = prophet_state(rng)
    return {"request": s["request"][:2000], "recent_turns": s["recent_turns"]}


def guard_state(rng: random.Random) -> dict:
    """Etat du garde-fou (Prophet.judge) : demande + action proposee (un morceau de 2000 caracteres au plus)."""
    code = "\n".join(f"def {rng.choice(WORDS)}_{i}(x):\n    return x  # {_words(rng, 6)}" for i in range(60))[:1800]
    return {"user_request": f"[{uuid.uuid4().hex}] Create a small {_words(rng, 3)} module with tests.",
            "proposed_action": f"write {_path(rng, 0)} ({len(code)} chars):\n{code}"}


def tool_questions() -> dict:
    """Un noul par outil du catalogue reel de Prophet, meme libelle que Prophet._select_tools."""
    with tempfile.TemporaryDirectory(prefix="prophet-cat-") as d:
        cat = Prophet(None, None, Workspace(d))._catalog("", [])
    return {f"t_{n}": {"type": "noul", "instructions": f"Would the tool `{n}` ({cat[n][0]['function']['description'][:120]}) "
                                                      "plausibly be useful for this request?"} for n in cat if n not in CORE_TOOLS}


def s1_kinds() -> dict[str, tuple[dict, Callable[[random.Random], dict]]]:
    """Type de decision -> (questions, fabrique d'etat). Les trois primitives seules, puis les lectures reelles d'un tour."""
    return {"noul": ({"direct": PROPHET_TURN["direct"]}, prophet_state),
            "choice": ({"intent": PROPHET_TURN["intent"]}, prophet_state),
            "score": ({"risk": META["risk"]}, prophet_state),
            "pre_tour": (PROPHET_TURN, prophet_state),
            "outils": (tool_questions(), tools_state),
            "garde_fou": (JUDGE_QUESTIONS, guard_state)}


# ---- serveurs -----------------------------------------------------------------------------------------------------------
def server_info(url: str) -> dict:
    """/health et /props d'un llama-server : modele (nom du fichier), contexte par slot, slots, build."""
    out: dict = {"url": url.rstrip("/")}
    try:
        r = requests.get(f"{out['url']}/health", timeout=5)
        out["health"] = r.status_code == 200 and (r.json() or {}).get("status") == "ok"
    except Exception as e:
        return {**out, "health": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}
    try:
        p = requests.get(f"{out['url']}/props", timeout=5).json()
        dg = p.get("default_generation_settings") or {}
        out.update(model=os.path.basename(str(p.get("model_path") or p.get("model_alias") or "").replace("\\", "/")),
                   n_ctx=dg.get("n_ctx"), slots=p.get("total_slots"), build=p.get("build_info"))
    except Exception as e:
        out["error"] = f"/props : {type(e).__name__}: {str(e)[:200]}"
    return out


# ---- S1 : latence -------------------------------------------------------------------------------------------------------
def s1_latency(engine: SystemOneEngine, rng: random.Random, n_cold: int = 4, n_warm: int = 2,
               kinds: dict | None = None, log: Callable[[str], None] = print) -> dict:
    """Par type : n_cold etats neufs (lus en entier : a froid), chacun relu n_warm fois (prefixe en cache : a chaud).
    prefill_tokens = tokens reellement traites par le serveur (hors cache), toutes branches comprises."""
    engine.answer({"state": "warmup", "questions": {"q": PROPHET_TURN["direct"]}})   # premier appel apres demarrage : hors mesure
    out = {}
    for kind, (qs, make) in (kinds or s1_kinds()).items():
        cold, warm, cold_tok, warm_tok, branches = [], [], [], [], 0
        try:
            for _ in range(n_cold):
                req = {"state": make(rng), "questions": qs}
                for series, toks, reps in ((cold, cold_tok, 1), (warm, warm_tok, n_warm)):
                    for _ in range(reps):
                        t0 = time.perf_counter()
                        r = engine.answer(req)
                        series.append((time.perf_counter() - t0) * 1000)
                        toks.append(r.usage.question_tokens)
                        branches = r.usage.branches
        except Exception as e:
            out[kind] = {"questions": len(qs), "error": f"{type(e).__name__}: {str(e)[:300]}"}
            log(f"  S1 {kind} : ECHEC {out[kind]['error']}")
            continue
        c, w = summary(cold), summary(warm)
        tok = med(cold_tok)
        out[kind] = {"questions": len(qs), "branches": branches, "cold": {**c, "prefill_tokens": tok,
                     "prefill_tok_s": round(tok / c["p50"] * 1000, 1) if tok and c["p50"] else None},
                     "warm": {**w, "prefill_tokens": med(warm_tok)}}
        log(f"  S1 {kind:<9} ({len(qs)} q.) froid p50 {c['p50']} / p95 {c['p95']} ms ; chaud p50 {w['p50']} / p95 {w['p95']} ms")
    return out


# ---- S1 : lecture des etiquettes ----------------------------------------------------------------------------------------
def _top_probs(url: str, prompt: str, labels: list[str], grammar: bool, n_probs: int, timeout: float = 300) -> dict:
    """Un token sous grammaire (requete de lecture de Prophet, probabilites renormalisees) ou libre (distribution brute du
    modele, avant tout sampler). Masse des tokens qui designent une seule etiquette (prefixe) : avec la grammaire elle doit
    valoir 1 ; sans, c'est la part de la distribution que le modele met deja sur les etiquettes."""
    payload = {"prompt": prompt, "n_predict": 1, "n_probs": n_probs, "post_sampling_probs": grammar, "samplers": [], "temperature": 1.0,
               "cache_prompt": True}
    if grammar:
        payload["grammar"] = label_grammar(labels)
    r = requests.post(f"{url.rstrip('/')}/completion", json=payload, timeout=timeout)
    raise_for_status(r)
    cp = r.json().get("completion_probabilities") or []
    top = (cp[0].get("top_probs") or cp[0].get("top_logprobs") or []) if cp else []
    probs = [(str(e.get("token", "")), float(e["prob"]) if "prob" in e else math.exp(float(e.get("logprob", -1e9)))) for e in top]
    total, hit = sum(p for _, p in probs), {}
    for tok, p in probs:
        t = tok if grammar else tok.strip()   # libre : " A" ou "A\n" designent aussi l'etiquette
        match = [lab for lab in labels if t and lab.startswith(t)]
        if len(match) == 1:
            hit[match[0]] = hit.get(match[0], 0.0) + p
    mass = sum(hit.values())
    return {"n_probs": n_probs, "returned": len(probs), "sum": round(total, 4), "label_mass": round(mass, 4),
            "label_share": round(mass / total, 4) if total > 0 else 0.0, "labels_found": f"{len(hit)}/{len(labels)}",
            "all_labels_found": len(hit) == len(labels), "top1": probs[0][0] if probs else None}


def s1_readout(engine: SystemOneEngine, rng: random.Random) -> dict:
    """noul, choice, score sur un etat Prophet neuf : grammaire (somme 1, toutes les etiquettes presentes), libre (masse que le
    modele met de lui-meme sur les etiquettes, dans son top n_probs) et reponse typee (probabilites dans [0, 1], somme 1)."""
    url, out = engine.backend.base_url, {}
    for kind, qid, q in (("noul", "direct", PROPHET_TURN["direct"]), ("choice", "intent", PROPHET_TURN["intent"]), ("score", "risk", META["risk"])):
        try:
            req = SystemOneRequest.model_validate({"state": prophet_state(rng), "questions": {qid: q}})
            br = engine.branches(req)[0]
            prompt = engine.fmt.prefix(req.state) + br.text
            g = _top_probs(url, prompt, br.labels, True, max(16, 4 * len(br.labels)))
            f = _top_probs(url, prompt, br.labels, False, max(20, 2 * len(br.labels)))
            a = engine.answer(req).answers[qid]
            ps = [a.noul, 1.0 - a.noul] if kind == "noul" else list(a.probabilities.values())
            out[kind] = {"labels": len(br.labels), "grammar": g, "free": f, "answer_sum": round(sum(ps), 6),
                         "answer_in_01": all(0.0 <= p <= 1.0 for p in ps),
                         "ok": g["all_labels_found"] and abs(g["label_share"] - 1) < 1e-3 and abs(sum(ps) - 1) < 1e-3 and all(0 <= p <= 1 for p in ps)}
        except Exception as e:
            out[kind] = {"error": f"{type(e).__name__}: {str(e)[:300]}", "ok": False}
    return out


# ---- S2 ------------------------------------------------------------------------------------------------------------------
def rates(tm: dict) -> dict:
    """Debits depuis les timings de llama-server (prompt_per_second absent : prompt_n / prompt_ms)."""
    pn, pms, gn, gms = (tm.get(k) for k in ("prompt_n", "prompt_ms", "predicted_n", "predicted_ms"))
    pps = tm.get("prompt_per_second") or (pn / pms * 1000 if pn and pms else None)
    gps = tm.get("predicted_per_second") or (gn / gms * 1000 if gn and gms else None)
    return {"prompt_tokens": pn, "cached_tokens": tm.get("cache_n"), "prompt_tok_s": round(float(pps), 1) if pps else None,
            "gen_tokens": gn, "gen_tok_s": round(float(gps), 1) if gps else None}


def s2_stream(s2: LlamaCppBackend, content: str, max_tokens: int, thinking_budget: int = 0) -> dict:
    """Une requete diffusee, comme Prophet : temps jusqu'au premier token (reflexion ou texte) et debits du serveur."""
    first: list[float] = []

    def on_delta(d: dict) -> None:
        if not first and d.get("type") in ("content", "reasoning") and d.get("text"):
            first.append(time.perf_counter())
    t0 = time.perf_counter()
    r = s2.chat([{"role": "user", "content": content}], max_tokens=max_tokens, thinking_budget=thinking_budget, temperature=0.7, on_delta=on_delta)
    return {"ttft_ms": round((first[0] - t0) * 1000, 1) if first else None, "total_ms": round((time.perf_counter() - t0) * 1000, 1),
            "finish": r["choices"][0].get("finish_reason"), **rates(r.get("timings") or {})}


def long_prompt(rng: random.Random, n_lines: int = 110) -> str:
    """~1 500 tokens, nonce en tete (jamais en cache) : mesure du prefill."""
    lines = [f"{i:03d} {rng.choice(WORDS)}.{rng.choice(WORDS)} {_words(rng, 7)}" for i in range(n_lines)]
    return f"[{uuid.uuid4().hex}] Voici un journal d'application.\n" + "\n".join(lines) + "\nResume ce journal en une phrase."


def s2_speed(s2: LlamaCppBackend, rng: random.Random, n: int = 3, log: Callable[[str], None] = print) -> dict:
    gen, pre = [], []
    for _ in range(n):
        gen.append(s2_stream(s2, f"[{uuid.uuid4().hex[:8]}] Ecris un paragraphe d'environ 150 mots sur l'histoire des cartes marines.", 256))
        pre.append(s2_stream(s2, long_prompt(rng), 16))
    out = {"generation": {"tok_s": med(g["gen_tok_s"] for g in gen), "ttft_ms": med(g["ttft_ms"] for g in gen),
                          "gen_tokens": med(g["gen_tokens"] for g in gen)},
           "prefill": {"tok_s": med(p["prompt_tok_s"] for p in pre), "ttft_ms": med(p["ttft_ms"] for p in pre),
                       "prompt_tokens": med(p["prompt_tokens"] for p in pre)},
           "runs": {"generation": gen, "prefill": pre}}
    log(f"  S2 generation {out['generation']['tok_s']} tok/s (premier token {out['generation']['ttft_ms']} ms) ; "
        f"prefill {out['prefill']['tok_s']} tok/s sur {out['prefill']['prompt_tokens']} tokens (premier token {out['prefill']['ttft_ms']} ms)")
    return out


def _reasoning(msg: dict) -> tuple[str, str]:
    """(reflexion, reponse) ; une reflexion restee dans le texte (<think>...</think>) compte aussi."""
    think, text = msg.get("reasoning_content") or "", msg.get("content") or ""
    if "</think>" in text:
        head, text = text.split("</think>", 1)
        think += head.replace("<think>", "")
    return think, text


def s2_thinking(s2: LlamaCppBackend, budgets: tuple[int, ...] = (0, 512), log: Callable[[str], None] = print) -> dict:
    """thinking_budget_tokens (envoye comme le fait Prophet) respecte ? Tokens de reflexion comptes par le tokenizer du serveur ;
    tolerance : balise de fin forcee et re-tokenisation (max(16, 5 %))."""
    q = ("Un train part a 14 h 07 et roule 2 h 58 min, avec un arret de 13 min en chemin. "
         "A quelle heure arrive-t-il ? Donne seulement l'heure.")
    out: dict = {}
    for b in budgets:
        r = s2.chat([{"role": "user", "content": f"[{uuid.uuid4().hex[:8]}] {q}"}], max_tokens=b + 256, thinking_budget=b, temperature=0.6)
        think, text = _reasoning(r["choices"][0]["message"])
        ntok = len(s2.tokenize(think)) if think.strip() else 0
        tol = max(16, int(b * 0.05))
        out[str(b)] = {"reasoning_tokens": ntok, "honored": ntok == 0 if b == 0 else ntok <= b + tol, "tolerance": 0 if b == 0 else tol,
                       "finish": r["choices"][0].get("finish_reason"), "answer": " ".join(text.split())[:120], **rates(r.get("timings") or {})}
    out["honored"] = all(v["honored"] for v in out.values() if isinstance(v, dict))
    log("  S2 reflexion : " + " ; ".join(f"budget {b} -> {out[str(b)]['reasoning_tokens']} tokens" for b in budgets)
        + (" (respecte)" if out["honored"] else " (NON RESPECTE)"))
    return out


# ---- tour Prophet complet --------------------------------------------------------------------------------------------------
def list_files(root: Path) -> set[str]:
    root = Path(root)
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and ".prophet" not in p.relative_to(root).parts}


def decision_problems(pre: dict | None) -> list[str]:
    """Decision S1 du pre-tour : chaque question presente, probabilites dans [0, 1], distributions sommees a 1."""
    if not pre:
        return ["decision S1 absente"]
    if pre.get("s1_error"):
        return [f"S1 en panne : {pre['s1_error']}"]
    bad = []
    for q in PROPHET_TURN:
        a = pre.get(q)
        if not isinstance(a, dict):
            bad.append(f"{q} absente"); continue
        probs = a.get("probabilities") or {}
        vals = ([a["noul"]] if "noul" in a else []) + list(probs.values()) + ([a["confidence"]] if "confidence" in a else [])
        if not vals or any(not 0.0 <= float(v) <= 1.0 for v in vals):
            bad.append(f"{q} : probabilite hors de [0, 1]")
        if probs and abs(sum(probs.values()) - 1.0) > 1e-3:
            bad.append(f"{q} : probabilites de somme {sum(probs.values()):.4f}")
    bad += [f"pertinence de {n} hors de [0, 1]" for n, v in (pre.get("tool_relevance") or {}).items() if not 0.0 <= float(v) <= 1.0]
    return bad


def turn_checks(t: dict, agent: bool) -> dict:
    """Invariants de structure d'un tour (aucune exigence de qualite)."""
    c = {"s1_decision": not t.get("s1_problems"), "s1_counted": (t.get("s1_calls") or 0) >= 1 and (t.get("s1_ms") or 0) > 0,
         "s1_within_turn": (t.get("s1_ms") or 0) <= (t.get("latency_ms") or 0) + 1, "s2_called": (t.get("s2_calls") or 0) >= 1,
         "latency_within_wall": (t.get("latency_ms") or 0) <= (t.get("wall_ms") or 0) + 1,
         "finished": not t.get("error") and t.get("stopped_by") not in ("cancelled", "error")}
    if "events_llm_end" in t:
        c["events_match_stats"] = t["events_llm_end"] == t.get("s2_calls")
    if agent:
        c.update(tool_called=bool(t.get("tools")), file_created=bool(t.get("files_created")))
    return c


def _decision(s1: dict | None) -> dict | None:
    """Resume de la decision S1 d'un tour (evenement s1.decision ou transcription de Studio)."""
    if not s1:
        return None
    pre = s1.get("pre") or {}
    val = lambda q, k: (pre.get(q) or {}).get(k)   # noqa: E731
    return {"latency_ms": s1.get("latency_ms"), "path": s1.get("path"), "budget": s1.get("budget"), "risk_level": s1.get("risk_level"),
            "calibrated": s1.get("calibrated"), "direct": val("direct", "noul"), "needs_reasoning": val("needs_reasoning", "noul"),
            "clarify": val("clarify", "noul"), "intent": val("intent", "choice"), "risk": val("risk", "score"), "s1_error": pre.get("s1_error")}


def finish_turn(t: dict, s1: dict | None, agent: bool) -> dict:
    t["s1_decision"] = _decision(s1)
    t["s1_problems"] = decision_problems((s1 or {}).get("pre"))
    t["checks"] = turn_checks(t, agent)
    t["ok"] = all(t["checks"].values())
    return t


def run_turn_local(s1_url: str, s2_url: str, request: str, workspace: str | Path, effort: str = "auto",
                   budgets: tuple[int, ...] | None = None, calibration: str | None = None, timeout: float = 900.0,
                   agent: bool | None = None) -> dict:
    """Un tour Prophet en processus contre deux llama-server, permission_mode auto, flux d'evenements comme dans Studio.
    Memes reglages que prophet_studio.sessions.AgentService (max_tokens, compaction, 24 tours d'outils)."""
    s1 = SystemOneEngine(LlamaCppBackend(s1_url, max_workers=4, timeout=300), calibration=load_calibration(calibration, agent=True))
    s2 = LlamaCppBackend(s2_url, max_workers=1, timeout=timeout)
    ctx = max(4096, int(server_info(s2_url).get("n_ctx") or 8192))
    ws = Workspace(workspace)
    before = list_files(ws.root)
    events: list[tuple[float, dict]] = []
    t0 = time.perf_counter()
    deadline = lambda: time.perf_counter() - t0 > timeout   # noqa: E731
    p = Prophet(s1, s2, ws, permission_mode="auto", on_event=lambda e: events.append((round((time.perf_counter() - t0) * 1000, 1), e)),
                should_stop=deadline, max_turns=24, max_tokens=max(1024, min(8192, ctx // 3)), max_context_chars=int(ctx * 3.2 * 0.7),
                **({"budgets": tuple(budgets)} if budgets else {}))
    t: dict = {"mode": "local", "request": request, "effort": effort}
    try:
        turn = p.handle(request, [], effort=effort)
    except Exception as e:
        t.update(error=f"{type(e).__name__}: {str(e)[:300]}", wall_ms=round((time.perf_counter() - t0) * 1000, 1))
        return finish_turn(t, None, bool(agent))
    wall = round((time.perf_counter() - t0) * 1000, 1)
    st = turn.stats or {}
    dec = next((e for _, e in events if e.get("type") == "s1.decision"), None)
    t.update(path=turn.path, rerouted=(turn.pre or {}).get("rerouted"), wall_ms=wall, latency_ms=turn.latency_ms,
             ttft_ms=next((ms for ms, e in events if e.get("type") in ("text.delta", "thinking.delta")), None),
             s1_calls=st.get("s1_calls"), s1_ms=st.get("s1_ms"), s2_calls=st.get("llm_calls"), s2_tokens=st.get("tokens"),
             s2_tok_s=st.get("tok_s"), s2_prompt_ms=st.get("prompt_ms"), ctx_tokens=st.get("ctx_tokens"),
             events_llm_end=sum(1 for _, e in events if e.get("type") == "llm.end"),
             tools=[e.get("name") for _, e in events if e.get("type") == "tool.call"],
             files_created=sorted(list_files(ws.root) - before), verification=turn.verification, stopped_by=turn.stopped_by,
             response=" ".join((turn.response or "").split())[:300], error=None)
    return finish_turn(t, {**(dec or {}), "pre": turn.pre}, turn.path == "agent" if agent is None else agent)


def temp_workspace(prefix: str = "prophet-mesure-") -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix))


def remove_tree(p: Path) -> None:
    shutil.rmtree(p, ignore_errors=True)


# ---- VRAM -------------------------------------------------------------------------------------------------------------------
def nvidia_smi() -> str | None:
    smi = shutil.which("nvidia-smi")
    if not smi and sys.platform == "win32":
        cand = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvidia-smi.exe")
        smi = cand if os.path.exists(cand) else None
    return smi


def gpu_sample(smi: str | None) -> dict | None:
    """Premier GPU : nom, pilote, VRAM utilisee / totale (Mio) ; None sans nvidia-smi."""
    if not smi:
        return None
    try:
        out = subprocess.run([smi, "--query-gpu=name,driver_version,memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
        name, driver, used, total = [x.strip() for x in out.strip().splitlines()[0].split(",")[:4]]
        return {"gpu": name, "driver": driver, "used_mib": float(used), "total_mib": float(total)}
    except Exception:
        return None


class VramSampler:
    """VRAM avant / pic / apres, lue par nvidia-smi toutes les `interval` secondes pendant la mesure."""

    def __init__(self, interval: float = 1.0):
        self.smi, self.interval = nvidia_smi(), interval
        self.before: dict | None = None
        self.peak = 0.0
        self._stop = threading.Event()
        self._th: threading.Thread | None = None

    def start(self) -> "VramSampler":
        self.before = gpu_sample(self.smi)
        if self.before:
            self.peak = self.before["used_mib"]
            self._th = threading.Thread(target=self._loop, daemon=True, name="vram")
            self._th.start()
        return self

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            s = gpu_sample(self.smi)
            if s:
                self.peak = max(self.peak, s["used_mib"])

    def stop(self) -> dict:
        self._stop.set()
        if self._th:
            self._th.join(timeout=15)
        after = gpu_sample(self.smi)
        if not self.before or not after:
            return {"available": False, "note": "nvidia-smi absent : VRAM non mesuree"}
        return {"available": True, "gpu": after["gpu"], "driver": after["driver"], "total_mib": after["total_mib"],
                "before_mib": self.before["used_mib"], "peak_mib": max(self.peak, after["used_mib"]), "after_mib": after["used_mib"]}

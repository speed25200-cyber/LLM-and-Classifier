"""Verdict du garde-fou, une seule regle pour le juge des outils de Prophet et le garde par pas du computer use.

  * hard_stop : arret humain obligatoire (classe risquee en tete, masse risquee >= 0.5, politique, risque maximal,
    classifieur en panne) ; aucune autorisation memorisee ne vaut, « toujours » n'est jamais propose
  * needs_confirmation : hard_stop, ou verdict incertain (masse risquee >= danger_threshold, risque eleve)
  * grant_for : cle d'autorisation « toujours » (outil, classe jugee) ; jamais pour un arret obligatoire ni pour un
    verdict incertain (une autorisation ne couvre que ce que le mode smart aurait laisse passer sans demander)
"""

from __future__ import annotations

from typing import Callable

from jev_clone.presets import GUARDRAILS, RISKY_TOOL_CLASSES

JUDGE_QUESTIONS = {k: GUARDRAILS[k] for k in ("tool_risk", "risk", "policy_violation")}
HARD_P_RISKY, HARD_POLICY, HARD_RISK = 0.5, 0.5, 2.5


def outage_verdict(error) -> dict:
    """Classifieur injoignable : prudence, on demande et aucune autorisation memorisee ne vaut."""
    return {"s1_consulted": True, "tool_risk": "unknown", "tool_risk_conf": 0.0, "p_risky": 1.0, "risk": 2.0, "policy_violation": 0.0,
            "needs_confirmation": True, "hard_stop": True, "latency_ms": 0.0, "s1_error": str(error)[:200]}


def verdict(answers, danger_threshold: float = 0.35, risk_confirm_level: int = 2, latency_ms: float = 0.0) -> dict:
    """Reponses du clone (tool_risk, risk, policy_violation) -> verdict. La masse risquee compte, pas seulement l'argmax."""
    tr = answers["tool_risk"]
    p_risky = sum(tr.probabilities.get(c, 0.0) for c in RISKY_TOOL_CLASSES)
    j = {"s1_consulted": True, "tool_risk": tr.choice, "tool_risk_conf": tr.confidence, "p_risky": round(p_risky, 4),
         "risk": answers["risk"].score, "policy_violation": answers["policy_violation"].noul}
    j["hard_stop"] = (j["tool_risk"] in RISKY_TOOL_CLASSES or p_risky >= HARD_P_RISKY or j["policy_violation"] >= HARD_POLICY
                      or j["risk"] >= HARD_RISK)
    # incertitude : readonly + workspace_write doivent l'emporter avec une marge, l'argmax ne suffit pas
    j["needs_confirmation"] = j["hard_stop"] or p_risky >= danger_threshold or j["risk"] >= risk_confirm_level - 0.5
    j["latency_ms"] = latency_ms
    return j


def judge_state(ask: Callable[[dict], object], state: dict, danger_threshold: float = 0.35, risk_confirm_level: int = 2) -> dict:
    """Une passe du clone sur l'etat (user_request, proposed_action, ...) ; panne -> verdict d'arret obligatoire."""
    try:
        r = ask({"state": state, "questions": JUDGE_QUESTIONS})
    except Exception as e:
        return outage_verdict(e)
    return verdict(r.answers, danger_threshold, risk_confirm_level, r.latency_ms)


def is_hard(judged: dict) -> bool:
    """Arret obligatoire, recalcule aussi depuis les champs (un appelant qui oublierait le drapeau ne l'efface pas)."""
    return bool(judged.get("hard_stop") or judged.get("s1_error") or judged.get("partial") or judged.get("unseen")
                or judged.get("tool_risk") in RISKY_TOOL_CLASSES or (judged.get("p_risky") or 0) >= HARD_P_RISKY
                or (judged.get("policy_violation") or 0) >= HARD_POLICY or (judged.get("risk") or 0) >= HARD_RISK)


def grant_for(tool: str, judged: dict) -> str | None:
    """Cle « toujours » d'une demande : "outil:classe" pour un verdict du clone, "outil" pour une action non jugee
    (mode « toujours demander »), None si rien ne doit etre memorise (arret obligatoire ou verdict incertain)."""
    s1 = judged.get("s1_consulted", "tool_risk" in judged)   # un verdict sans le drapeau reste un verdict
    if is_hard(judged) or (s1 and judged.get("needs_confirmation", True)):
        return None
    if not s1:
        return tool
    return f"{tool}:{judged['tool_risk']}" if judged.get("tool_risk") else None

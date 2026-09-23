"""Agent "computer use" navigateur a deux vitesses (Playwright).

  observe() -> etat compact (url, titre, arbre ARIA, elements interactifs numerotes, capture optionnelle)
  FastPolicy (clone de Jev) : en UNE passe partagee, choisit l'action (choice), l'element cible (un noul par
      candidat pre-classe, en parallele), le slot a saisir (choice) et juge si l'objectif est deja atteint (noul).
      Le texte a saisir vient de "slots" structures (jamais du modele). Porte : probabilite de la meilleure option
      et marge sur la deuxieme (action, cible, slot) ; "done" n'est accepte que si l'objectif est juge atteint.
      Sous le seuil -> escalade.
  SlowPolicy (Bonsai 2 + outils) : planifie, agit via les outils click/type/scroll/back/done et peut
      consulter le clone (judge_*). Ses actions deviennent des etiquettes pour entrainer la FastPolicy (DAgger).
  StepGuard : chaque clic / saisie (et raccourci / lancement d'application au bureau) est juge par le clone avant
      execution (tool_risk, risk, policy_violation) ; un pas risque escalade et demande votre confirmation.
  verify : apres chaque action, un noul "l'action a-t-elle fait progresser l'objectif ?"

Budget par pas sur RTX 4060 (etat DOM, clone 0.8-1.7B) : observe ~50 ms + decision ~100-250 ms + garde ~100 ms + action.
Toute la trajectoire est journalisee (ledger) et convertible en exemples d'entrainement
(training/make_from_trajectories.py).
"""

from __future__ import annotations

import base64
import importlib.util
import json
import os
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from jev_clone.presets import GUARDRAILS, RISKY_TOOL_CLASSES
from jev_clone.tools import AgentLoop, SystemOneToolbox, _tool

ACTIONS = {
    "click": "click on the target element",
    "type": "type a value from the available slots into the target field",
    "scroll_down": "scroll the page down to reveal more content",
    "go_back": "go back to the previous page",
    "done": "the goal is achieved on the current page: stop",
    "escalate": "unclear, risky, or needs reading/reasoning: hand over to the reasoning model",
}
INTERACTIVE = ("a[href], button, input:not([type=hidden]), textarea, select, [role=button], [role=link], "
               "[role=textbox], [role=checkbox], [role=radio], [role=menuitem], [role=tab], [role=option], [onclick], [contenteditable=true]")

DEFAULT_CHROMIUM = [os.environ.get("JEV_CHROMIUM", ""), "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"]
BROWSER_HINT = "uv pip install playwright && playwright install chromium"

# questions de la politique rapide (partagees avec trajectory_to_examples : entrainement = inference)
ACTION_Q = {"type": "choice", "instructions": "What is the best next action to progress toward the goal?", "criteria": dict(ACTIONS)}
ACHIEVED_Q = {"type": "noul", "instructions": "Is the goal already fully achieved in the current state (nothing left to do)?"}
# garde-fou par pas : memes questions que le juge des outils de Prophet (tool_risk, risk, policy_violation)
STEP_RISK = {k: GUARDRAILS[k] for k in ("tool_risk", "risk", "policy_violation")}
GUARDED_ACTIONS = ("click", "type", "press_keys", "open_app")
FIELD_ROLES = ("input", "textarea", "textbox", "searchbox", "combobox", "document")


def target_question(line: str) -> dict:
    return {"type": "noul", "instructions": f"Is element {line} the right element to interact with for the next step?"}


def slot_question(slots) -> dict:
    return {"type": "choice", "instructions": "If typing is needed, which slot value should be typed?",
            "criteria": {**{k: None for k in slots}, "none": "no typing needed"}}


def _words(text: str) -> set[str]:
    t = unicodedata.normalize("NFD", text.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return {w for w in re.findall(r"[a-z0-9]+", t) if len(w) >= 3 or w.isdigit()}


def _top2(probs) -> tuple[float, float]:
    top = sorted((float(p) for p in probs), reverse=True) + [0.0, 0.0]
    return top[0], top[1]


def _playwright_dirs() -> list[Path]:
    env = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "")
    if env == "0":   # navigateurs installes dans le paquet lui-meme
        spec = importlib.util.find_spec("playwright")
        return [Path(spec.origin).parent / "driver" / "package" / ".local-browsers"] if spec and spec.origin else []
    if env:
        return [Path(env)]
    home = Path.home()
    return [home / ".cache" / "ms-playwright", home / "Library" / "Caches" / "ms-playwright",
            Path(os.environ.get("LOCALAPPDATA") or home / "AppData" / "Local") / "ms-playwright"]


def browser_available() -> tuple[bool, str]:
    """(disponible, raison) pour l'outil navigateur : Playwright importable et un Chromium installe (JEV_CHROMIUM, chemins
    connus ou cache de Playwright, marque INSTALLATION_COMPLETE). Aucun navigateur n'est lance : assez rapide pour /api/state."""
    if importlib.util.find_spec("playwright") is None:
        return False, f"Playwright n'est pas installe : {BROWSER_HINT}"
    if any(c and os.path.exists(c) for c in [os.environ.get("JEV_CHROMIUM", ""), *DEFAULT_CHROMIUM]):
        return True, ""
    for root in _playwright_dirs():
        try:
            if any((d / "INSTALLATION_COMPLETE").exists() for d in root.glob("chromium*")):
                return True, ""
        except OSError:
            continue
    return False, "navigateur Chromium introuvable : playwright install chromium"


@dataclass
class Element:
    index: int
    role: str
    name: str
    value: str = ""
    href: str = ""
    bbox: tuple[int, int, int, int] | None = None

    def line(self) -> str:
        v = f' value="{self.value[:40]}"' if self.value else ""
        h = f" -> {self.href[:60]}" if self.href else ""
        return f"[{self.index}] {self.role} \"{self.name[:60]}\"{v}{h}"


@dataclass
class PageState:
    url: str
    title: str
    aria: str
    elements: list[Element]
    screenshot_b64: str | None = None

    def text(self, max_aria_chars: int = 2500) -> dict:
        return {"url": self.url, "title": self.title, "page": self.aria[:max_aria_chars],
                "elements": [e.line() for e in self.elements]}


class BrowserSession:
    """Enveloppe Playwright synchrone. `chromium` : chemin de l'executable (None = navigateur Playwright)."""

    def __init__(self, headless: bool = True, chromium: str | None = None, max_elements: int = 40,
                 screenshot: bool = False, viewport=(1280, 800)):
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        exe = chromium
        if exe is None:
            for c in DEFAULT_CHROMIUM:
                if c and os.path.exists(c):
                    exe = c; break
        try:
            try:
                self.browser = self._pw.chromium.launch(headless=headless, executable_path=exe) if exe else self._pw.chromium.launch(headless=headless)
            except Exception:
                self.browser = self._pw.chromium.launch(headless=headless)
            self.page = self.browser.new_page(viewport={"width": viewport[0], "height": viewport[1]})
        except Exception:
            # sans cet arret, la boucle de Playwright reste active dans ce fil et tout sync_playwright() suivant echoue
            self._pw.stop()
            raise
        self.max_elements = max_elements
        self.screenshot = screenshot
        self._locators: list = []
        self.last: PageState | None = None   # derniere observation : les indices des actions s'y rapportent

    def close(self):
        try:
            self.browser.close()
        finally:
            self._pw.stop()

    def goto(self, url: str):
        self.page.goto(url, wait_until="domcontentloaded")

    def set_content(self, html: str):
        self.page.set_content(html)

    def observe(self) -> PageState:
        page = self.page
        try:
            aria = page.locator("body").aria_snapshot()
        except Exception:
            aria = ""
        loc = page.locator(INTERACTIVE)
        n = loc.count()
        elements, self._locators = [], []
        for i in range(n):
            if len(elements) >= self.max_elements:
                break
            el = loc.nth(i)
            try:
                if not el.is_visible():
                    continue
                info = el.evaluate("""e => ({role: e.getAttribute('role') || e.tagName.toLowerCase(),
                    name: (e.getAttribute('aria-label') || e.innerText || e.getAttribute('placeholder') || e.getAttribute('name') || e.getAttribute('title') || e.value || '').trim().slice(0, 120),
                    value: ('value' in e && typeof e.value === 'string') ? e.value.slice(0, 80) : '',
                    href: e.getAttribute('href') || '', type: e.getAttribute('type') || ''})""")
                bb = el.bounding_box()
            except Exception:
                continue
            role = info["role"] + (f":{info['type']}" if info.get("type") else "")
            elements.append(Element(index=len(elements), role=role, name=info["name"], value=info["value"], href=info["href"],
                                    bbox=(int(bb["x"]), int(bb["y"]), int(bb["width"]), int(bb["height"])) if bb else None))
            self._locators.append(el)
        shot = base64.b64encode(page.screenshot(type="jpeg", quality=60)).decode() if self.screenshot else None
        self.last = PageState(url=page.url, title=page.title(), aria=aria, elements=elements, screenshot_b64=shot)
        return self.last

    def act(self, action: dict) -> dict:
        t = action.get("type")
        page = self.page
        try:
            if t == "click":
                self._locators[int(action["target"])].click(timeout=5000)
            elif t == "type":
                el = self._locators[int(action["target"])]
                el.click(timeout=5000); el.fill(str(action.get("text", "")))
                if action.get("submit"):
                    el.press("Enter")
            elif t == "scroll_down":
                page.mouse.wheel(0, 600)
            elif t == "go_back":
                page.go_back(wait_until="domcontentloaded")
            elif t == "goto":
                page.goto(action["url"], wait_until="domcontentloaded")
            elif t in ("done", "escalate"):
                return {"ok": True}
            else:
                return {"ok": False, "error": f"unknown action {t}"}
            page.wait_for_load_state("domcontentloaded", timeout=5000)
            page.wait_for_timeout(150)
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)[:200]}


# --------------------------------------------------------------------------------------------------
@dataclass
class FastDecision:
    action: str
    action_conf: float
    action_probs: dict[str, float]
    target: int | None
    target_prob: float
    slot: str | None
    escalate: bool
    latency_ms: float
    raw: dict = field(default_factory=dict)
    action_prob: float = 0.0          # probabilite de l'action retenue
    action_margin: float = 0.0        # ecart avec la deuxieme action
    target_margin: float = 0.0        # ecart entre les deux meilleurs elements
    achieved: float = 0.0             # noul "objectif deja atteint ?"
    why: str = ""                     # raisons de l'escalade (vide = voie rapide)


class FastPolicy:
    """System One : une requete /v1/systemone = action (choice) + objectif atteint (noul) + un noul par element candidat
    + slot (choice). La porte lit la probabilite de tete et la marge sur la deuxieme option, pas la confiance entropique
    (1 - entropie normalisee donne 0,61 a un 51/49 sur six actions : une quasi-egalite passait). "done" doit etre
    confirme par le noul `achieved`, sinon Bonsai reprend la main."""

    def __init__(self, engine, action_threshold: float = 0.5, action_margin: float = 0.2, target_threshold: float = 0.5,
                 target_margin: float = 0.15, done_threshold: float = 0.6, max_candidates: int = 24):
        self.engine = engine
        self.action_threshold, self.action_margin = action_threshold, action_margin
        self.target_threshold, self.target_margin = target_threshold, target_margin
        self.done_threshold = done_threshold
        self.max_candidates = max_candidates

    @staticmethod
    def state_for(goal: str, state: PageState, history: list[dict], slots: dict[str, str]) -> dict:
        return {"goal": goal, "available_slots": {k: (v if len(v) < 40 else v[:37] + "...") for k, v in slots.items()},
                "last_actions": history[-4:], **state.text()}

    def candidates(self, goal: str, state: PageState, slots: dict[str, str]) -> list[Element]:
        """Pre-classement gratuit (sans modele) : au-dela de max_candidates, on garde les elements qui partagent le plus de
        mots avec l'objectif et les noms de slots (plus les champs de saisie quand il y a des slots), puis l'ordre de la page."""
        els = state.elements
        if len(els) <= self.max_candidates:
            return list(els)
        words = _words(goal + " " + " ".join(slots))

        def score(e: Element) -> int:
            return len(words & _words(f"{e.name} {e.value} {e.href}")) + (1 if slots and e.role.split(":")[0] in FIELD_ROLES else 0)
        keep = sorted(els, key=lambda e: (-score(e), e.index))[: self.max_candidates]
        return sorted(keep, key=lambda e: e.index)

    def decide(self, goal: str, state: PageState, history: list[dict], slots: dict[str, str]) -> FastDecision:
        cands = self.candidates(goal, state, slots)
        questions: dict[str, Any] = {"action": ACTION_Q, "achieved": ACHIEVED_Q}
        for e in cands:
            questions[f"t{e.index}"] = target_question(e.line())
        if slots:
            questions["slot"] = slot_question(slots)
        resp = self.engine.answer({"state": self.state_for(goal, state, history, slots), "questions": questions})
        a = resp.answers["action"]
        p1, p2 = _top2(a.probabilities.values())
        ranked = sorted(((resp.answers[f"t{e.index}"].noul, e.index) for e in cands), key=lambda x: -x[0])
        best_p, best_t = ranked[0] if ranked else (0.0, None)
        t_margin = best_p - (ranked[1][0] if len(ranked) > 1 else 0.0)
        slot, slot_p = None, 0.0
        if slots:
            s = resp.answers["slot"]
            slot, slot_p = s.choice, float(s.probabilities.get(s.choice, 0.0))
        achieved = resp.answers["achieved"].noul
        needs_target = a.choice in ("click", "type")
        why = []
        if a.choice == "escalate":
            why.append("the fast policy asked for help")
        if p1 < self.action_threshold or p1 - p2 < self.action_margin:
            why.append(f"action not decisive: {a.choice} p={p1:.2f}, margin {p1 - p2:.2f}")
        if needs_target and (best_t is None or best_p < self.target_threshold or t_margin < self.target_margin):
            why.append(f"target uncertain: {best_t} p={best_p:.2f}, margin {t_margin:.2f}")
        if a.choice == "type" and (slot in (None, "none") or slot_p < self.action_threshold):
            why.append(f"no confident slot to type ({slot} p={slot_p:.2f})")
        if a.choice == "done" and achieved < self.done_threshold:
            why.append(f"done not confirmed: goal achieved p={achieved:.2f}")
        return FastDecision(action=a.choice, action_conf=a.confidence, action_probs=a.probabilities,
                            target=best_t if needs_target else None, target_prob=best_p,
                            slot=slot if a.choice == "type" else None, escalate=bool(why),
                            latency_ms=resp.latency_ms, raw={k: v.model_dump() for k, v in resp.answers.items()},
                            action_prob=p1, action_margin=p1 - p2, target_margin=t_margin, achieved=achieved, why="; ".join(why))

    def verify(self, goal: str, before: PageState, action: dict, after: PageState) -> float:
        st = {"goal": goal, "action": action, "before": {"url": before.url, "title": before.title},
              "after": after.text(max_aria_chars=1500)}
        r = self.engine.answer({"state": st, "questions": {"ok": {"type": "noul", "instructions": "Did the last action make progress toward the goal (no error, expected page/state)?"}}})
        return r.answers["ok"].noul


def describe_action(action: dict, state: PageState | None) -> str:
    """Action concrete en une ligne (pour le juge et pour la demande d'autorisation)."""
    t = action.get("type")
    el = ""
    if "target" in action:
        try:
            i = int(action["target"])
            el = state.elements[i].line() if state is not None and 0 <= i < len(state.elements) else f"[{i}] (unknown element)"
        except (TypeError, ValueError):
            el = f"[{action['target']}] (unknown element)"
    if t == "click":
        return f"click {el}"
    if t == "type":
        return (f"type {json.dumps(str(action.get('text', '')), ensure_ascii=False)[:300]} into {el}"
                + (" then press Enter" if action.get("submit") else ""))
    if t == "press_keys":
        return f"press the keyboard shortcut {str(action.get('keys', ''))[:80]}"
    if t == "open_app":
        return f"open the application or file {str(action.get('name', ''))[:200]}"
    return str(t)


class StepGuard:
    """Garde-fou par pas : le clone juge chaque clic, saisie, raccourci ou lancement d'application AVANT execution
    (~100 ms, memes questions que le juge des outils de Prophet). Pas risque : confirmation humaine via `confirm`
    (meme contrat que Prophet.confirm : describe, judged -> bool) ; sans `confirm`, le pas est refuse et signale.
    Classifieur indisponible : prudence, le pas est traite comme risque."""

    def __init__(self, engine, confirm: Callable[[str, dict], bool] | None = None, kind: str = "browse", risk_level: int = 2,
                 danger_threshold: float = 0.35):
        self.engine, self.confirm, self.kind, self.risk_level = engine, confirm, kind, risk_level
        self.danger_threshold = danger_threshold

    def judge(self, goal: str, action: dict, state: PageState | None) -> dict:
        describe = describe_action(action, state)
        st: dict[str, Any] = {"user_request": goal[:1200], "proposed_action": describe}
        if state is not None:
            st["where"] = {"url": state.url, "title": state.title}
        try:
            r = self.engine.answer({"state": st, "questions": STEP_RISK})
        except Exception as e:
            return {"tool_risk": "unknown", "tool_risk_conf": 0.0, "risk": 2.0, "policy_violation": 0.0, "needs_confirmation": True,
                    "latency_ms": 0.0, "s1_error": str(e)[:200], "describe": describe}
        a = r.answers
        # meme regle que le juge des outils de Prophet : la masse de probabilite risquee compte, pas seulement l'argmax
        p_risky = sum(a["tool_risk"].probabilities.get(c, 0.0) for c in RISKY_TOOL_CLASSES)
        j = {"tool_risk": a["tool_risk"].choice, "tool_risk_conf": a["tool_risk"].confidence, "p_risky": round(p_risky, 4),
             "risk": a["risk"].score, "policy_violation": a["policy_violation"].noul}
        j["needs_confirmation"] = (j["tool_risk"] in RISKY_TOOL_CLASSES or p_risky >= self.danger_threshold
                                   or j["risk"] >= self.risk_level - 0.5 or j["policy_violation"] >= 0.5)
        j["latency_ms"], j["describe"] = r.latency_ms, describe
        return j

    def check(self, goal: str, action: dict, state: PageState | None, judged: dict | None = None) -> dict:
        """-> {"allowed": bool, "judged": dict, "reason"?: str}. Ne demande a l'humain que si le pas est juge risque."""
        j = judged or self.judge(goal, action, state)
        if not j["needs_confirmation"]:
            return {"allowed": True, "judged": j}
        if self.confirm is None:
            return {"allowed": False, "judged": j,
                    "reason": "risky step refused: nobody can approve it here; find a safer way or stop with done and explain why"}
        try:
            ok = bool(self.confirm(f"{self.kind} step: {j['describe']}", {**j, "tool": f"{self.kind}_step", "preview": {"command": j["describe"]}}))
        except Exception:
            ok = False
        return {"allowed": ok, "judged": j, "asked": True, **({} if ok else {"reason": "the user declined this step"})}


class SlowPolicy:
    """System Two : Bonsai planifie avec des outils d'action + les outils de jugement du clone. Quand l'agent l'attache
    (attach), chaque clic / saisie passe par le garde-fou du pas et la boucle de Bonsai suit l'annulation."""

    def __init__(self, s2_backend, session: BrowserSession, toolbox: SystemOneToolbox | None = None,
                 thinking_budget: int = 2048, max_turns: int = 6, vision: bool = False):
        self.session = session
        self.vision = vision
        self.executed: list[dict] = []
        self.guard: StepGuard | None = None
        self.goal = ""
        self.emit: Callable[[dict], None] = lambda evt: None
        s = session

        def observe(a):
            return s.observe().text()
        def done(a):
            self.executed.append({"type": "done", "summary": a.get("summary", "")}); return {"__stop__": True, "summary": a.get("summary", "")}

        tools = {
            "click": (_tool("click", "Click the interactive element with this index (from the observed elements list).",
                            {"index": {"type": "integer"}}, ["index"]), lambda a: self.act({"type": "click", "target": a["index"]})),
            "type": (_tool("type", "Type text into the field with this index; submit=true presses Enter afterwards.",
                           {"index": {"type": "integer"}, "text": {"type": "string"}, "submit": {"type": "boolean"}}, ["index", "text"]),
                     lambda a: self.act({"type": "type", "target": a["index"], "text": a["text"], "submit": a.get("submit", False)})),
            "scroll_down": (_tool("scroll_down", "Scroll the page down.", {}, []), lambda a: self.act({"type": "scroll_down"})),
            "go_back": (_tool("go_back", "Go back to the previous page.", {}, []), lambda a: self.act({"type": "go_back"})),
            "observe": (_tool("observe", "Re-observe the page (url, title, ARIA tree, interactive elements) after your actions.", {}, []), observe),
            "done": (_tool("done", "Declare the goal achieved (or impossible) and stop.", {"summary": {"type": "string"}}, ["summary"]), done),
        }
        self.loop = AgentLoop(s2_backend, toolbox, tools, max_turns=max_turns, thinking_budget=thinking_budget)

    def attach(self, guard: StepGuard | None = None, should_stop: Callable[[], bool] | None = None,
               emit: Callable[[dict], None] | None = None) -> None:
        self.guard = guard
        if should_stop is not None:
            self.loop.should_stop = should_stop
        if emit is not None:
            self.emit = emit

    def act(self, act: dict, observe: bool = False) -> dict:
        """Execute une action de Bonsai. Clic, saisie, raccourci, lancement : juges d'abord ; refus -> rien n'est execute."""
        if self.guard is not None and act["type"] in GUARDED_ACTIONS:
            chk = self.guard.check(self.goal, act, getattr(self.session, "last", None))
            if not chk["allowed"]:
                self.executed.append({**act, "blocked": True})
                self.emit({"type": "computer.action", "path": "escalated", "action": act["type"], "blocked": True})
                return {"ok": False, "blocked": True, "reason": chk["reason"],
                        "judged": {k: chk["judged"][k] for k in ("tool_risk", "risk", "policy_violation") if k in chk["judged"]}}
        r = self.session.act(act)
        self.executed.append(act)
        self.emit({"type": "computer.action", "path": "escalated", "action": act["type"], "ok": r.get("ok", False)})
        return {**r, "observation": self.session.observe().text()} if observe else r

    def step(self, goal: str, state: PageState, history: list[dict], slots: dict[str, str], why: str) -> dict:
        self.executed, self.goal = [], goal
        content: list | str = (f"# Goal\n{goal}\n\n# Why you are called\n{why}\n\n# Slots you may type (never invent values)\n"
                               f"{json.dumps(slots)}\n\n# Recent actions\n{json.dumps(history[-6:])}\n\n# Page\n{json.dumps(state.text(), ensure_ascii=False)}\n\n"
                               "Act with the tools (click/type/scroll_down/go_back), re-observe if needed, call done when the goal is achieved. "
                               "Risky steps are checked first and may need the user's approval: if one is refused, find another way or call done "
                               "and explain. You may call judge_* tools for fast checks.")
        if self.vision and state.screenshot_b64:
            content = [{"type": "text", "text": content},
                       {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{state.screenshot_b64}"}}]
        res = self.loop.run([{"role": "system", "content": "You are the reasoning tier of a two-speed browser agent. Be precise and economical."},
                             {"role": "user", "content": content}])
        return {"actions": list(self.executed), "stopped_by": res.stopped_by, "content": res.content}


class ComputerUseAgent:
    """Boucle observe -> decide (clone) -> garde -> agit -> verifie, avec escalade vers Bonsai.

    confirm(describe, judged) : autorisation humaine des pas risques (meme contrat que Prophet.confirm) ; sans elle, un
    pas risque est refuse et signale. on_event(evt) : un evenement `computer.step` par pas (voie rapide ou escalade,
    probabilite, raison) et `computer.action` par action de Bonsai. should_stop() : annulation cooperative, transmise a
    la boucle de Bonsai. Escalades plafonnees (max_escalations) ; un Bonsai en panne arrete la boucle apres
    max_s2_errors echecs consecutifs. Une erreur du clone n'arrete jamais la tache : le pas part a Bonsai."""

    def __init__(self, session: BrowserSession, fast: FastPolicy, slow: SlowPolicy | None = None,
                 max_steps: int = 30, verify: bool = True, verify_threshold: float = 0.4,
                 ledger: str | Path | None = "runs/trajectories.jsonl", confirm: Callable[[str, dict], bool] | None = None,
                 on_event: Callable[[dict], None] | None = None, should_stop: Callable[[], bool] | None = None,
                 guard: StepGuard | bool = True, max_escalations: int = 8, max_s2_errors: int = 2, kind: str = "browse"):
        self.session, self.fast, self.slow = session, fast, slow
        self.max_steps, self.verify, self.verify_threshold = max_steps, verify, verify_threshold
        self.ledger = Path(ledger) if ledger else None
        self.guard = StepGuard(fast.engine, confirm, kind) if guard is True else (guard or None)
        self.on_event = on_event
        self.should_stop = should_stop or (lambda: False)
        self.max_escalations, self.max_s2_errors, self.kind = max_escalations, max_s2_errors, kind
        self.escalations = self.s2_errors = 0
        if slow is not None:
            slow.attach(self.guard, self.should_stop, self._emit)

    def _emit(self, evt: dict) -> None:
        if self.on_event is not None:
            try:
                self.on_event({"tool": self.kind, **evt})
            except Exception:   # une interface defaillante ne doit jamais casser l'agent
                pass

    def _escalate(self, goal: str, state: PageState, history: list[dict], slots: dict[str, str], why: str, rec: dict,
                  path: str) -> str | None:
        """Pas confie a Bonsai. Rend un statut de fin, ou None pour continuer."""
        rec["path"], rec["why"] = path, why
        if path != "escalated":   # Bonsai voit l'etat d'APRES l'action rapide : c'est lui qu'on etiquette (DAgger)
            rec["slow_state"] = self.fast.state_for(goal, state, history, slots)
        if self.slow is None:
            return "needs_reasoning"
        if self.escalations >= self.max_escalations:
            return "max_escalations"
        self.escalations += 1
        out = self.slow.step(goal, state, history, slots, why)
        rec["slow"] = out
        history.extend(out["actions"])
        self.s2_errors = self.s2_errors + 1 if out["stopped_by"] == "error" else 0
        if out["stopped_by"] == "stop_tool":
            return "done"
        if out["stopped_by"] == "cancelled" or self.should_stop():
            return "cancelled"
        if self.s2_errors >= self.max_s2_errors:
            return "s2_error"
        return None

    def run(self, goal: str, url: str | None = None, slots: dict[str, str] | None = None) -> dict:
        slots = slots or {}
        if url:
            self.session.goto(url)
        history: list[dict] = []
        records: list[dict] = []
        status, fails = "max_steps", 0
        self.escalations = self.s2_errors = 0
        for step in range(self.max_steps):
            if self.should_stop():
                status = "cancelled"; break
            state = self.session.observe()
            rec: dict[str, Any] = {"step": step, "url": state.url, "state": self.fast.state_for(goal, state, history, slots)}
            end, action = None, None
            try:
                d = self.fast.decide(goal, state, history, slots)
            except Exception as e:   # le clone n'est jamais un point de panne : Bonsai prend le pas
                d, why = None, f"fast policy unavailable ({str(e)[:160]})"
                rec["fast"] = {"error": str(e)[:200]}
            else:
                why = d.why
                rec["fast"] = {"action": d.action, "conf": d.action_conf, "p": round(d.action_prob, 4), "margin": round(d.action_margin, 4),
                               "target": d.target, "target_prob": d.target_prob, "slot": d.slot, "achieved": d.achieved,
                               "escalate": d.escalate, "ms": d.latency_ms}
            if d is not None and not d.escalate:
                action = {"type": d.action}
                if d.target is not None:
                    action["target"] = d.target
                if d.action == "type":
                    action["text"] = slots.get(d.slot, ""); action["submit"] = False
                if self.guard is not None and d.action in GUARDED_ACTIONS:
                    j = self.guard.judge(goal, action, state)
                    rec["guard"] = {k: v for k, v in j.items() if k != "describe"}
                    if j["needs_confirmation"]:
                        if self.slow is not None:   # pas risque : Bonsai le reexamine, et son action demandera l'autorisation
                            why, action = f"risky step proposed by the fast policy ({j['describe']}): double-check it", None
                        elif not self.guard.check(goal, action, state, j)["allowed"]:
                            history.append({**action, "blocked": True})
                            rec["path"], end, action = "blocked", "blocked", None
            if end is None and action is None:
                end = self._escalate(goal, state, history, slots, why, rec, "escalated")
            elif action is not None:
                rec["path"] = "fast"
                if action["type"] == "done":
                    end = "done"
                else:
                    before = state
                    r = self.session.act(action)
                    history.append({**action, "ok": r.get("ok", False)})
                    rec["result"] = r
                    if self.verify:
                        after = self.session.observe()
                        try:
                            p: float | None = self.fast.verify(goal, before, action, after)
                        except Exception as e:
                            p = None; rec["verify_error"] = str(e)[:200]
                        rec["verify"] = p
                        fails = fails + 1 if (p is None or p < self.verify_threshold or not r.get("ok")) else 0
                        if self.slow is not None and (fails >= 2 or p is None):
                            why = "fast policy unavailable for verification" if p is None else "two consecutive low-confidence verifications"
                            end = self._escalate(goal, after, history, slots, why, rec, "escalated_after_verify"); fails = 0
            records.append(rec)
            f = rec.get("fast") or {}
            self._emit({"type": "computer.step", "step": step, "path": rec.get("path"), "action": f.get("action"), "p": f.get("p"),
                        "target_prob": f.get("target_prob"), "ms": f.get("ms"), "why": rec.get("why"), "verify": rec.get("verify"),
                        "escalations": self.escalations, "error": f.get("error"),
                        "slow_actions": [a.get("type") for a in (rec.get("slow") or {}).get("actions", [])]})
            if end:
                status = end; break
        self._log(goal, status, records)
        summary = next((a.get("summary") for a in reversed(history) if a.get("type") == "done"), None)
        return {"status": status, "steps": len(records), "history": history, "records": records, "escalations": self.escalations,
                "fast_steps": sum(1 for r in records if r.get("path") == "fast"), "summary": summary,
                "blocked": [a for a in history if a.get("blocked")]}

    def _log(self, goal, status, records):
        if not self.ledger:
            return
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ledger, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "kind": self.kind, "goal": goal, "status": status, "records": records},
                               ensure_ascii=False, default=str) + "\n")


def run_result(out: dict) -> dict:
    """Resultat compact rendu a Bonsai par les outils browse / desktop."""
    r = {"ok": out["status"] == "done", "status": out["status"], "steps": out["steps"], "fast_steps": out.get("fast_steps", 0),
         "escalations": out.get("escalations", 0)}
    if out.get("summary"):
        r["summary"] = out["summary"]
    if out.get("blocked"):
        r["blocked_steps"] = [{k: v for k, v in a.items() if k != "text"} for a in out["blocked"][:5]]
    return r


# ---- DAgger : trajectoires -> exemples d'entrainement de la politique rapide ---------------------------------------------
def _as_int(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _slot_of(text: str, slots: dict[str, str]) -> str | None:
    """Slot dont la valeur (eventuellement tronquee dans l'etat : '...') est exactement le texte saisi."""
    for k, v in slots.items():
        if v == text or (v.endswith("...") and len(v) > 3 and text.startswith(v[:-3])):
            return k
    return None


def decision_example(st: dict, action: str, target: int | None = None, slot: str | None = None, achieved: bool = False,
                     source: str = "bonsai") -> dict:
    """Un exemple au format de training/train_lora_rlcd.py, avec les memes questions que FastPolicy.decide."""
    qs: dict[str, Any] = {"action": ACTION_Q, "achieved": ACHIEVED_Q}
    labels: dict[str, Any] = {"action": action, "achieved": bool(achieved)}
    for line in st.get("elements", []):
        idx = int(line.split("]")[0][1:])
        qs[f"t{idx}"] = target_question(line)
        labels[f"t{idx}"] = target == idx
    slots = st.get("available_slots") or {}
    if slots:
        qs["slot"] = slot_question(slots)
        labels["slot"] = slot or "none"
    return {"state": st, "questions": qs, "labels": labels, "source": source}


def trajectory_to_examples(traj: dict) -> list[dict]:
    """DAgger : les pas escalades (actions decidees par Bonsai) deviennent des exemples etiquetes pour la FastPolicy,
    sur l'etat que Bonsai a vu (`slow_state` quand il reprend apres une action rapide). Les pas rapides verifies avec
    succes sont gardes aussi (auto-etiquetage, poids a moderer a l'entrainement). Une action refusee par le garde-fou
    n'est jamais une etiquette."""
    out = []
    for rec in traj.get("records", []):
        if rec.get("path", "").startswith("escalated"):
            acts = [a for a in (rec.get("slow") or {}).get("actions") or [] if not a.get("blocked")]
            if not acts:
                continue
            a = acts[0]
            st = rec.get("slow_state") or rec["state"]
            action, slot = (a["type"] if a["type"] in ACTIONS else "escalate"), None
            if action == "type":
                slot = _slot_of(str(a.get("text", "")), st.get("available_slots") or {})
                if slot is None:   # texte libre : la politique rapide ne sait pas l'ecrire, la bonne decision est d'escalader
                    action = "escalate"
            target = _as_int(a.get("target")) if action in ("click", "type") else None
            out.append(decision_example(st, action, target, slot, a["type"] == "done", "bonsai"))
        elif rec.get("path") == "fast" and (rec.get("verify") or 0) >= 0.8 and (rec.get("result") or {}).get("ok"):
            f = rec["fast"]
            out.append(decision_example(rec["state"], f["action"], f.get("target"), f.get("slot"), False, "self"))
    return out

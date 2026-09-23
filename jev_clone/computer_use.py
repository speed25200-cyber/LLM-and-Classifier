"""Agent "computer use" navigateur a deux vitesses (Playwright).

  observe() -> etat compact (url, titre, arbre ARIA, elements interactifs numerotes, capture optionnelle)
  FastPolicy (clone de Jev) : en UNE passe partagee, choisit l'action (choice) et l'element cible
      (un noul par candidat, en parallele) ; le texte a saisir vient de "slots" structures (jamais du modele).
      Confiance sous le seuil -> escalade.
  SlowPolicy (Bonsai 2 + outils) : planifie, agit via les outils click/type/scroll/back/done et peut
      consulter le clone (judge_*). Ses actions deviennent des etiquettes pour entrainer la FastPolicy (DAgger).
  verify : apres chaque action, un noul "l'action a-t-elle fait progresser l'objectif ?"

Budget par pas sur RTX 4060 (etat DOM, clone 0.8-1.7B) : observe ~50 ms + decision ~100-250 ms + action.
Toute la trajectoire est journalisee (ledger) et convertible en exemples d'entrainement.
"""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
        return PageState(url=page.url, title=page.title(), aria=aria, elements=elements, screenshot_b64=shot)

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


class FastPolicy:
    """System One : une requete /v1/systemone = action (choice) + un noul par element candidat + slot (choice)."""

    def __init__(self, engine, action_threshold: float = 0.35, target_threshold: float = 0.5, max_candidates: int = 24):
        self.engine = engine
        self.action_threshold = action_threshold
        self.target_threshold = target_threshold
        self.max_candidates = max_candidates

    @staticmethod
    def state_for(goal: str, state: PageState, history: list[dict], slots: dict[str, str]) -> dict:
        return {"goal": goal, "available_slots": {k: (v if len(v) < 40 else v[:37] + "...") for k, v in slots.items()},
                "last_actions": history[-4:], **state.text()}

    def decide(self, goal: str, state: PageState, history: list[dict], slots: dict[str, str]) -> FastDecision:
        t0 = time.perf_counter()
        cands = state.elements[: self.max_candidates]
        questions: dict[str, Any] = {
            "action": {"type": "choice", "instructions": "What is the best next action to progress toward the goal?", "criteria": dict(ACTIONS)},
        }
        for e in cands:
            questions[f"t{e.index}"] = {"type": "noul", "instructions": f"Is element {e.line()} the right element to interact with for the next step?"}
        if slots:
            questions["slot"] = {"type": "choice", "instructions": "If typing is needed, which slot value should be typed?",
                                 "criteria": {**{k: None for k in slots}, "none": "no typing needed"}}
        resp = self.engine.answer({"state": self.state_for(goal, state, history, slots), "questions": questions})
        a = resp.answers["action"]
        best_t, best_p = None, 0.0
        for e in cands:
            p = resp.answers[f"t{e.index}"].noul
            if p > best_p:
                best_t, best_p = e.index, p
        slot = resp.answers["slot"].choice if slots else None
        needs_target = a.choice in ("click", "type")
        escalate = (a.choice == "escalate" or a.confidence < self.action_threshold
                    or (needs_target and (best_t is None or best_p < self.target_threshold))
                    or (a.choice == "type" and (slot in (None, "none"))))
        return FastDecision(action=a.choice, action_conf=a.confidence, action_probs=a.probabilities,
                            target=best_t if needs_target else None, target_prob=best_p,
                            slot=slot if a.choice == "type" else None, escalate=escalate,
                            latency_ms=resp.latency_ms, raw={k: v.model_dump() for k, v in resp.answers.items()})

    def verify(self, goal: str, before: PageState, action: dict, after: PageState) -> float:
        st = {"goal": goal, "action": action, "before": {"url": before.url, "title": before.title},
              "after": after.text(max_aria_chars=1500)}
        r = self.engine.answer({"state": st, "questions": {"ok": {"type": "noul", "instructions": "Did the last action make progress toward the goal (no error, expected page/state)?"}}})
        return r.answers["ok"].noul


class SlowPolicy:
    """System Two : Bonsai planifie avec des outils d'action + les outils de jugement du clone."""

    def __init__(self, s2_backend, session: BrowserSession, toolbox: SystemOneToolbox | None = None,
                 thinking_budget: int = 2048, max_turns: int = 6, vision: bool = False):
        self.session = session
        self.vision = vision
        self.executed: list[dict] = []
        s = session

        def click(a):
            r = s.act({"type": "click", "target": a["index"]}); self.executed.append({"type": "click", "target": a["index"]}); return r
        def type_(a):
            act = {"type": "type", "target": a["index"], "text": a["text"], "submit": a.get("submit", False)}
            r = s.act(act); self.executed.append(act); return r
        def scroll(a):
            r = s.act({"type": "scroll_down"}); self.executed.append({"type": "scroll_down"}); return r
        def back(a):
            r = s.act({"type": "go_back"}); self.executed.append({"type": "go_back"}); return r
        def observe(a):
            return s.observe().text()
        def done(a):
            self.executed.append({"type": "done", "summary": a.get("summary", "")}); return {"__stop__": True, "summary": a.get("summary", "")}

        tools = {
            "click": (_tool("click", "Click the interactive element with this index (from the observed elements list).",
                            {"index": {"type": "integer"}}, ["index"]), click),
            "type": (_tool("type", "Type text into the field with this index; submit=true presses Enter afterwards.",
                           {"index": {"type": "integer"}, "text": {"type": "string"}, "submit": {"type": "boolean"}}, ["index", "text"]), type_),
            "scroll_down": (_tool("scroll_down", "Scroll the page down.", {}, []), scroll),
            "go_back": (_tool("go_back", "Go back to the previous page.", {}, []), back),
            "observe": (_tool("observe", "Re-observe the page (url, title, ARIA tree, interactive elements) after your actions.", {}, []), observe),
            "done": (_tool("done", "Declare the goal achieved (or impossible) and stop.", {"summary": {"type": "string"}}, ["summary"]), done),
        }
        self.loop = AgentLoop(s2_backend, toolbox, tools, max_turns=max_turns, thinking_budget=thinking_budget)

    def step(self, goal: str, state: PageState, history: list[dict], slots: dict[str, str], why: str) -> dict:
        self.executed = []
        content: list | str = (f"# Goal\n{goal}\n\n# Why you are called\n{why}\n\n# Slots you may type (never invent values)\n"
                               f"{json.dumps(slots)}\n\n# Recent actions\n{json.dumps(history[-6:])}\n\n# Page\n{json.dumps(state.text(), ensure_ascii=False)}\n\n"
                               "Act with the tools (click/type/scroll_down/go_back), re-observe if needed, call done when the goal is achieved. "
                               "You may call judge_* tools for fast checks.")
        if self.vision and state.screenshot_b64:
            content = [{"type": "text", "text": content},
                       {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{state.screenshot_b64}"}}]
        res = self.loop.run([{"role": "system", "content": "You are the reasoning tier of a two-speed browser agent. Be precise and economical."},
                             {"role": "user", "content": content}])
        return {"actions": list(self.executed), "stopped_by": res.stopped_by, "content": res.content}


class ComputerUseAgent:
    def __init__(self, session: BrowserSession, fast: FastPolicy, slow: SlowPolicy | None = None,
                 max_steps: int = 30, verify: bool = True, verify_threshold: float = 0.4,
                 ledger: str | Path | None = "runs/trajectories.jsonl"):
        self.session, self.fast, self.slow = session, fast, slow
        self.max_steps, self.verify, self.verify_threshold = max_steps, verify, verify_threshold
        self.ledger = Path(ledger) if ledger else None

    def run(self, goal: str, url: str | None = None, slots: dict[str, str] | None = None) -> dict:
        slots = slots or {}
        if url:
            self.session.goto(url)
        history: list[dict] = []
        records: list[dict] = []
        status, fails = "max_steps", 0
        for step in range(self.max_steps):
            state = self.session.observe()
            d = self.fast.decide(goal, state, history, slots)
            rec = {"step": step, "url": state.url, "state": self.fast.state_for(goal, state, history, slots),
                   "fast": {"action": d.action, "conf": d.action_conf, "target": d.target, "target_prob": d.target_prob,
                            "slot": d.slot, "escalate": d.escalate, "ms": d.latency_ms}}
            if d.escalate or (self.slow is None and d.action == "escalate"):
                if self.slow is None:
                    status = "needs_reasoning"; records.append(rec); break
                why = (f"fast policy not confident: action={d.action} ({d.action_conf:.2f}), target={d.target} ({d.target_prob:.2f})")
                out = self.slow.step(goal, state, history, slots, why)
                rec["path"], rec["slow"] = "escalated", out
                history.extend(out["actions"])
                records.append(rec)
                if out["stopped_by"] == "stop_tool":
                    status = "done"; break
                continue
            action = {"type": d.action}
            if d.target is not None:
                action["target"] = d.target
            if d.action == "type":
                action["text"] = slots.get(d.slot, ""); action["submit"] = False
            rec["path"] = "fast"
            if d.action == "done":
                records.append(rec); status = "done"; break
            before = state
            r = self.session.act(action)
            action_log = {**action, "ok": r.get("ok", False)}
            history.append(action_log)
            rec["result"] = r
            if self.verify:
                after = self.session.observe()
                p = self.fast.verify(goal, before, action, after)
                rec["verify"] = p
                fails = fails + 1 if (p < self.verify_threshold or not r.get("ok")) else 0
                if fails >= 2 and self.slow is not None:
                    out = self.slow.step(goal, after, history, slots, "two consecutive low-confidence verifications")
                    rec["path"], rec["slow"] = "escalated_after_verify", out
                    history.extend(out["actions"]); fails = 0
                    if out["stopped_by"] == "stop_tool":
                        records.append(rec); status = "done"; break
            records.append(rec)
        self._log(goal, status, records)
        return {"status": status, "steps": len(records), "history": history, "records": records}

    def _log(self, goal, status, records):
        if not self.ledger:
            return
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ledger, "a") as f:
            f.write(json.dumps({"ts": time.time(), "goal": goal, "status": status, "records": records}, ensure_ascii=False, default=str) + "\n")


def trajectory_to_examples(traj: dict) -> list[dict]:
    """DAgger : les pas escalades (actions decidees par Bonsai) deviennent des exemples etiquetes pour la FastPolicy.
    Les pas rapides verifies avec succes sont gardes aussi (auto-etiquetage, poids a moderer a l'entrainement)."""
    out = []
    for rec in traj.get("records", []):
        st = rec["state"]
        if rec.get("path", "").startswith("escalated"):
            acts = rec.get("slow", {}).get("actions") or []
            if not acts:
                continue
            a = acts[0]
            labels = {"action": a["type"] if a["type"] in ACTIONS else "escalate"}
            qs = {"action": {"type": "choice", "instructions": "What is the best next action to progress toward the goal?", "criteria": dict(ACTIONS)}}
            for line in st.get("elements", []):
                idx = int(line.split("]")[0][1:])
                qs[f"t{idx}"] = {"type": "noul", "instructions": f"Is element {line} the right element to interact with for the next step?"}
                labels[f"t{idx}"] = (a.get("target") == idx)
            out.append({"state": st, "questions": qs, "labels": labels, "source": "bonsai"})
        elif rec.get("path") == "fast" and rec.get("verify", 0) >= 0.8 and rec.get("result", {}).get("ok"):
            f = rec["fast"]
            labels = {"action": f["action"]}
            qs = {"action": {"type": "choice", "instructions": "What is the best next action to progress toward the goal?", "criteria": dict(ACTIONS)}}
            for line in st.get("elements", []):
                idx = int(line.split("]")[0][1:])
                qs[f"t{idx}"] = {"type": "noul", "instructions": f"Is element {line} the right element to interact with for the next step?"}
                labels[f"t{idx}"] = (f.get("target") == idx)
            out.append({"state": st, "questions": qs, "labels": labels, "source": "self"})
    return out

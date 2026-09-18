"""Jarvis : assistant personnel local a deux vitesses, qui cree des applications et execute des demandes.

A chaque tour :
  1. System One (clone) pre-traite la demande en une passe : intention, langage, ambiguite, besoin de
     raisonnement, risque (~100-250 ms).
  2. Selon la porte : reponse courte directe (Bonsai sans reflexion), question de clarification, ou
     **boucle d'agent** : Bonsai planifie et agit avec des outils (fichiers, commandes, navigateur, memoire,
     jugements du clone), budget de reflexion proportionnel au risque.
  3. Chaque commande passe par un garde-fou System One (risque d'outil) ; au-dela d'un niveau, confirmation
     humaine obligatoire. Tout reste dans un espace de travail (workspace) borne.
  4. Le clone verifie le resultat ; le tour est journalise (etat, decisions, outils, issue) : c'est le jeu de
     donnees de Jarvis, qui n'existait pas avant, et qui sert a l'entrainer (training/).
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from jev_clone.presets import GUARDRAILS, with_meta
from jev_clone.prompt import render_text
from jev_clone.tools import AgentLoop, SystemOneToolbox, _tool

INTENTS = {
    "chat": "a question or conversation that needs only a short answer, no files or commands",
    "create_app": "create a new application, script, website, tool or project from scratch",
    "modify_code": "change, fix, extend or refactor existing files in the workspace",
    "run_command": "run a program, command, test or build; install something",
    "browse": "look something up on the web or interact with a website",
    "remember": "store or recall a personal preference, fact or note",
    "other": "anything else",
}
LANGUAGES = {"python": None, "javascript_typescript": "Node, React, web apps", "html_css": "static pages", "bash": "shell scripts",
             "other": "another language", "none": "no code involved"}

JARVIS_TURN = with_meta({
    "intent": {"type": "choice", "instructions": "What does the user want Jarvis to do?", "criteria": INTENTS},
    "language": {"type": "choice", "instructions": "If code is involved, which language or stack fits best?", "criteria": LANGUAGES},
    "clarify": {"type": "noul", "instructions": "Is the request too ambiguous or incomplete to act on without asking one question first?"},
})

SYSTEM_PROMPT = """You are Jarvis, a local personal engineering assistant running on the user's machine.
You act through tools inside the workspace {workspace}. Rules:
- Plan briefly, then act: create files with write_file, run and test with run_command, read what you need.
- Prefer small, working, well-structured projects (README, requirements or package.json, a way to run it, a test).
- Never invent secrets or credentials; ask the user through the `done` summary if something is missing.
- Commands are checked by a safety judge; destructive or privileged commands need the user's confirmation.
- Use judge_* tools for quick checks instead of long reasoning when a yes/no or a choice is enough.
- When the task is complete, call done with a short summary of what exists now and how to run it.
{memory}"""


@dataclass
class Turn:
    request: str
    pre: dict                      # reponses System One
    path: str                      # "chat" | "clarify" | "agent"
    response: str
    tool_calls: list[dict] = field(default_factory=list)
    verification: float | None = None
    latency_ms: float = 0.0


class Workspace:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.memory_file = self.root / ".jarvis" / "memory.jsonl"

    def resolve(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        if self.root not in p.parents and p != self.root:
            raise PermissionError(f"chemin hors de l'espace de travail : {rel}")
        return p

    def write(self, rel: str, content: str) -> dict:
        p = self.resolve(rel); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(content, encoding="utf-8")
        return {"ok": True, "path": rel, "bytes": len(content.encode())}

    def read(self, rel: str, max_chars: int = 20000) -> dict:
        p = self.resolve(rel)
        if not p.exists():
            return {"ok": False, "error": "fichier introuvable"}
        t = p.read_text(encoding="utf-8", errors="replace")
        return {"ok": True, "path": rel, "content": t[:max_chars], "truncated": len(t) > max_chars}

    def listing(self, rel: str = ".", max_entries: int = 200) -> dict:
        p = self.resolve(rel)
        out = []
        for q in sorted(p.rglob("*")):
            if any(part.startswith(".") or part in ("node_modules", "__pycache__", ".venv") for part in q.relative_to(self.root).parts):
                continue
            out.append(str(q.relative_to(self.root)) + ("/" if q.is_dir() else ""))
            if len(out) >= max_entries:
                break
        return {"ok": True, "entries": out}

    def run(self, cmd: str, timeout: int = 120) -> dict:
        try:
            r = subprocess.run(cmd, shell=True, cwd=self.root, capture_output=True, text=True, timeout=timeout)
            return {"ok": r.returncode == 0, "returncode": r.returncode, "stdout": r.stdout[-6000:], "stderr": r.stderr[-3000:]}
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"timeout apres {timeout}s"}

    def remember(self, note: str) -> dict:
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.memory_file, "a") as f:
            f.write(json.dumps({"ts": time.time(), "note": note}, ensure_ascii=False) + "\n")
        return {"ok": True}

    def memory(self, last: int = 20) -> list[str]:
        if not self.memory_file.exists():
            return []
        rows = [json.loads(l)["note"] for l in open(self.memory_file) if l.strip()]
        return rows[-last:]


class Jarvis:
    def __init__(self, s1_engine, s2_backend, workspace: Workspace, confirm: Callable[[str, dict], bool] | None = None,
                 ledger: str | Path | None = None, max_turns: int = 12, risk_confirm_level: int = 2,
                 budgets: tuple[int, ...] = (0, 512, 2048, 6144), command_timeout: int = 120):
        self.s1, self.s2, self.ws = s1_engine, s2_backend, workspace
        self.confirm = confirm or (lambda cmd, judged: False)   # sans callback : on refuse ce qui demande confirmation
        self.ledger = Path(ledger) if ledger else workspace.root / ".jarvis" / "ledger.jsonl"
        self.max_turns, self.risk_confirm_level, self.budgets, self.command_timeout = max_turns, risk_confirm_level, budgets, command_timeout
        self.toolbox = SystemOneToolbox(s1_engine)

    # ---- garde-fou des commandes -------------------------------------------------------------------
    def judge_command(self, cmd: str, request: str) -> dict:
        r = self.s1.answer({"state": {"user_request": request, "proposed_tool_call": f"shell: {cmd}"},
                            "questions": {"tool_risk": GUARDRAILS["tool_risk"], "risk": GUARDRAILS["risk"],
                                          "policy_violation": GUARDRAILS["policy_violation"]}})
        a = r.answers
        return {"tool_risk": a["tool_risk"].choice, "tool_risk_conf": a["tool_risk"].confidence, "risk": a["risk"].score,
                "policy_violation": a["policy_violation"].noul}

    def _tools(self, request: str, calls: list[dict]):
        ws = self.ws

        def write_file(a):
            r = ws.write(a["path"], a["content"]); calls.append({"tool": "write_file", "path": a["path"]}); return r
        def read_file(a):
            return ws.read(a["path"])
        def list_files(a):
            return ws.listing(a.get("path", "."))
        def run_command(a):
            cmd = a["command"]
            judged = self.judge_command(cmd, request)
            risky = judged["tool_risk"] in ("destructive", "privileged", "exfiltration") or round(judged["risk"]) >= self.risk_confirm_level \
                or judged["policy_violation"] >= 0.5
            if risky and not self.confirm(cmd, judged):
                calls.append({"tool": "run_command", "command": cmd, "blocked": True, "judged": judged})
                return {"ok": False, "blocked": True, "reason": "command needs user confirmation and was not confirmed", "judged": judged}
            r = ws.run(cmd, timeout=self.command_timeout)
            calls.append({"tool": "run_command", "command": cmd, "ok": r.get("ok"), "judged": judged})
            return r
        def remember(a):
            calls.append({"tool": "remember"}); return ws.remember(a["note"])
        def done(a):
            calls.append({"tool": "done"}); return {"__stop__": True, "summary": a.get("summary", "")}

        return {
            "write_file": (_tool("write_file", "Create or overwrite a text file in the workspace.", {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]), write_file),
            "read_file": (_tool("read_file", "Read a text file from the workspace.", {"path": {"type": "string"}}, ["path"]), read_file),
            "list_files": (_tool("list_files", "List files under a workspace directory.", {"path": {"type": "string"}}, []), list_files),
            "run_command": (_tool("run_command", "Run a shell command in the workspace (tests, builds, installs, running the app). Risky commands need user confirmation.", {"command": {"type": "string"}}, ["command"]), run_command),
            "remember": (_tool("remember", "Store a durable note about the user's preferences or project.", {"note": {"type": "string"}}, ["note"]), remember),
            "done": (_tool("done", "Finish the task with a short summary (what exists now, how to run it, what is missing).", {"summary": {"type": "string"}}, ["summary"]), done),
        }

    # ---- un tour ------------------------------------------------------------------------------------------
    def handle(self, request: str, history: list[dict] | None = None) -> Turn:
        t0 = time.perf_counter()
        history = history or []
        pre_resp = self.s1.answer({"state": {"request": request, "workspace_files": self.ws.listing()["entries"][:60],
                                             "recent_turns": history[-4:]}, "questions": JARVIS_TURN})
        pre = {k: v.model_dump(exclude={"legend"}) for k, v in pre_resp.answers.items()}
        intent, conf = pre["intent"]["choice"], pre["intent"]["confidence"]
        risk_level = int(round(pre["risk"]["score"]))
        budget = self.budgets[min(risk_level, len(self.budgets) - 1)]
        if pre["needs_reasoning"]["noul"] >= 0.5:
            budget = max(budget, self.budgets[1])
        memory = self.ws.memory()
        mem_txt = ("\nWhat you remember about the user:\n- " + "\n- ".join(memory)) if memory else ""
        system = SYSTEM_PROMPT.format(workspace=self.ws.root, memory=mem_txt)
        calls: list[dict] = []

        if pre["clarify"]["noul"] >= 0.7:
            q = self.s2.chat([{"role": "system", "content": system},
                              {"role": "user", "content": f"The request below is ambiguous. Ask the single most useful clarifying question, nothing else.\n\nRequest: {request}"}],
                             max_tokens=120, thinking_budget=0, temperature=0.3)["choices"][0]["message"].get("content", "").strip()
            turn = Turn(request, pre, "clarify", q)
        elif intent in ("chat", "other") and conf >= 0.5 and pre["needs_reasoning"]["noul"] < 0.5:
            a = self.s2.chat([{"role": "system", "content": system}, *history[-6:], {"role": "user", "content": request}],
                             max_tokens=400, thinking_budget=0, temperature=0.5)["choices"][0]["message"].get("content", "").strip()
            turn = Turn(request, pre, "chat", a)
        else:
            loop = AgentLoop(self.s2, self.toolbox, self._tools(request, calls), max_turns=self.max_turns, thinking_budget=budget, max_tokens=4096)
            res = loop.run([{"role": "system", "content": system}, *history[-6:],
                            {"role": "user", "content": f"{request}\n\n(Judged intent: {intent}, language: {pre['language']['choice']}, risk level: {risk_level}.)"}])
            summary = next((c["result"].get("summary") for s in reversed(res.steps) for c in s.tool_calls
                            if c["name"] == "done" and isinstance(c["result"], dict)), None)
            turn = Turn(request, pre, "agent", summary or (res.content or "").strip() or "(no summary)", tool_calls=calls)
        v = self.s1.answer({"state": {"request": request, "response": turn.response, "files": self.ws.listing()["entries"][:60]},
                            "questions": {"ok": {"type": "noul", "instructions": "Does the response (and the workspace state) satisfy the user's request?"}}})
        turn.verification = round(v.answers["ok"].noul, 3)
        turn.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        self._log(turn)
        return turn

    def _log(self, turn: Turn) -> None:
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ledger, "a") as f:
            f.write(json.dumps({"ts": time.time(), "request": turn.request, "pre": turn.pre, "path": turn.path,
                                "response": turn.response, "tool_calls": turn.tool_calls, "verification": turn.verification,
                                "latency_ms": turn.latency_ms}, ensure_ascii=False, default=str) + "\n")


def repl(jarvis: Jarvis) -> None:
    """Boucle interactive : lignes de l'utilisateur -> tours de Jarvis. `/quit` pour sortir."""
    history: list[dict] = []
    print(f"Jarvis pret. Espace de travail : {jarvis.ws.root}. /quit pour quitter.")
    while True:
        try:
            req = input("\nvous > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not req:
            continue
        if req in ("/quit", "/exit"):
            break
        t = jarvis.handle(req, history)
        print(f"\njarvis [{t.path}, {t.latency_ms:.0f} ms, verif {t.verification}] > {t.response}")
        if t.tool_calls:
            print("   outils : " + ", ".join(c.get("tool", "?") + (" (bloque)" if c.get("blocked") else "") for c in t.tool_calls))
        history += [{"role": "user", "content": req}, {"role": "assistant", "content": t.response}]


def confirm_in_terminal(cmd: str, judged: dict) -> bool:
    print(f"\n[garde-fou] commande jugee {judged['tool_risk']} (risque {judged['risk']:.1f}) : {cmd}")
    try:
        return input("executer ? [o/N] ").strip().lower() in ("o", "y", "oui", "yes")
    except EOFError:
        return False

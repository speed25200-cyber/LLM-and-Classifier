"""Prophet : agent general local a deux vitesses. Il sait tout faire ; la fusion le rend rapide et sur.

Principe : **le clone de Jev n'enferme jamais l'agent, il l'accelere et le protege.** Il n'y a pas de liste
fermee d'intentions qui aiguille vers des cases : Bonsai (System Two) garde toujours la pleine autonomie, avec
un catalogue d'outils ouvert qu'il peut etendre lui-meme (`create_tool`). System One (le clone) juge en une
passe, a chaque tour, des *proprietes* de la demande :
  * `direct`   : une reponse courte suffit-elle, sans outil ? -> chemin rapide (Bonsai sans reflexion, 1-3 s)
  * `needs_reasoning`, `risk` : budget de reflexion de Bonsai (0 / 512 / 2048 / 6144 tokens)
  * `clarify`  : indice transmis a Bonsai (qui decide lui-meme de poser une question ou d'agir)
  * un noul par outil du catalogue : les outils pertinents sont presentes en premier, les outils de base et
    les jugements restent toujours disponibles ; au-dela de `max_tools`, seuls les plus pertinents sont exposes
  * `intent`, `language` : etiquettes d'observation (journal, entrainement), jamais un aiguillage
Pendant la boucle, Bonsai consulte le clone (`judge_*`) ; chaque commande, code, nouvel outil, appel de competence,
navigation web ou ecriture de fichier executable passe par le garde-fou du clone (risque d'outil, juge sur tout ce qui
va s'executer) et, si l'action est risquee ou le verdict incertain, par votre confirmation. `.prophet/` n'est jamais
ecrit par les outils de fichiers. Une voie directe ratee est reprise en voie agent. Tout tour est journalise.
"""

from __future__ import annotations

import ast
import difflib
import fnmatch
import json
import os
import platform
import re
import subprocess
import sys
import time
import types
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from jev_clone.presets import GUARDRAILS, RISKY_TOOL_CLASSES, with_meta
from jev_clone.tools import AgentLoop, SystemOneToolbox, _tool

INTENTS = {"chat": "conversation or question", "create_app": "create an application, script, site or project", "modify_code": "change existing files",
           "run_command": "run, test, build, install", "browse": "use the web", "remember": "store or recall a personal note", "other": "anything else"}
LANGUAGES = {"python": None, "javascript_typescript": None, "html_css": None, "bash": None, "other": None, "none": "no code involved"}

PROPHET_TURN = with_meta({
    "direct": {"type": "noul", "instructions": "Can a short direct answer fully satisfy this request, without using any tool, file, command or web access?"},
    "clarify": {"type": "noul", "instructions": "Is the request so ambiguous that asking one question first would save a lot of wasted work?"},
    "intent": {"type": "choice", "instructions": "Observation only: what kind of request is this?", "criteria": INTENTS},
    "language": {"type": "choice", "instructions": "Observation only: if code is involved, which stack fits best?", "criteria": LANGUAGES},
})

SYSTEM_PROMPT = """You are Prophet, a general-purpose local agent running on the user's machine. You can do anything the user asks:
write and run code, create complete applications, operate the shell, use the web, keep notes, and build new tools for
yourself when the existing ones are not enough (create_tool). You act through tools inside the workspace {workspace}.
Machine: {platform}; run_command uses {shell}.
Guidelines:
- Explore before changing: glob / grep / read_file, then edit_file for targeted changes (write_file for new files).
- Think only as much as the task needs; act, observe, adjust. Use judge_* tools for quick checks instead of long reasoning.
- Prefer working, well-structured results (a way to run it, a README, a test) and say clearly what exists and what is missing.
- Never invent secrets or credentials. Risky commands are checked by a safety judge and may require the user's confirmation.
- If the request is truly ambiguous, you may ask one question (finish with done and put the question in the summary); otherwise act.
- When finished, call done with a short summary.
{memory}"""

# voie directe : prompt propre (aucun outil ni `done` dans cette reponse) ; NEEDS_TOOLS = Bonsai demande la voie agent
DIRECT_PROMPT = """You are Prophet, a local assistant running on the user's machine ({platform}). This is the quick-answer path:
answer the user's message directly and concisely, in the user's language. No tool, file, command or web access is available
in this reply, so never claim to have created, read, changed or run anything.
If a correct answer really needs files, commands, the web or several steps of work, reply with exactly NEEDS_TOOLS and nothing else.
{memory}"""
NEEDS_TOOLS = "NEEDS_TOOLS"
# une reponse directe qui pretend avoir agi (sans outils, c'est faux) ou dit ne pas avoir acces : la voie agent s'impose
CLAIMS_ACTION = re.compile(r"\bi(?:'ve| have)? (?:just )?(?:created|written|saved|updated|modified|edited|deleted|installed|executed|ran)\b"
                           r"|\bj'ai (?:bien )?(?:cr[eé][eé]|[eé]crit|enregistr[eé]|modifi[eé]|mis [aà] jour|supprim[eé]|install[eé]|ex[eé]cut[eé]|lanc[eé])"
                           r"|\b(?:i (?:don't|do not|can't|cannot) (?:have )?access|je n'ai pas acc[eè]s|je ne peux pas acc[eé]der)"
                           r"|<tool_call>|\"name\":\s*\"(?:write_file|edit_file|run_command|python|done)\"", re.IGNORECASE)
CORE_TOOLS = ("done", "remember", "create_tool")   # toujours exposes, avec les judge_*
BUILTIN_TOOLS = ("write_file", "edit_file", "read_file", "list_files", "glob", "grep", "run_command", "python", "browse",
                 "remember", "create_tool", "done", "desktop")   # une competence ne les remplace jamais
# fichiers dont le contenu s'execute (ecrits : juges en mode smart ; lances par run_command : montres au juge)
EXEC_SUFFIXES = (".py", ".pyw", ".ps1", ".psm1", ".bat", ".cmd", ".sh", ".bash", ".zsh", ".js", ".mjs", ".cjs", ".ts", ".vbs",
                 ".rb", ".pl", ".php")
EXEC_NAMES = ("makefile", "package.json", "justfile")
META_REFUSAL = ("the .prophet/ folder is managed by Prophet itself (tools, memory, journal): write elsewhere; "
                "use create_tool to add a tool and remember to store a note")
MUTATING_TOOLS = ("write_file", "edit_file", "run_command", "python", "create_tool", "browse", "desktop")
PERMISSION_MODES = ("smart", "ask", "auto")   # smart : le clone decide quand demander ; ask : toujours ; auto : jamais
EFFORTS = ("auto", "fast", "deep")            # auto : budget de reflexion choisi par le clone
PLAN_MODE_NOTE = ("\nPLAN MODE is ON: do not modify files or run commands. Investigate with read-only tools "
                  "(read_file, list_files, glob, grep, judge_*), then call done with a numbered, concrete plan.")
IS_WINDOWS = os.name == "nt"
SHELL_NAME = "PowerShell (Windows)" if IS_WINDOWS else "bash"
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".prophet", "dist", "build", ".mypy_cache", ".pytest_cache"}



@dataclass
class Turn:
    request: str
    pre: dict
    path: str                    # "direct" | "agent"
    response: str
    tool_calls: list[dict] = field(default_factory=list)
    tools_exposed: list[str] = field(default_factory=list)
    verification: float | None = None
    latency_ms: float = 0.0
    stopped_by: str = ""
    stats: dict = field(default_factory=dict)


def _clip(x, n: int) -> str:
    """Texte borne pour les etats du classifieur (debut + fin, qui portent l'essentiel)."""
    t = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, default=str)
    return t if len(t) <= n else t[: n * 2 // 3] + " [...] " + t[-n // 3:]


def _executable(path: str) -> bool:
    n = Path(str(path).replace("\\", "/")).name.lower()
    return n.endswith(EXEC_SUFFIXES) or n in EXEC_NAMES


def _skill_tool(source: str) -> dict:
    """TOOL d'une competence lu SANS executer son code (ast) : litteral Python, ou json.loads('...') du gabarit."""
    tree = ast.parse(source)
    tool = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "TOOL" for t in node.targets):
            v = node.value
            if isinstance(v, ast.Call) and getattr(v.func, "attr", getattr(v.func, "id", "")) == "loads" and len(v.args) == 1:
                tool = json.loads(ast.literal_eval(v.args[0]))
            else:
                tool = ast.literal_eval(v)
    if not isinstance(tool, dict) or not any(isinstance(n, ast.FunctionDef) and n.name == "run" for n in tree.body):
        raise ValueError("a literal TOOL definition and def run(args) are required")
    if not str(tool["function"]["name"]).isidentifier():
        raise ValueError("the tool name must be a Python identifier")
    return tool


class Skill:
    """Competence creee par Prophet. Le code lu au chargement est exactement celui qui s'execute, et seulement a
    l'appel (que Prophet fait passer par le garde-fou) : rien ne tourne au chargement, rien ne change entre les deux."""

    def __init__(self, path: Path, source: str):
        self.path, self.source = path, source

    def __call__(self, args: dict):
        mod = types.ModuleType(f"prophet_skill_{self.path.stem}")
        mod.__file__ = str(self.path)
        exec(compile(self.source, str(self.path), "exec"), mod.__dict__)
        return mod.run(args)


class Workspace:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta = self.root / ".prophet"
        self.meta.mkdir(exist_ok=True)
        self.skills_dir = self.meta / "skills"
        self.skills_dir.mkdir(exist_ok=True)
        self.memory_file = self.meta / "memory.jsonl"

    def resolve(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        if self.root not in p.parents and p != self.root:
            raise PermissionError(f"chemin hors de l'espace de travail : {rel}")
        return p

    def in_meta(self, rel: str) -> bool:
        """Chemin dans .prophet/ (competences executables, memoire, journal) ? Casse ignoree (Windows, macOS)."""
        p, m = str(self.resolve(rel)).casefold(), str(self.meta).casefold()
        return p == m or p.startswith(m + os.sep)

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
        p = self.resolve(rel); out = []
        for q in sorted(p.rglob("*")):
            if any(part.startswith(".") or part in SKIP_DIRS for part in q.relative_to(self.root).parts):
                continue
            out.append(q.relative_to(self.root).as_posix() + ("/" if q.is_dir() else ""))   # "/" partout (Windows compris)
            if len(out) >= max_entries:
                break
        return {"ok": True, "entries": out}

    def run(self, cmd: str | list[str], timeout: int = 120) -> dict:
        """Commande shell (PowerShell sous Windows, sh ailleurs) ou argv explicite."""
        if isinstance(cmd, str) and IS_WINDOWS:
            # sortie en UTF-8 (sinon code page 1252/850 : accents illisibles)
            cmd = ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                   "[Console]::OutputEncoding=[Text.Encoding]::UTF8; $OutputEncoding=[Text.Encoding]::UTF8; " + cmd]
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
        try:
            r = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=self.root, capture_output=True, text=True,
                               timeout=timeout, encoding="utf-8", errors="replace", env=env)
            return {"ok": r.returncode == 0, "returncode": r.returncode, "stdout": r.stdout[-6000:], "stderr": r.stderr[-3000:]}
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"timeout apres {timeout}s"}
        except FileNotFoundError as e:
            return {"ok": False, "error": str(e)}

    def run_python(self, code: str, timeout: int = 120) -> dict:
        """Execute un extrait Python via un fichier temporaire (portable : pas de heredoc bash)."""
        tmp = self.meta / "tmp"; tmp.mkdir(exist_ok=True)
        f = tmp / f"snippet_{uuid.uuid4().hex[:8]}.py"
        f.write_text(code, encoding="utf-8")
        try:
            return self.run([sys.executable, str(f)], timeout=timeout)
        finally:
            f.unlink(missing_ok=True)

    @staticmethod
    def _diff(rel: str, before: str, after: str, max_lines: int = 400) -> str:
        lines = list(difflib.unified_diff(before.splitlines(), after.splitlines(), f"a/{rel}", f"b/{rel}", lineterm="", n=3))
        return "\n".join(lines[:max_lines]) + ("\n..." if len(lines) > max_lines else "")

    def write_with_diff(self, rel: str, content: str) -> dict:
        p = self.resolve(rel)
        before = p.read_text(encoding="utf-8", errors="replace") if p.exists() else None
        r = self.write(rel, content)
        r["__ui__"] = {"diff": self._diff(rel, before or "", content), "created": before is None,
                       "lines": content.count("\n") + (0 if content.endswith("\n") else 1)}
        return r

    def edit(self, rel: str, old: str, new: str, replace_all: bool = False) -> dict:
        """Remplacement exact (comme l'outil Edit de Claude Code) : economise le contexte d'un 27B a 8-16 k."""
        p = self.resolve(rel)
        if not p.exists():
            return {"ok": False, "error": "fichier introuvable ; utiliser write_file pour creer"}
        text = p.read_text(encoding="utf-8", errors="replace")
        n = text.count(old) if old else 0
        if n == 0:
            return {"ok": False, "error": "old_string introuvable : relire le fichier (read_file) et copier le texte exact"}
        if n > 1 and not replace_all:
            return {"ok": False, "error": f"old_string apparait {n} fois : donner plus de contexte ou replace_all=true"}
        after = text.replace(old, new) if replace_all else text.replace(old, new, 1)
        p.write_text(after, encoding="utf-8")
        return {"ok": True, "path": rel, "replacements": n if replace_all else 1, "__ui__": {"diff": self._diff(rel, text, after)}}

    def _walk(self, rel: str = "."):
        base = self.resolve(rel)
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
            for fn in sorted(filenames):
                yield Path(dirpath) / fn

    def glob(self, pattern: str, max_entries: int = 300) -> dict:
        out = []
        for f in self._walk():
            r = f.relative_to(self.root).as_posix()
            if fnmatch.fnmatch(r, pattern) or fnmatch.fnmatch(f.name, pattern):
                out.append(r)
                if len(out) >= max_entries:
                    break
        return {"ok": True, "files": out, "truncated": len(out) >= max_entries}

    def grep(self, pattern: str, rel: str = ".", file_glob: str | None = None, max_matches: int = 200,
             ignore_case: bool = False) -> dict:
        try:
            rx = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        except re.error as e:
            return {"ok": False, "error": f"regex invalide : {e}"}
        out = []
        for f in self._walk(rel):
            if file_glob and not fnmatch.fnmatch(f.name, file_glob):
                continue
            try:
                if f.stat().st_size > 2_000_000:
                    continue
                with open(f, encoding="utf-8") as fh:
                    for i, line in enumerate(fh, 1):
                        if rx.search(line):
                            out.append(f"{f.relative_to(self.root).as_posix()}:{i}: {line.rstrip()[:240]}")
                            if len(out) >= max_matches:
                                return {"ok": True, "matches": out, "truncated": True}
            except (UnicodeDecodeError, OSError):
                continue
        return {"ok": True, "matches": out, "truncated": False}

    def remember(self, note: str) -> dict:
        with open(self.memory_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "note": note}, ensure_ascii=False) + "\n")
        return {"ok": True}

    def memory(self, last: int = 20) -> list[str]:
        if not self.memory_file.exists():
            return []
        with open(self.memory_file, encoding="utf-8") as f:
            return [json.loads(l)["note"] for l in f if l.strip()][-last:]

    # ---- outils crees par Prophet lui-meme ----------------------------------------------------------------
    def load_skills(self) -> dict[str, tuple[dict, Callable]]:
        """Chaque `.prophet/skills/<nom>.py` definit TOOL (definition OpenAI litterale) et run(args) -> dict.
        Lecture statique : aucun code de competence ne s'execute au chargement (voir Skill)."""
        out = {}
        for f in sorted(self.skills_dir.glob("*.py")):
            try:
                src = f.read_text(encoding="utf-8")
                tool = _skill_tool(src)
                out[tool["function"]["name"]] = (tool, Skill(f, src))
            except Exception as e:  # une competence cassee ne doit pas bloquer l'agent
                out[f"broken_{f.stem}"] = (_tool(f"broken_{f.stem}", f"Skill file {f.name} failed to load: {str(e)[:120]}", {}, []),
                                            lambda a, _e=str(e): {"ok": False, "error": _e})
        return out


SKILL_TEMPLATE = '''"""Outil cree par Prophet : {name}"""
import json, subprocess, sys
TOOL = json.loads({tool_json!r})

def run(args):
{body}
'''


class _MeteredS1:
    """Le clone vu par les outils judge_* du tour : chaque appel compte dans la latence S1 du tour."""

    def __init__(self, prophet: "Prophet"):
        self.prophet = prophet

    def answer(self, req):
        return self.prophet._s1(req)


class _TurnToolbox(SystemOneToolbox):
    """Outils judge_* d'un tour : comptes (latence S1) et inscrits au journal du tour comme les autres outils."""

    def __init__(self, prophet: "Prophet", calls: list[dict]):
        super().__init__(_MeteredS1(prophet))
        self.calls = calls

    def call(self, name: str, args: dict) -> dict:
        t0 = time.perf_counter()
        rec = {"tool": name, "question": str(args.get("question", ""))[:200]}
        try:
            r = super().call(name, args)
        except Exception as e:
            self.calls.append({**rec, "ok": False, "error": str(e)[:200]})
            raise
        self.calls.append({**rec, "ok": True, "ms": round((time.perf_counter() - t0) * 1000, 1)})
        return r


class Prophet:
    """on_event(evt) : flux d'evenements pour une interface (decision System One, texte diffuse, outils, verification).
    should_stop() : annulation cooperative. permission_mode : smart | ask | auto (voir PERMISSION_MODES).
    plan_mode : lecture seule, Bonsai rend un plan. max_context_chars : compaction de la boucle d'agent.
    danger_threshold : masse de probabilite destructive + privileged + exfiltration a partir de laquelle le mode smart
    demande (la masse benigne doit l'emporter nettement, pas seulement l'argmax). reroute_below : verification S1 sous
    laquelle une reponse directe est reprise en voie agent. thinking_share : part maximale de max_tokens pour la reflexion."""

    def __init__(self, s1_engine, s2_backend, workspace: Workspace, confirm: Callable[[str, dict], bool] | None = None,
                 ledger: str | Path | None = None, max_turns: int = 16, risk_confirm_level: int = 2,
                 budgets: tuple[int, ...] = (0, 512, 2048, 6144), command_timeout: int = 180, max_tools: int = 12,
                 browser_factory: Callable | None = None, on_event: Callable[[dict], None] | None = None,
                 should_stop: Callable[[], bool] | None = None, permission_mode: str = "smart", plan_mode: bool = False,
                 max_context_chars: int | None = None, max_tokens: int = 4096, desktop_factory: Callable | None = None,
                 danger_threshold: float = 0.35, reroute_below: float = 0.35, thinking_share: float = 0.6):
        self.s1, self.s2, self.ws = s1_engine, s2_backend, workspace
        self.confirm = confirm or (lambda cmd, judged: False)
        self.ledger = Path(ledger) if ledger else workspace.meta / "ledger.jsonl"
        self.max_turns, self.risk_confirm_level, self.budgets = max_turns, risk_confirm_level, budgets
        self.command_timeout, self.max_tools, self.browser_factory = command_timeout, max_tools, browser_factory
        self.toolbox = SystemOneToolbox(s1_engine)
        self.on_event = on_event
        self.should_stop = should_stop or (lambda: False)
        if permission_mode not in PERMISSION_MODES:
            raise ValueError(f"permission_mode doit etre dans {PERMISSION_MODES}")
        self.permission_mode, self.plan_mode = permission_mode, plan_mode
        self.max_context_chars, self.max_tokens = max_context_chars, max_tokens
        self.desktop_factory = desktop_factory
        self.danger_threshold, self.reroute_below, self.thinking_share = danger_threshold, reroute_below, thinking_share
        self._s1_ms, self._s1_calls = 0.0, 0

    def _emit(self, evt: dict) -> None:
        if self.on_event is not None:
            try:
                self.on_event(evt)
            except Exception:
                pass

    def _s1(self, req):
        """Tout appel au clone passe ici : la latence S1 d'un tour est la somme de tous ses appels (echecs compris)."""
        t0 = time.perf_counter()
        try:
            return self.s1.answer(req)
        finally:
            self._s1_ms += (time.perf_counter() - t0) * 1000
            self._s1_calls += 1

    # ---- garde-fou -----------------------------------------------------------------------------------------
    def judge(self, request: str, proposed: str, chunk: int = 2000, max_chunks: int = 6) -> dict:
        """Risque d'une action selon le clone. Une action longue est jugee en entier, par morceaux (le pire l'emporte) ;
        au-dela de max_chunks, la partie non vue impose un arret humain."""
        parts = [proposed[i:i + chunk] for i in range(0, len(proposed), chunk)] or [""]
        partial = len(parts) > max_chunks
        if partial:
            parts = parts[: max_chunks - 1] + parts[-1:]
        head = proposed.split("\n", 1)[0][:160]   # l'action (outil, fichier) accompagne chaque morceau
        verdicts = []
        for k, part in enumerate(parts):
            label = part if len(parts) == 1 else f"{head}\n[part {k + 1} of {len(parts)}{' (middle omitted before)' if partial and k == len(parts) - 1 else ''}]\n{part}"
            verdicts.append(self._judge_one(request, label))
            if "s1_error" in verdicts[-1]:
                break
        j = dict(max(verdicts, key=lambda v: (v["hard_stop"], v["needs_confirmation"], v["p_risky"], v["risk"])))
        j["latency_ms"] = round(sum(v["latency_ms"] for v in verdicts), 2)
        if len(parts) > 1:
            j["chunks"] = len(parts)
        if partial:   # une partie de ce qui va s'executer n'a pas ete vue par le juge : on demande, toujours
            j.update(partial=True, needs_confirmation=True, hard_stop=True)
        return j

    def _judge_one(self, request: str, proposed: str) -> dict:
        try:
            r = self._s1({"state": {"user_request": _clip(request, 1200), "proposed_action": proposed},
                          "questions": {"tool_risk": GUARDRAILS["tool_risk"], "risk": GUARDRAILS["risk"], "policy_violation": GUARDRAILS["policy_violation"]}})
        except Exception as e:   # classifieur indisponible : prudence, on demande (et aucune autorisation memorisee ne vaut)
            return {"s1_consulted": True, "tool_risk": "unknown", "tool_risk_conf": 0.0, "p_risky": 1.0, "risk": 2.0, "policy_violation": 0.0,
                    "needs_confirmation": True, "hard_stop": True, "latency_ms": 0.0, "s1_error": str(e)[:200]}
        a = r.answers
        p_risky = sum(a["tool_risk"].probabilities.get(c, 0.0) for c in RISKY_TOOL_CLASSES)
        j = {"s1_consulted": True, "tool_risk": a["tool_risk"].choice, "tool_risk_conf": a["tool_risk"].confidence,
             "p_risky": round(p_risky, 4), "risk": a["risk"].score, "policy_violation": a["policy_violation"].noul}
        # arret humain obligatoire, meme pour un outil « toujours autorise » : danger probable, politique, risque maximal
        j["hard_stop"] = j["tool_risk"] in RISKY_TOOL_CLASSES or p_risky >= 0.5 or j["policy_violation"] >= 0.5 or j["risk"] >= 2.5
        # incertitude : readonly + workspace_write doivent l'emporter avec une marge, l'argmax ne suffit pas
        j["needs_confirmation"] = j["hard_stop"] or p_risky >= self.danger_threshold or j["risk"] >= self.risk_confirm_level - 0.5
        j["latency_ms"] = r.latency_ms
        return j

    def _guarded(self, request: str, calls: list[dict], kind: str, describe: str, action: Callable[[], dict], record: dict,
                 judge: bool = True, preview: dict | None = None) -> dict:
        """Politique d'autorisation. plan : toute action qui modifie est refusee. auto : jamais de question (le clone
        n'est pas consulte). smart : le clone juge l'action et on ne demande que si elle est risquee ou incertaine.
        ask : on demande toujours. Le verdict passe a confirm() porte hard_stop (arret humain obligatoire)."""
        if self.plan_mode:
            calls.append({"tool": kind, **record, "blocked": True, "plan_mode": True})
            return {"ok": False, "blocked": True, "reason": "plan mode: no changes allowed; finish with done and a plan"}
        judged = self.judge(request, describe) if judge and self.permission_mode != "auto" else None
        log = {"judged": judged} if judged is not None else {"judged": None, "s1_consulted": False}
        ask = self.permission_mode == "ask" or (self.permission_mode == "smart" and judged is not None and judged["needs_confirmation"])
        if ask and not self.confirm(describe, {**(judged or {"s1_consulted": False}), "tool": kind, "preview": preview or {}}):
            calls.append({"tool": kind, **record, "blocked": True, **log})
            return {"ok": False, "blocked": True, "reason": "the user declined this action", "judged": judged}
        try:
            r = action()
        except Exception as e:   # l'echec de l'outil est rendu au modele et journalise
            r = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:500]}"}
        calls.append({"tool": kind, **record, "ok": r.get("ok") if isinstance(r, dict) else True, **log, **({"confirmed": True} if ask else {})})
        return r

    def _scripts_of(self, cmd: str, max_files: int = 2, max_chars: int = 2400) -> str:
        """Contenu (debut + fin, borne) des scripts du workspace qu'une commande lance : le juge voit ce qui va tourner,
        pas seulement `python x.py`."""
        out, seen = [], set()
        for a, b, c in re.findall(r'"([^"]+)"|\'([^\']+)\'|([^\s;&|<>()`\'"]+)', cmd):
            tok = (a or b or c).strip()
            if not _executable(tok) or tok in seen:
                continue
            seen.add(tok)
            try:
                p = self.ws.resolve(tok)
            except (PermissionError, OSError, ValueError):   # hors du workspace : la commande elle-meme est jugee
                continue
            if p.is_file():
                out.append(f"\n--- content of {tok} ---\n" + _clip(p.read_text(encoding="utf-8", errors="replace"), max_chars))
                if len(out) >= max_files:
                    break
        return "".join(out)

    def _skill(self, request: str, calls: list[dict], name: str, skill: Skill) -> Callable:
        """Appel d'une competence creee par Prophet : garde-fou comme run_command, le juge voit les arguments et le code."""
        def call(a):
            args = json.dumps(a, ensure_ascii=False, default=str)
            return self._guarded(request, calls, name, f"run the self-made tool `{name}` with arguments {_clip(args, 600)}; its code:\n{skill.source}",
                                 lambda: skill(a), {"skill": True, "args": _clip(args, 200)}, preview={"code": skill.source[:6000]})
        return call

    # ---- catalogue d'outils ------------------------------------------------------------------------------------
    def _catalog(self, request: str, calls: list[dict]) -> dict[str, tuple[dict, Callable]]:
        ws = self.ws

        def guard_write(exe: bool) -> bool:   # smart : seuls les fichiers executables sont juges (python x.py suivra)
            return self.permission_mode == "ask" or self.plan_mode or (exe and self.permission_mode == "smart")
        def meta_refused(tool: str, path: str) -> dict:   # .prophet/skills/*.py serait execute aux tours suivants
            calls.append({"tool": tool, "path": path, "ok": False, "blocked": True, "meta": True})
            return {"ok": False, "blocked": True, "error": META_REFUSAL}
        def write_file(a):
            path, content = a["path"], a["content"]
            if ws.in_meta(path):
                return meta_refused("write_file", path)
            exe = _executable(path)
            if guard_write(exe):
                before = ws.resolve(path).read_text(encoding="utf-8", errors="replace") if ws.resolve(path).exists() else ""
                return self._guarded(request, calls, "write_file", f"write {path} ({len(content)} chars)" + (f":\n{content}" if exe else ""),
                                     lambda: ws.write_with_diff(path, content), {"path": path}, judge=exe,
                                     preview={"diff": ws._diff(path, before, content)})
            r = ws.write_with_diff(path, content); calls.append({"tool": "write_file", "path": path, "ok": r.get("ok")}); return r
        def edit_file(a):
            path, old, new, rep = a["path"], a.get("old_string", ""), a.get("new_string", ""), bool(a.get("replace_all", False))
            if ws.in_meta(path):
                return meta_refused("edit_file", path)
            exe = _executable(path)
            if guard_write(exe):
                return self._guarded(request, calls, "edit_file", f"edit {path}" + (f": replace\n{old}\n--- with ---\n{new}" if exe else ""),
                                     lambda: ws.edit(path, old, new, rep), {"path": path}, judge=exe, preview={"old": old[:4000], "new": new[:4000]})
            r = ws.edit(path, old, new, rep); calls.append({"tool": "edit_file", "path": path, "ok": r.get("ok")}); return r
        def run_command(a):
            cmd = a["command"]
            return self._guarded(request, calls, "run_command", f"shell: {cmd}" + self._scripts_of(cmd), lambda: ws.run(cmd, timeout=self.command_timeout),
                                 {"command": cmd}, preview={"command": cmd})
        def python(a):
            code = a["code"]
            return self._guarded(request, calls, "python", f"python code:\n{code}",
                                 lambda: ws.run_python(code, timeout=self.command_timeout), {"code": code[:200]}, preview={"code": code[:6000]})
        def create_tool(a):
            name, desc, params, body = a["name"], a["description"], a.get("parameters") or {"type": "object", "properties": {}}, a["python_body"]
            if not name.isidentifier() or name in BUILTIN_TOOLS or name.startswith(("judge_", "broken_")):
                return {"ok": False, "error": "name must be a Python identifier that is not a built-in tool name"}
            if isinstance(params, str):
                params = json.loads(params)
            src = SKILL_TEMPLATE.format(name=name, tool_json=json.dumps(_tool(name, desc, params.get("properties", {}), params.get("required", []))),
                                        body="\n".join("    " + line for line in body.splitlines()) or "    return {}")
            def do():
                try:   # verification sans rien executer : definition litterale et code compilable
                    _skill_tool(src); compile(src, f"{name}.py", "exec")
                except Exception as e:
                    return {"ok": False, "error": f"the tool does not load: {str(e)[:300]}"}
                (ws.skills_dir / f"{name}.py").write_text(src, encoding="utf-8")
                return {"ok": True, "tool": name, "note": "available from the next turn (and now via run_command if needed)"}
            return self._guarded(request, calls, "create_tool", f"new tool `{name}`: {desc}\n{body}", do, {"name": name},
                                 preview={"code": body[:6000]})
        def browse(a):
            goal, url, slots = a["goal"], a.get("url"), a.get("slots") or {}
            return self._guarded(request, calls, "browse", f"web browser: {goal}" + (f" (start at {url})" if url else "")
                                 + (f"; values it may type into web forms: {json.dumps(slots, ensure_ascii=False)}" if slots else ""),
                                 lambda: self.browser_factory()(goal, url, slots), {"goal": goal},
                                 preview={"command": f"navigateur : {goal}" + (f" · {url}" if url else "")})
        def desktop(a):
            goal, app = a["goal"], a.get("app")
            return self._guarded(request, calls, "desktop", f"control the computer desktop: {goal}" + (f" (open {app})" if app else ""),
                                 lambda: self.desktop_factory()(goal, app, a.get("slots") or {}), {"goal": goal},
                                 preview={"command": f"bureau : {goal}" + (f" · ouvre {app}" if app else "")})
        def remember(a):
            calls.append({"tool": "remember"}); return ws.remember(a["note"])
        def done(a):
            calls.append({"tool": "done"}); return {"__stop__": True, "summary": a.get("summary", "")}
        def read_file(a):
            calls.append({"tool": "read_file", "path": a["path"]}); return ws.read(a["path"])
        def list_files(a):
            calls.append({"tool": "list_files", "path": a.get("path", ".")}); return ws.listing(a.get("path", "."))
        def glob_files(a):
            calls.append({"tool": "glob", "pattern": a["pattern"]}); return ws.glob(a["pattern"])
        def grep(a):
            calls.append({"tool": "grep", "pattern": a["pattern"]})
            return ws.grep(a["pattern"], a.get("path", "."), a.get("glob"), ignore_case=bool(a.get("ignore_case", False)))

        cat = {
            "write_file": (_tool("write_file", "Create or overwrite a text file in the workspace (prefer edit_file for changes to an existing file).", {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]), write_file),
            "edit_file": (_tool("edit_file", "Replace an exact string in an existing file (copy old_string exactly from read_file, with enough context to be unique).",
                                {"path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}, "replace_all": {"type": "boolean"}},
                                ["path", "old_string", "new_string"]), edit_file),
            "read_file": (_tool("read_file", "Read a text file from the workspace.", {"path": {"type": "string"}}, ["path"]), read_file),
            "list_files": (_tool("list_files", "List files under a workspace directory.", {"path": {"type": "string"}}, []), list_files),
            "glob": (_tool("glob", "Find files by name pattern, e.g. '**/*.py' or '*.md'.", {"pattern": {"type": "string"}}, ["pattern"]), glob_files),
            "grep": (_tool("grep", "Search file contents with a regular expression; returns path:line: text.",
                           {"pattern": {"type": "string"}, "path": {"type": "string"}, "glob": {"type": "string"}, "ignore_case": {"type": "boolean"}}, ["pattern"]), grep),
            "run_command": (_tool("run_command", f"Run a {SHELL_NAME} command in the workspace (tests, builds, installs, running apps).", {"command": {"type": "string"}}, ["command"]), run_command),
            "python": (_tool("python", "Run a Python snippet in the workspace and return its output.", {"code": {"type": "string"}}, ["code"]), python),
            "remember": (_tool("remember", "Store a durable note about the user or the project.", {"note": {"type": "string"}}, ["note"]), remember),
            "create_tool": (_tool("create_tool", "Build a new reusable tool for yourself: a Python function body (args: dict) -> dict, with a name, description and JSON-schema parameters. It becomes available on later turns.",
                                  {"name": {"type": "string"}, "description": {"type": "string"}, "parameters": {"type": "object"}, "python_body": {"type": "string"}},
                                  ["name", "description", "python_body"]), create_tool),
            "done": (_tool("done", "Finish with a short summary (what exists now, how to run it, what is missing, or your one clarifying question).", {"summary": {"type": "string"}}, ["summary"]), done),
        }
        if self.browser_factory is not None:   # pas de navigateur configure : l'outil n'est pas propose
            cat["browse"] = (_tool("browse", "Achieve a goal in a web browser (search, read, fill forms). Give the goal, optionally a start URL and slot values to type.",
                                   {"goal": {"type": "string"}, "url": {"type": "string"}, "slots": {"type": "object"}}, ["goal"]), browse)
        if self.desktop_factory is not None:
            cat["desktop"] = (_tool("desktop", "Operate desktop applications like a person (click buttons, fill fields, shortcuts, open apps) to reach a goal. "
                                    "Optionally give the app to open first and slot values to type.",
                                    {"goal": {"type": "string"}, "app": {"type": "string"}, "slots": {"type": "object"}}, ["goal"]), desktop)
        if not self.plan_mode:   # mode plan : les competences ne sont ni chargees ni executees
            for name, (tool, fn) in ws.load_skills().items():
                if name in cat or name in BUILTIN_TOOLS or name.startswith("judge_"):   # jamais a la place d'un outil de base
                    continue
                cat[name] = (tool, self._skill(request, calls, name, fn) if isinstance(fn, Skill) else fn)
        return cat

    def _select_tools(self, request: str, catalog: dict, pre: dict, recent: list[dict] | None = None) -> list[str]:
        """Pertinence par outil (un noul chacun, une passe, avec les derniers echanges) : ordre de presentation, et filtre
        au-dela de max_tools. Les outils de base (done, remember, create_tool) restent toujours exposes."""
        names = [n for n in catalog if n not in CORE_TOOLS]
        if not names:
            return list(CORE_TOOLS)
        qs = {f"t_{n}": {"type": "noul", "instructions": f"Would the tool `{n}` ({catalog[n][0]['function']['description'][:120]}) plausibly be useful for this request?"} for n in names}
        try:
            r = self._s1({"state": {"request": _clip(request, 2000), "recent_turns": recent or []}, "questions": qs})
        except Exception:
            return names[: max(1, self.max_tools - len(CORE_TOOLS))] + [c for c in CORE_TOOLS if c in catalog]
        ranked = sorted(names, key=lambda n: -r.answers[f"t_{n}"].noul)
        keep = ranked[: max(1, self.max_tools - len(CORE_TOOLS))]
        pre["tool_relevance"] = {n: round(r.answers[f"t_{n}"].noul, 3) for n in ranked}
        return keep + [c for c in CORE_TOOLS if c in catalog]

    def _budget(self, pre: dict, effort: str) -> tuple[int, int]:
        risk_level = int(round(pre["risk"]["score"]))
        budget = self.budgets[min(risk_level, len(self.budgets) - 1)]
        if pre["needs_reasoning"]["noul"] >= 0.5:
            budget = max(budget, self.budgets[1])
        if effort == "fast":
            budget = 0
        elif effort == "deep":
            budget = self.budgets[-1]
        return self._cap(budget), risk_level

    def _cap(self, budget: int) -> int:
        """La reflexion ne consomme jamais toute la generation : il reste de quoi repondre ou appeler un outil."""
        return min(budget, int(self.max_tokens * self.thinking_share))

    def _verify(self, request: str, response: str) -> float | None:
        try:
            v = self._s1({"state": {"request": _clip(request, 1500), "response": _clip(response, 2500), "files": self.ws.listing()["entries"][:60]},
                          "questions": {"ok": {"type": "noul", "instructions": "Does the response (and the workspace state) satisfy the user's request?"}}})
            return round(v.answers["ok"].noul, 3)
        except Exception:
            return None

    # ---- voie directe ---------------------------------------------------------------------------------------------
    def _direct(self, request: str, history: list[dict], pre: dict, memory: str, track: Callable[[dict], None]) -> Turn:
        system = DIRECT_PROMPT.format(platform=f"{platform.system()} {platform.release()}", memory=memory)
        kw: dict = {}
        if self.on_event is not None:
            def on_delta(d: dict) -> None:
                if d["type"] in ("content", "reasoning"):
                    self._emit({"type": "text.delta" if d["type"] == "content" else "thinking.delta", "text": d["text"]})
            kw = {"on_delta": on_delta, "should_stop": self.should_stop}
            self._emit({"type": "llm.start", "turn": 0, "thinking_budget": 0})
        resp = self.s2.chat([{"role": "system", "content": system}, *history[-6:], {"role": "user", "content": request}],
                            max_tokens=600, thinking_budget=0, temperature=0.5, **kw)
        finish = resp["choices"][0].get("finish_reason")
        track({"type": "llm.end", "turn": 0, "timings": resp.get("timings") or {}, "usage": resp.get("usage") or {}, "finish_reason": finish})
        a = (resp["choices"][0]["message"].get("content") or "").strip()
        return Turn(request, pre, "direct", a, stopped_by="cancelled" if resp.get("cancelled") else "length" if finish == "length" else "final")

    def _reroute_reason(self, request: str, turn: Turn) -> str | None:
        """Une voie directe ratee est reprise en voie agent : reponse vide, NEEDS_TOOLS, tronquee, qui pretend avoir agi,
        ou verification S1 trop basse. Sinon la verification calculee ici reste celle du tour."""
        if turn.stopped_by == "cancelled" or self.should_stop():
            return None
        text = turn.response
        if not text:
            return "empty"
        if text.lstrip().upper().startswith(NEEDS_TOOLS):
            return "needs_tools"
        if turn.stopped_by == "length":
            return "truncated"
        if CLAIMS_ACTION.search(text):
            return "claims_action"
        turn.verification = self._verify(request, text)
        if turn.verification is not None and turn.verification < self.reroute_below:
            return "low_verification"
        return None

    # ---- un tour --------------------------------------------------------------------------------------------------
    def handle(self, request: str, history: list[dict] | None = None, effort: str = "auto") -> Turn:
        t0 = time.perf_counter()
        history = history or []
        self._s1_ms, self._s1_calls = 0.0, 0
        self._emit({"type": "turn.start", "request": request, "plan_mode": self.plan_mode, "permission_mode": self.permission_mode, "effort": effort})
        files = self.ws.listing()["entries"]
        # le classifieur lit ~2 k tokens par slot (4 slots sur 8 k) : on borne ce qu'on lui montre
        recent = [{"role": m.get("role"), "content": _clip(m.get("content"), 400)} for m in history[-4:]]
        s1_ms = 0.0
        try:
            pre_resp = self._s1({"state": {"request": _clip(request, 2500), "workspace_files": files[:60], "recent_turns": recent},
                                 "questions": PROPHET_TURN})
            pre = {k: v.model_dump(exclude={"legend"}) for k, v in pre_resp.answers.items()}
            s1_ms = pre_resp.latency_ms
        except Exception as e:   # le clone accelere et protege, il n'est jamais un point de panne : Bonsai continue seul
            pre = {"direct": {"noul": 0.0}, "clarify": {"noul": 0.0}, "needs_reasoning": {"noul": 1.0}, "risk": {"score": 1.0},
                   "s1_error": str(e)[:200]}
        budget, risk_level = self._budget(pre, effort)
        direct = (pre["direct"]["noul"] >= 0.8 and pre["needs_reasoning"]["noul"] < 0.5 and risk_level == 0 and effort != "deep"
                  and not self.plan_mode)
        self._emit({"type": "s1.decision", "pre": pre, "latency_ms": s1_ms, "budget": budget, "risk_level": risk_level,
                    "path": "direct" if direct else "agent"})
        memory = self.ws.memory()
        memory_note = ("\nWhat you remember about the user:\n- " + "\n- ".join(memory)) if memory else ""
        calls: list[dict] = []
        stats: dict = {"s1_ms": 0.0, "llm_calls": 0, "tokens": 0, "tok_s": None, "prompt_ms": 0.0}

        def track(evt: dict) -> None:
            if evt.get("type") == "llm.end":
                tm = evt.get("timings") or {}
                stats["llm_calls"] += 1
                stats["tokens"] += int(tm.get("predicted_n") or 0)
                stats["prompt_ms"] += float(tm.get("prompt_ms") or 0)
                if tm.get("predicted_per_second"):
                    stats["tok_s"] = round(float(tm["predicted_per_second"]), 1)
                us = evt.get("usage") or {}
                used = int(us.get("prompt_tokens") or 0) + int(us.get("completion_tokens") or 0) or int(tm.get("prompt_n") or 0) + int(tm.get("cache_n") or 0) + int(tm.get("predicted_n") or 0)
                stats["ctx_tokens"] = max(stats.get("ctx_tokens", 0), used)   # contexte occupe au plus haut du tour
            self._emit(evt)

        turn = self._direct(request, history, pre, memory_note, track) if direct else None
        if turn is not None:
            reason = self._reroute_reason(request, turn)
            if reason:   # le clone s'est trompe de voie : Bonsai reprend le tour avec les outils (l'interface le montre)
                pre["rerouted"] = {"reason": reason, "verification": turn.verification}
                self._emit({"type": "s1.reroute", "reason": reason, "verification": turn.verification, "from": "direct", "to": "agent"})
                if effort != "fast":
                    budget = self._cap(max(budget, self.budgets[1]))
                turn = None
        if turn is None:
            system = SYSTEM_PROMPT.format(workspace=self.ws.root, platform=f"{platform.system()} {platform.release()}", shell=SHELL_NAME,
                                          memory=memory_note)
            if self.plan_mode:
                system += PLAN_MODE_NOTE
            catalog = self._catalog(request, calls)
            exposed = self._select_tools(request, catalog, pre, recent)
            self._emit({"type": "s1.tools", "relevance": pre.get("tool_relevance", {}), "exposed": exposed})
            tools = {n: catalog[n] for n in exposed}
            loop = AgentLoop(self.s2, _TurnToolbox(self, calls), tools, max_turns=self.max_turns, thinking_budget=budget, max_tokens=self.max_tokens,
                             on_event=track if self.on_event is not None else None, should_stop=self.should_stop,
                             max_context_chars=self.max_context_chars)
            if pre.get("s1_error"):   # pas d'indice fabrique : Bonsai sait que le juge rapide est absent
                hint = "(Fast judge unavailable this turn: no hint, decide by yourself. Not exposed but creatable: any tool you need.)"
            else:
                hint = f"(Fast judge: clarify={pre['clarify']['noul']:.2f}, needs_reasoning={pre['needs_reasoning']['noul']:.2f}, risk={risk_level}. Not exposed but creatable: any tool you need.)"
            # apercu de l'espace de travail dans le message (pas dans le prompt systeme : le prefixe mis en cache reste stable)
            # -> souvent un aller-retour d'outil de moins, soit plusieurs secondes sur un 27B
            hint += ("\n(Workspace: " + ", ".join(files[:40]) + (f", ... {len(files) - 40} more" if len(files) > 40 else "") + ")") if files else "\n(Workspace is empty.)"
            res = loop.run([{"role": "system", "content": system}, *history[-6:], {"role": "user", "content": f"{request}\n\n{hint}"}])
            summary = next((c["result"].get("summary") for s in reversed(res.steps) for c in s.tool_calls if c["name"] == "done" and isinstance(c["result"], dict)), None)
            turn = Turn(request, pre, "agent", summary or (res.content or "").strip() or "(no summary)", tool_calls=calls, tools_exposed=exposed,
                        stopped_by=res.stopped_by)
            if turn.stopped_by != "cancelled":
                turn.verification = self._verify(request, turn.response)
        turn.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        stats["s1_ms"], stats["s1_calls"] = round(self._s1_ms, 1), self._s1_calls   # toutes les lectures S1 du tour
        turn.stats = stats
        self._emit({"type": "turn.end", "path": turn.path, "response": turn.response, "verification": turn.verification,
                    "latency_ms": turn.latency_ms, "stopped_by": turn.stopped_by, "stats": stats, "tools_exposed": turn.tools_exposed})
        self._log(turn)
        return turn

    def _log(self, turn: Turn) -> None:
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ledger, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "request": turn.request, "pre": turn.pre, "path": turn.path, "response": turn.response,
                                "tool_calls": turn.tool_calls, "tools_exposed": turn.tools_exposed, "verification": turn.verification,
                                "latency_ms": turn.latency_ms, "stopped_by": turn.stopped_by, "stats": turn.stats},
                               ensure_ascii=False, default=str) + "\n")


def make_browser_factory(s1_engine, s2_backend, headless: bool = True, confirm: Callable[[str, dict], bool] | None = None,
                         on_event: Callable[[dict], None] | None = None, should_stop: Callable[[], bool] | None = None,
                         ledger: str | Path | None = "runs/trajectories.jsonl", max_steps: int = 20):
    """Fabrique l'outil `browse` : agent navigateur a deux vitesses (computer_use) partageant les deux modeles.
    confirm : autorisation des pas risques (sans elle, ils sont refuses) ; on_event / should_stop : progression par pas et
    annulation ; ledger : journal des trajectoires (re-entrainement de la politique rapide, training/make_from_trajectories.py)."""
    def factory():
        from jev_clone.computer_use import BrowserSession, ComputerUseAgent, FastPolicy, SlowPolicy, browser_available, run_result
        def run(goal, url=None, slots=None):
            ok, why = browser_available()
            if not ok:   # erreur explicite pour Bonsai plutot qu'un ModuleNotFoundError
                return {"ok": False, "error": f"browser unavailable: {why}"}
            session = BrowserSession(headless=headless)
            try:
                agent = ComputerUseAgent(session, FastPolicy(s1_engine), SlowPolicy(s2_backend, session, SystemOneToolbox(s1_engine)), max_steps=max_steps,
                                         ledger=ledger, confirm=confirm, on_event=on_event, should_stop=should_stop, kind="browse")
                out = agent.run(goal, url=url, slots=slots or {})
                final = session.observe()
                return {**run_result(out), "url": final.url, "title": final.title, "page": final.aria[:3000]}
            finally:
                session.close()
        return run
    return factory


def repl(prophet: Prophet) -> None:
    history: list[dict] = []
    print(f"Prophet pret. Espace de travail : {prophet.ws.root}. /quit pour quitter.")
    while True:
        try:
            req = input("\nvous > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not req:
            continue
        if req in ("/quit", "/exit"):
            break
        t = prophet.handle(req, history)
        print(f"\nprophet [{t.path}, {t.latency_ms:.0f} ms, verif {t.verification}] > {t.response}")
        if t.tool_calls:
            print("   outils : " + ", ".join(c.get("tool", "?") + (" (bloque)" if c.get("blocked") else "") for c in t.tool_calls))
        history += [{"role": "user", "content": req}, {"role": "assistant", "content": t.response}]


def confirm_in_terminal(describe: str, judged: dict) -> bool:
    tr, risk = judged.get("tool_risk"), judged.get("risk")
    verdict = f"action jugee {tr} (risque {risk:.1f})" if tr and risk is not None else "action non jugee par le classifieur"
    print(f"\n[garde-fou] {verdict} :\n{describe[:600]}")
    try:
        return input("executer ? [o/N] ").strip().lower() in ("o", "y", "oui", "yes")
    except EOFError:
        return False

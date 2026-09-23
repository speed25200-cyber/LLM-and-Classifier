"""Computer use et interface, deuxieme vague de correctifs.

Un pas refuse puis done n'est jamais un succes ni une etiquette done / objectif atteint ; progression par pas rattachee a
l'outil en cours et sauvegardee (miroir TypeScript verifie) ; porte du slot avec marge ; Stop coupe la generation de
Bonsai ; reflexion plafonnee sous max_tokens et reponses tronquees comptees comme echecs ; navigateur absent jamais
propose ; lectures S1 du run rendues par browse / desktop. Interface (carte d'outil, modeles, inspecteur, reglages)
verifiee dans Chromium sur une construction temporaire de ui/ (ignore sans Node, node_modules ou navigateur)."""

import contextlib
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from jev_clone.backend_llamacpp import LlamaCppBackend
from jev_clone.computer_use import (ComputerUseAgent, Element, FastPolicy, PageState, SlowPolicy, run_result,
                                    trajectory_to_examples)
from jev_clone.desktop_use import SimulatedDesktop, make_desktop_factory
from jev_clone.engine import SystemOneEngine
from prophet_studio.sessions import CU_KEEP, AgentService, SessionStore, reduce_event
from tests.conftest import MockS2, ScriptedBackend
from tests.test_fix_computer_use_voice import A, CLEAR, EQUALS, FIVE, ONE, PLUS, act, agent, guard_on, s1, tc

ROOT = Path(__file__).resolve().parents[1]


class FakeSession:
    """Page fixe (formulaire de connexion) : observe / act sans navigateur."""

    def __init__(self, st: PageState):
        self.st, self.last, self.acts = st, st, []

    def goto(self, url):
        pass

    def observe(self):
        return self.st

    def act(self, a):
        self.acts.append(a)
        return {"ok": True}


LOGIN = PageState("https://site/login", "Login", "", [Element(0, "input:text", "Email"), Element(1, "input:password", "Password"),
                                                      Element(2, "button", "Sign in")])
CREDS = {"email": "bob@example.com", "password": "hunter2"}


def login_engine(slot_probs):
    table = {"action": [0.02, 0.9, 0.02, 0.02, 0.02, 0.02], "achieved": [0.05, 0.95], "t0": [0.95, 0.05], "t1": [0.1, 0.9],
             "t2": [0.05, 0.95], "slot": slot_probs}   # action "type" sure, champ Email sur
    return SystemOneEngine(ScriptedBackend([table], declared={"action": A, "slot": ["email", "password", "none"]}), model_name="mock")


# ---- 1. un pas refuse suivi de done n'est pas un succes, ni une etiquette done / atteint ----------------------------------
def test_declined_step_then_done_is_not_a_success_and_never_labels_done_achieved(tmp_path):
    desk = SimulatedDesktop(); desk.display = "42"
    eng, _ = s1([act("escalate")], guard=guard_on("Effacer"))
    s2 = MockS2([tc("click", {"index": CLEAR}), tc("done", {"summary": "The user declined the reset; the goal is NOT achieved."}, "c2")])
    ledger = tmp_path / "desktop_trajectories.jsonl"
    r = make_desktop_factory(eng, s2, lambda: desk, confirm=lambda d, j: False, ledger=ledger)()("remets la calculatrice a zero")
    assert desk.display == "42" and r["ok"] is False and r["status"] == "blocked" and r["blocked_steps"]
    done = next(d for d in s2.calls[0][1]["tools"] if d["function"]["name"] == "done")["function"]["parameters"]
    assert done["required"] == ["summary", "achieved"] and done["properties"]["achieved"]["type"] == "boolean"
    traj = json.loads(ledger.read_text(encoding="utf-8").splitlines()[0])
    assert traj["status"] == "blocked"
    ex = trajectory_to_examples(traj)
    assert ex and not any(e["labels"]["action"] == "done" or e["labels"]["achieved"] for e in ex)
    assert ex[0]["labels"]["action"] == "escalate" and ex[0]["declined"] is True
    # non atteint dit explicitement (rien de refuse) : pas un succes non plus
    s2b = MockS2([tc("done", {"summary": "introuvable", "achieved": False})])
    r2 = make_desktop_factory(s1([act("escalate")])[0], s2b, SimulatedDesktop)()("ouvre un fichier qui n'existe pas")
    assert (r2["status"], r2["ok"], r2["summary"]) == ("not_achieved", False, "introuvable")
    # pas approuve puis done(achieved=true) : succes, et la bonne action est l'etiquette
    desk3 = SimulatedDesktop(); desk3.display = "42"
    s2c = MockS2([tc("click", {"index": CLEAR}), tc("done", {"summary": "remis a zero", "achieved": True}, "c2")])
    out3 = agent(desk3, s1([act("escalate")], guard=guard_on("Effacer"))[0], s2c, max_steps=3, confirm=lambda d, j: True).run("remets a zero")
    assert out3["status"] == "done" and run_result(out3)["ok"] and desk3.display == "0"
    e3 = trajectory_to_examples(out3)
    assert e3[0]["labels"]["action"] == "click" and "declined" not in e3[0]


def test_old_ledgers_without_achieved_are_relabelled():
    st = {"goal": "g", "available_slots": {}, "last_actions": [], "url": "app://calc.exe", "title": "Calculatrice", "page": "",
          "elements": ['[0] button "Effacer"']}
    rec = lambda acts: {"records": [{"step": 0, "path": "escalated", "state": st, "slow": {"actions": acts}}]}   # noqa: E731
    refused = trajectory_to_examples(rec([{"type": "click", "target": 0, "blocked": True}, {"type": "done", "summary": "refuse"}]))
    assert refused[0]["labels"]["action"] == "escalate" and refused[0]["labels"]["achieved"] is False and refused[0]["declined"]
    assert trajectory_to_examples(rec([{"type": "done", "summary": "ok"}]))[0]["labels"] == {"action": "done", "achieved": True, "t0": False}
    said_no = trajectory_to_examples(rec([{"type": "done", "summary": "non", "achieved": "false"}]))[0]["labels"]
    assert said_no["action"] == "escalate" and said_no["achieved"] is False


# ---- 2. porte du slot : probabilite ET marge sur le deuxieme slot --------------------------------------------------------------
def test_a_slot_near_tie_escalates_instead_of_typing_the_wrong_value():
    d = FastPolicy(login_engine([0.47, 0.51, 0.02])).decide("sign in with my email", LOGIN, [], CREDS)
    assert d.escalate and "slot" in d.why and d.slot_margin < 0.1
    ok = FastPolicy(login_engine([0.9, 0.08, 0.02])).decide("sign in with my email", LOGIN, [], CREDS)
    assert not ok.escalate and (ok.action, ok.target, ok.slot) == ("type", 0, "email") and ok.slot_margin > 0.5
    sess = FakeSession(LOGIN)
    s2 = MockS2([tc("done", {"summary": "a verifier", "achieved": False})])
    out = ComputerUseAgent(sess, FastPolicy(login_engine([0.47, 0.51, 0.02])), SlowPolicy(s2, sess), max_steps=2, ledger=None).run(
        "sign in with my email", slots=CREDS)
    assert sess.acts == [] and out["records"][0]["path"] == "escalated" and out["records"][0]["fast"]["slot_margin"] < 0.1


# ---- 3. Stop coupe la generation de Bonsai (flux SSE ferme), sans attendre la fin -----------------------------------------------
class SlowLlama(BaseHTTPRequestHandler):
    """llama-server qui met GEN_S secondes a generer : en flux, un delta de reflexion toutes les 50 ms."""
    protocol_version = "HTTP/1.1"
    GEN_S = 6.0
    seen: list = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        type(self).seen.append({k: body.get(k) for k in ("stream", "max_tokens", "thinking_budget_tokens")})
        call = {"id": "c1", "type": "function", "function": {"name": "scroll_down", "arguments": "{}"}}
        if not body.get("stream"):
            time.sleep(self.GEN_S)
            b = json.dumps({"choices": [{"index": 0, "message": {"role": "assistant", "content": "", "tool_calls": [call]},
                                         "finish_reason": "tool_calls"}]}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b)))
            self.end_headers(); self.wfile.write(b)
            return
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def send(obj) -> bool:
            data = f"data: {json.dumps(obj) if not isinstance(obj, str) else obj}\n\n".encode()
            try:
                self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n"); self.wfile.flush(); return True
            except OSError:
                return False
        t0 = time.time()
        while time.time() - t0 < self.GEN_S:
            if not send({"choices": [{"index": 0, "delta": {"reasoning_content": "hmm "}}]}):
                return   # client parti : la generation s'arrete
            time.sleep(0.05)
        send({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, **call}]}, "finish_reason": "tool_calls"}]})
        send("[DONE]")
        try:
            self.wfile.write(b"0\r\n\r\n"); self.wfile.flush()
        except OSError:
            pass


def test_stop_interrupts_the_inner_bonsai_generation():
    SlowLlama.seen = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), SlowLlama)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        s2 = LlamaCppBackend(f"http://127.0.0.1:{srv.server_address[1]}", max_workers=1, timeout=60)
        stop = threading.Event()
        run = make_desktop_factory(s1([act("escalate")])[0], s2, SimulatedDesktop, should_stop=stop.is_set)()
        threading.Timer(0.5, stop.set).start()
        t0 = time.time()
        r = run("tache longue")
        elapsed = time.time() - t0
    finally:
        srv.shutdown(); srv.server_close()
    assert r["status"] == "cancelled" and not r["ok"]
    assert SlowLlama.seen[0]["stream"] is True and elapsed < 3.0, (elapsed, SlowLlama.seen)   # et non GEN_S = 6 s


# ---- 4. reflexion plafonnee sous max_tokens ; une reponse coupee est un echec de Bonsai ------------------------------------------
class TruncatedS2:
    """Bonsai qui reflechit jusqu'a max_tokens : reponse coupee ('length'), aucune action."""

    def __init__(self):
        self.calls = []

    def chat(self, messages, **kw):
        self.calls.append(kw)
        return {"choices": [{"message": {"role": "assistant", "content": "", "reasoning_content": "..." * 500}, "finish_reason": "length"}]}


def test_bonsai_thinking_stays_below_its_generation_and_truncated_replies_stop_the_run():
    s2 = MockS2([tc("done", {"summary": "ok", "achieved": True})])
    make_desktop_factory(s1([act("escalate")])[0], s2, SimulatedDesktop, max_tokens=3000)()("x")
    kw = s2.calls[0][1]
    assert kw["max_tokens"] == 3000 and 0 < kw["thinking_budget"] <= 0.6 * kw["max_tokens"]
    sp = SlowPolicy(MockS2([{}]), FakeSession(LOGIN))                  # valeurs par defaut : 2048 de reflexion demandes
    assert 0 < sp.loop.thinking_budget <= 0.6 * sp.loop.max_tokens
    t = TruncatedS2()
    out = agent(SimulatedDesktop(), s1([act("escalate")])[0], t, max_steps=20).run("x")
    assert out["status"] == "s2_error" and len(t.calls) == 2           # et non 8 escalades brulees


# ---- 5. navigateur indisponible : l'outil n'est jamais propose a Bonsai ----------------------------------------------------------
def test_an_enabled_browser_tool_is_not_offered_without_playwright_or_chromium(tmp_path, monkeypatch):
    import jev_clone.computer_use as cu
    import prophet_studio.sessions as ss
    from prophet_studio.config import Settings
    got = {}

    class FakeProphet:
        def __init__(self, *a, **kw):
            got["prophet"] = kw

        def handle(self, text, history, effort="auto"):
            return types.SimpleNamespace(response="ok")
    monkeypatch.setattr(ss, "Prophet", FakeProphet)
    monkeypatch.setattr(ss, "make_browser_factory", lambda s1_, s2_, **kw: got.update(browser=kw) or (lambda: None))
    for available in (False, True):
        monkeypatch.setattr(cu, "browser_available", lambda a=available: (a, "" if a else "Playwright n'est pas installe"))
        got.clear()
        st = Settings(browser_tool=True, workspace=str(tmp_path / "ws"))
        svc = AgentService(SessionStore(tmp_path / "sessions"), lambda e: None, urls=lambda: ("http://127.0.0.1:9", "http://127.0.0.1:9"),
                           settings=lambda st=st: st, ctx=lambda: 8192)
        sid = svc.store.create(str(tmp_path / "ws"))["id"]
        svc.submit(sid, "cherche")
        assert svc.wait_idle(sid, 10)
        if available:   # meme generation (et meme part de reflexion) que Prophet
            assert got["prophet"]["browser_factory"] is not None and got["browser"]["max_tokens"] == got["prophet"]["max_tokens"]
        else:
            assert got["prophet"]["browser_factory"] is None and "browser" not in got


# ---- 6. lectures S1 du run (decisions, garde-fou, verifications, judge_* de Bonsai) ------------------------------------------------
class CountingS1:
    def __init__(self, inner, delay: float = 0.002):
        self.inner, self.delay, self.n = inner, delay, 0

    def answer(self, req):
        self.n += 1
        time.sleep(self.delay)
        return self.inner.answer(req)


def test_browse_and_desktop_report_the_s1_time_and_calls_of_the_run():
    ce = CountingS1(s1([act("click", ONE), act("click", PLUS), act("click", FIVE), act("click", EQUALS), act("done", achieved=0.9)])[0])
    r = make_desktop_factory(ce, MockS2([{"content": "inutile"}]), SimulatedDesktop)()("calcule 1 + 5")
    assert r["ok"] and ce.n == 5 + 4 + 4                                 # 5 decisions, 4 clics juges, 4 verifications
    assert r["s1_calls"] == ce.n and isinstance(r["s1_ms"], float) and r["s1_ms"] >= ce.n * 2 * 0.9
    ce2 = CountingS1(s1([act("escalate")])[0])
    s2 = MockS2([tc("judge_noul", {"state": "calculatrice", "question": "Affiche-t-elle 0 ?"}), tc("done", {"summary": "ok", "achieved": True}, "c2")])
    r2 = make_desktop_factory(ce2, s2, SimulatedDesktop)()("verifie")
    assert r2["ok"] and r2["s1_calls"] == ce2.n == 2                     # la decision + le judge_noul de Bonsai


# ---- 7. progression par pas : rattachee a l'outil en cours, bornee, sauvegardee avec la session ------------------------------------
def desktop_turn(tmp_path, monkeypatch):
    """Tour reel de l'outil desktop dans AgentService (Prophet factice qui appelle l'outil) : evenements publies + session."""
    import prophet_studio.sessions as ss
    from prophet_studio.config import Settings
    published = []

    class FakeProphet:
        def __init__(self, *a, **kw):
            self.kw = kw

        def handle(self, text, history, effort="auto"):
            emit = self.kw["on_event"]
            emit({"type": "tool.call", "id": "c1", "name": "desktop", "args": {"goal": text}})
            r = self.kw["desktop_factory"]()(text, None, {})
            emit({"type": "tool.result", "id": "c1", "name": "desktop", "ok": r["ok"], "result": r})
            return types.SimpleNamespace(response="ok")
    monkeypatch.setattr(ss, "Prophet", FakeProphet)
    st = Settings(desktop_tool=True, workspace=str(tmp_path / "ws"), permission_mode="auto")
    svc = AgentService(SessionStore(tmp_path / "sessions"), published.append, urls=lambda: ("http://127.0.0.1:9", "http://127.0.0.1:9"),
                       settings=lambda: st, ctx=lambda: 8192, desktop_backend=SimulatedDesktop, runs_dir=tmp_path / "runs")
    eng, _ = s1([act("click", ONE), act("escalate")])
    s2 = MockS2([tc("click", {"index": PLUS}), tc("done", {"summary": "1+", "achieved": True}, "c2")])
    svc.engines = lambda: (eng, s2)
    sid = svc.store.create(str(tmp_path / "ws"))["id"]
    svc.submit(sid, "tape 1 +")
    assert svc.wait_idle(sid, 20)
    return published, svc.store.load(sid)


def test_computer_progress_is_attached_to_the_running_tool_and_saved(tmp_path, monkeypatch):
    published, session = desktop_turn(tmp_path, monkeypatch)
    kinds = [e["type"] for e in published if e["type"].startswith("computer.")]
    assert kinds == ["computer.step", "computer.escalate", "computer.think", "computer.action", "computer.think", "computer.step"]
    tool = next(b for b in session["transcript"][-1]["blocks"] if b["type"] == "tool")
    cu = tool["cu"]
    assert tool["status"] == "done" and cu["n"] == 2 and cu["escalations"] == 1 and cu["live"] is None and cu["pending"] == []
    fast, slow = cu["steps"]
    assert (fast["path"], fast["action"], fast["step"]) == ("fast", "click", 0) and fast["p"] >= 0.8 and fast["slow"] == []
    assert slow["path"] == "escalated" and slow["why"] == "the fast policy asked for help"
    assert slow["slow"] == [{"action": "click", "blocked": False, "ok": True}] and slow["escalations"] == 1
    # en direct : entre deux pas, l'etat courant (escalade, tour de Bonsai, action) est visible
    item: dict = {"blocks": []}
    seen = []
    for e in published:
        reduce_event(item, e)
        if e["type"] in ("computer.think", "computer.action"):
            live = item["blocks"][0]["cu"]["live"]
            seen.append((live["kind"], live["step"], live["turn"], live["action"]))
            assert "help" in live["why"]
    assert seen == [("think", 1, 0, None), ("action", 1, 0, "click"), ("think", 1, 1, "click")]


def test_computer_progress_is_bounded_and_only_goes_to_the_running_tool_of_the_same_name():
    item: dict = {"blocks": []}
    reduce_event(item, {"type": "computer.step", "tool": "desktop", "step": 0, "path": "fast"})     # aucun outil en cours
    reduce_event(item, {"type": "tool.call", "id": "a", "name": "desktop", "args": {}})
    reduce_event(item, {"type": "computer.step", "tool": "browse", "step": 0, "path": "fast"})      # autre outil
    for k in range(CU_KEEP + 10):
        reduce_event(item, {"type": "computer.step", "tool": "desktop", "step": k, "path": "fast", "action": "click", "p": 0.9,
                            "escalations": 0, "why": None})
    cu = item["blocks"][0]["cu"]
    assert cu["n"] == CU_KEEP + 10 and len(cu["steps"]) == CU_KEEP and cu["steps"][0]["step"] == 10
    reduce_event(item, {"type": "tool.result", "id": "a", "ok": True, "result": {}})
    reduce_event(item, {"type": "computer.step", "tool": "desktop", "step": 99})
    assert item["blocks"][0]["cu"]["n"] == CU_KEEP + 10


def _node_reduce(work: Path, events: list[dict]) -> list:
    """Rejoue les evenements dans ui/src/lib/transcript.ts (Node, types effaces) ; ignore sans Node recent."""
    node = shutil.which("node")
    if not node:
        pytest.skip("Node absent")
    tmp_path = Path(tempfile.mkdtemp(dir=work))
    lib = tmp_path / "lib"
    lib.mkdir()
    for f in (ROOT / "ui" / "src" / "lib").glob("*.ts"):   # specificateurs relatifs completes en .ts (resolution ESM de Node)
        src = f.read_text(encoding="utf-8")
        (lib / f.name).write_text(re.sub(r'(from\s+"\./[\w.-]+?)(?<!\.ts)"', r'\1.ts"', src), encoding="utf-8")
    (tmp_path / "events.json").write_text(json.dumps(events), encoding="utf-8")
    script = tmp_path / "run.mjs"
    script.write_text('import { readFileSync } from "node:fs";\nimport { pathToFileURL } from "node:url";\n'
                      'const m = await import(pathToFileURL(process.argv[2]).href);\n'
                      'const item = m.newAssistant("t");\n'
                      'for (const e of JSON.parse(readFileSync(process.argv[3], "utf8"))) m.reduce(item, e);\n'
                      'console.log(JSON.stringify(item.blocks.filter((b) => b.type === "tool").map((b) => b.cu ?? null)));\n', encoding="utf-8")
    for flags in ([], ["--experimental-strip-types"]):
        p = subprocess.run([node, *flags, str(script), str(lib / "transcript.ts"), str(tmp_path / "events.json")],
                           capture_output=True, text=True, encoding="utf-8", timeout=60)
        if p.returncode == 0:
            return json.loads(p.stdout.strip().splitlines()[-1])
    pytest.skip(f"Node ne sait pas executer du TypeScript : {p.stderr[-300:]}")


def test_the_ui_reducer_mirrors_the_server_for_computer_progress(tmp_path, monkeypatch):
    published, session = desktop_turn(tmp_path, monkeypatch)
    ui = _node_reduce(tmp_path, published)
    server = [b.get("cu") for b in session["transcript"][-1]["blocks"] if b["type"] == "tool"]
    assert ui == server and ui[0]["n"] == 2
    item: dict = {"blocks": []}
    evs = [{"type": "tool.call", "id": "a", "name": "browse", "args": {}},
           {"type": "computer.escalate", "tool": "browse", "step": 3, "why": "x" * 300, "escalations": 2},
           {"type": "computer.think", "tool": "browse", "turn": 1},
           {"type": "computer.action", "tool": "browse", "action": "type", "blocked": True}]
    for e in evs:
        reduce_event(item, e)
    assert _node_reduce(tmp_path, evs) == [item["blocks"][0]["cu"]]


# ---- 8. interface : ui/src construit a part (jamais dans prophet_studio/web), servi par le coeur, pilote par Chromium --------------
CHROMIUM = [os.environ.get("JEV_CHROMIUM", ""), "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"]


@pytest.fixture(scope="module")
def ui_web(tmp_path_factory):
    pytest.importorskip("playwright.sync_api")
    vite = ROOT / "ui" / "node_modules" / ".bin" / "vite"
    if os.name == "nt" or not vite.exists() or not shutil.which("node"):
        pytest.skip("interface non constructible ici (cd ui && npm ci)")
    out = tmp_path_factory.mktemp("web")
    p = subprocess.run([str(vite), "build", "--outDir", str(out), "--emptyOutDir", "--logLevel", "error"], cwd=ROOT / "ui",
                       capture_output=True, text=True, encoding="utf-8", timeout=600)
    assert p.returncode == 0, p.stderr[-2000:]
    return out


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def studio_page(tmp_path, monkeypatch, web, gpu="NVIDIA GeForce RTX 5060:8151:650:display", settings=None, prepare=None, size=(1280, 800)):
    """Coeur de demonstration (modeles arretes) servant la construction temporaire ; rend (studio, page)."""
    import uvicorn
    from playwright.sync_api import sync_playwright

    import prophet_studio.catalog as catalog
    import prophet_studio.server as server
    from prophet_studio.config import Paths
    monkeypatch.setattr(server, "WEB_DIR", web)
    monkeypatch.setenv("PROPHET_FAKE_GPU", gpu)
    customs = set(catalog.CUSTOM)
    paths = Paths(tmp_path / "home")
    paths.settings.write_text(json.dumps({"workspace": str(tmp_path / "ws"), "s2_port": free_port(), "s1_port": free_port(),
                                          "voice": {"enabled": False}, "onboarding_done": True, "autostart_models": False, **(settings or {})}))
    port = free_port()
    st = server.Studio(paths, "tok-ui", port, demo=True)
    srv = uvicorn.Server(uvicorn.Config(server.build_app(st), host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=srv.run, daemon=True)
    errors: list[str] = []
    try:
        if prepare:
            prepare(st)
        th.start()
        t0 = time.time()
        while not srv.started and time.time() - t0 < 30:
            time.sleep(0.05)
        with sync_playwright() as p:
            browser = None
            for exe in [None] + [c for c in CHROMIUM if c and Path(c).exists()]:
                try:
                    browser = p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()
                    break
                except Exception:
                    continue
            if browser is None:
                pytest.skip("aucun Chromium disponible")
            page = browser.new_page(viewport={"width": size[0], "height": size[1]})
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"http://127.0.0.1:{port}/")
            yield st, page
            browser.close()
        assert errors == []
    finally:
        srv.should_exit = True
        if th.is_alive():
            th.join(timeout=10)
        st.runtime.stop()
        for k in set(catalog.CUSTOM) - customs:   # GGUF importes : jamais visibles des autres tests
            catalog.CUSTOM.pop(k, None)


def wait_until(fn, timeout: float = 10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = fn()
        if v:
            return v
        time.sleep(0.05)
    return fn()


LIVE = [{"type": "tool.call", "id": "c1", "name": "desktop", "args": {"goal": "remets la calculatrice a zero"}},
        {"type": "computer.step", "tool": "desktop", "step": 0, "path": "fast", "action": "click", "p": 0.9, "escalations": 0},
        {"type": "computer.escalate", "tool": "desktop", "step": 1, "why": "risky step proposed by the fast policy", "escalations": 1},
        {"type": "computer.think", "tool": "desktop", "turn": 0}]
END = [{"type": "computer.action", "tool": "desktop", "action": "click", "blocked": True},
       {"type": "computer.step", "tool": "desktop", "step": 1, "path": "escalated", "action": "click", "p": 0.88, "escalations": 1,
        "why": "risky step proposed by the fast policy", "slow_actions": ["click", "done"]},
       {"type": "tool.result", "id": "c1", "name": "desktop", "ok": False,
        "result": {"ok": False, "status": "blocked", "steps": 2, "fast_steps": 1, "escalations": 1, "summary": "refuse par l'utilisateur",
                   "blocked_steps": [{"type": "click", "target": 15, "blocked": True}], "s1_ms": 812.5, "s1_calls": 9,
                   "window": "Calculatrice", "app": "app://calc.exe"}}]


def test_ui_shows_computer_progress_live_and_reloaded_unclipped_grants_and_unavailable_tools(tmp_path, monkeypatch, ui_web):
    import jev_clone.computer_use as cu
    monkeypatch.setattr(cu, "browser_available", lambda: (False, "Playwright n'est pas installe"))
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "index.html").write_text("\n".join(f"<p>ligne {i}</p>" for i in range(300)), encoding="utf-8")
    for i in range(6):
        (ws / f"note_{i}.md").write_text("x", encoding="utf-8")
    sid = {}

    def prepare(st):
        s = st.sessions.create(str(ws))
        item = {"kind": "assistant", "turn_id": "t0", "blocks": [], "status": "done", "ts": time.time()}
        for e in LIVE + END:
            reduce_event(item, e)
        s["transcript"] = [{"kind": "user", "text": "remets a zero", "ts": time.time(), "turn_id": "t0"}, item]
        s["always_allow"] = ["write_file:workspace_write", "run_command:readonly"]
        st.sessions.save(s)
        sid["id"] = s["id"]
    with studio_page(tmp_path, monkeypatch, ui_web, settings={"browser_tool": True}, prepare=prepare) as (st, page):
        # session rechargee : les pas sauvegardes s'affichent, avec le statut en clair (pas refuse, jamais un succes)
        card = page.locator(".tool").first
        card.locator(".cs").nth(1).wait_for(timeout=15000)
        rows = card.locator(".cs").all_inner_texts()
        assert "voie rapide" in rows[0] and "click" in rows[0] and "90" in rows[0] and "Bonsai" in rows[1] and "click (refuse)" in rows[1]
        body = card.inner_text()
        assert "pas refuse : objectif non atteint" in body and "refuse par l'utilisateur" in body and "Pas refuses : click [15]" in body
        assert "classifieur : 9 lectures" in body
        # en direct : un nouveau tour publie ses evenements, la carte les montre avant la fin de l'outil
        for e in LIVE:
            st.bus.publish({**e, "session_id": sid["id"], "turn_id": "t1"})
        running = page.locator(".tool.running")
        running.locator(".live").wait_for(timeout=10000)
        assert wait_until(lambda: "pas 2 · Bonsai reflechit" in running.locator(".live").inner_text())
        assert running.locator(".cs").count() == 1 and "voie rapide" in running.locator(".cs").first.inner_text()
        assert "1 escalade" in running.inner_text()
        st.bus.publish({**END[0], "session_id": sid["id"], "turn_id": "t1"})
        assert wait_until(lambda: "Bonsai : click (refuse)" in running.locator(".live").inner_text())
        # inspecteur : fichier ouvert, les deux autorisations memorisees restent entieres
        page.keyboard.press("Control+b")
        page.locator("aside .node", has_text="index.html").click()
        page.locator("aside .code .ln").nth(10).wait_for(timeout=10000)
        g = page.locator(".grants")
        g.locator(".grant").nth(1).wait_for(timeout=10000)
        h = g.evaluate("e => [e.clientHeight, e.scrollHeight]")
        assert h[0] >= h[1] - 1, h
        # reglages : un outil actif mais indisponible est dit non propose ; decimales a la francaise
        page.get_by_role("button", name="Reglages", exact=True).click()
        text = page.locator(".page").inner_text()
        assert "Active mais indisponible ici, donc non propose a Bonsai : Playwright n'est pas installe" in text
        assert "Debit de la voix · 1,00x" in text


def test_ui_models_reset_a_removed_choice_warn_clearly_and_show_the_bench_series(tmp_path, monkeypatch, ui_web):
    brain, clone = tmp_path / "my-brain.gguf", tmp_path / "gone" / "my-clone.gguf"
    clone.parent.mkdir()
    for f in (brain, clone):   # fichiers creux de 1 Gio (un GGUF de quelques octets fait diviser le planificateur par zero)
        with open(f, "wb") as fh:
            fh.truncate(2**30)
    ids = {}

    def prepare(st):
        ids["s2"] = st.installer.import_gguf(str(brain), "s2")
        ids["s1"] = st.installer.import_gguf(str(clone), "s1")
        st.settings.update({"s2_model": ids["s2"], "s1_model": ids["s1"]})
        shutil.rmtree(clone.parent)   # GGUF deplace hors de l'application : choix non applique
        st.last_bench = {"ts": time.time(), "s1_p50_ms": 41.3, "s1_p95_ms": 55.0, "s1_cold_p50_ms": 812.4, "s1_cold_p95_ms": 950.0,
                         "s1_cold_prefill_tokens": 2100, "s1_warm_p50_ms": 95.2, "s1_warm_p95_ms": 120.0, "s1_warm_prefill_tokens": 40,
                         "s1_series": {"s1_cold": "etat neuf"}, "s2_tok_s": 41.5, "s2_prefill_tok_s": 900.0, "s2_total_s": 4.1,
                         "plan": None, "gpu": "RTX 5060", "demo": True}
    with studio_page(tmp_path, monkeypatch, ui_web, prepare=prepare) as (st, page):
        page.get_by_role("button", name="Modeles", exact=True).click()
        bench = page.locator(".bench")
        bench.wait_for(timeout=15000)
        assert "etat neuf p50 812,4 ms" in bench.inner_text() and "relu p50 95,2 ms" in bench.inner_text()
        # GGUF introuvable : texte et chemin en un seul bloc qui s'etire, icone et bouton a cote
        missing = page.locator(".warnbox", has_text="GGUF importe introuvable")
        missing.wait_for(timeout=10000)
        geo = missing.evaluate("e => ({n: e.children.length, box: e.getBoundingClientRect().width, "
                               "txt: e.querySelector('.wtxt') ? e.querySelector('.wtxt').getBoundingClientRect().width : 0})")
        assert geo["n"] == 3 and geo["txt"] >= 0.6 * geo["box"], geo
        # choix non applique : retour a l'automatique en un clic
        page.locator(".warnbox", has_text="non applique").get_by_role("button", name="Revenir au choix automatique").click()
        assert wait_until(lambda: st.settings.get().s1_model == "auto")
        # le cerveau importe et choisi, retire de la liste : le reglage revient a l'automatique
        page.locator(".model", has_text="my-brain").get_by_title("Retirer de la liste (le fichier GGUF est conserve)").click()
        assert wait_until(lambda: st.settings.get().s2_model == "auto"), st.settings.get().s2_model
        assert wait_until(lambda: page.locator(".warnbox", has_text="non applique").count() == 0)


def test_ui_a_cpu_plan_that_does_not_fit_in_ram_is_never_started_blindly(tmp_path, monkeypatch, ui_web):
    huge = tmp_path / "huge-brain.gguf"
    ids = {}

    def prepare(st):
        with open(huge, "wb") as f:   # fichier creux, 4 x la RAM de la machine
            f.truncate(int(st.hw.ram_total_gib * 4 * 2**30))
        ids["s2"] = st.installer.import_gguf(str(huge), "s2")
        st.settings.update({"s2_model": ids["s2"]})
    with studio_page(tmp_path, monkeypatch, ui_web, gpu="NVIDIA GeForce GT 710:2048:14", prepare=prepare) as (st, page):
        plan = st.plan()
        assert plan.backend == "cpu" and not plan.fits and plan.s2.model_id == ids["s2"]
        page.get_by_role("button", name="Modeles", exact=True).click()
        warn = page.locator(".warnbox.ram")
        warn.wait_for(timeout=15000)
        assert "RAM insuffisante" in warn.inner_text() and warn.get_by_role("button", name="Lancer quand meme").count() == 1
        assert page.get_by_role("button", name="Demarrer", exact=True).is_disabled()
        assert page.get_by_role("button", name="Appliquer et redemarrer").count() == 0

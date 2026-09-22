"""Flux d'evenements de l'agent, diffusion SSE, annulation, compaction, modes d'autorisation et outils Claude-Code-like,
contre le faux llama-server (meme API HTTP que llama-server)."""

import json

import pytest

from jev_clone.backend_llamacpp import LlamaCppBackend
from jev_clone.engine import SystemOneEngine
from jev_clone.prophet import Prophet, Workspace
from jev_clone.tools import AgentLoop, _tool
from prophet_studio.demo.fake_llama import FakeLlama


@pytest.fixture(scope="module")
def fake_url():
    srv, _ = FakeLlama().serve()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_stream_reassembles_reasoning_content_and_tool_calls(fake_url):
    be = LlamaCppBackend(fake_url)
    tools = [_tool("glob", "find", {"pattern": {"type": "string"}}, ["pattern"]), _tool("write_file", "w", {}, []), _tool("done", "d", {}, [])]
    deltas = []
    r = be.chat([{"role": "user", "content": "Cree une app minuteur"}], tools=tools, on_delta=deltas.append)
    msg = r["choices"][0]["message"]
    assert msg["tool_calls"][0]["function"]["name"] == "glob"
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"pattern": "*"}
    assert msg["reasoning_content"].startswith("Je regarde")
    assert "".join(d["text"] for d in deltas if d["type"] == "reasoning") == msg["reasoning_content"]
    assert any(d["type"] == "tool_call" and d["name"] == "glob" for d in deltas)
    assert r["timings"]["predicted_per_second"] > 0
    # sans flux : meme forme
    r2 = be.chat([{"role": "user", "content": "Cree une app minuteur"}], tools=tools)
    assert r2["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "glob"


def test_stream_can_be_cancelled(fake_url):
    be = LlamaCppBackend(fake_url)
    seen = []
    r = be.chat([{"role": "user", "content": "bonjour"}], on_delta=seen.append, should_stop=lambda: len(seen) >= 2)
    assert r["cancelled"] is True and r["choices"][0]["finish_reason"] == "cancelled"


def test_prophet_full_agent_turn_emits_events_and_writes_files(fake_url, tmp_path):
    ws = Workspace(tmp_path / "ws")
    events = []
    s1 = SystemOneEngine(LlamaCppBackend(fake_url), model_name="fake")
    p = Prophet(s1, LlamaCppBackend(fake_url), ws, on_event=events.append, max_context_chars=20000)
    t = p.handle("Cree une app minuteur pomodoro en HTML")
    types = [e["type"] for e in events]
    assert types[0] == "turn.start" and types[-1] == "turn.end"
    dec = next(e for e in events if e["type"] == "s1.decision")
    assert dec["path"] == "agent" and dec["pre"]["direct"]["noul"] < 0.5
    assert "s1.tools" in types and "thinking.delta" in types
    calls = [e["name"] for e in events if e["type"] == "tool.call"]
    assert calls == ["glob", "write_file", "python", "done"]
    wr = next(e for e in events if e["type"] == "tool.result" and e["name"] == "write_file")
    assert wr["ok"] and wr["ui"]["created"] and "+<!doctype html>" in wr["ui"]["diff"] and "__ui__" not in wr["result"]
    py = next(e for e in events if e["type"] == "tool.result" and e["name"] == "python")
    assert "index.html" in py["result"]["stdout"]
    assert (ws.root / "index.html").exists() and t.path == "agent" and "index.html" in t.response
    assert t.stats["llm_calls"] == 4 and t.stats["tok_s"] > 0 and t.verification > 0.5
    assert t.stats["ctx_tokens"] > 500   # contexte occupe (prompt + generation) remonte a l interface


def test_direct_path_streams_text(fake_url, tmp_path):
    events = []
    s1 = SystemOneEngine(LlamaCppBackend(fake_url), model_name="fake")
    t = Prophet(s1, LlamaCppBackend(fake_url), Workspace(tmp_path / "ws"), on_event=events.append).handle("bonjour, qui es-tu ?")
    assert t.path == "direct" and t.response.startswith("Bonjour")
    assert "".join(e["text"] for e in events if e["type"] == "text.delta").strip() == t.response


def test_plan_mode_blocks_changes_and_returns_plan(fake_url, tmp_path):
    ws = Workspace(tmp_path / "ws")
    s1 = SystemOneEngine(LlamaCppBackend(fake_url), model_name="fake")
    t = Prophet(s1, LlamaCppBackend(fake_url), ws, plan_mode=True).handle("Cree une app minuteur")
    assert t.response.startswith("Plan propose") and not (ws.root / "index.html").exists()


def test_ask_mode_requests_confirmation_with_diff_preview(fake_url, tmp_path):
    ws = Workspace(tmp_path / "ws")
    asked = []
    s1 = SystemOneEngine(LlamaCppBackend(fake_url), model_name="fake")
    p = Prophet(s1, LlamaCppBackend(fake_url), ws, permission_mode="ask", confirm=lambda d, j: asked.append(j) or False)
    t = p.handle("Cree une app minuteur")
    wf = next(j for j in asked if j["tool"] == "write_file")
    assert "+<!doctype html>" in wf["preview"]["diff"] and not (ws.root / "index.html").exists()
    assert any(c.get("blocked") for c in t.tool_calls)


def test_cancel_stops_agent_loop(fake_url, tmp_path):
    ws = Workspace(tmp_path / "ws")
    n = {"calls": 0}

    def ev(e):
        if e["type"] == "tool.result":
            n["calls"] += 1
    s1 = SystemOneEngine(LlamaCppBackend(fake_url), model_name="fake")
    t = Prophet(s1, LlamaCppBackend(fake_url), ws, on_event=ev, should_stop=lambda: n["calls"] >= 1).handle("Cree une app minuteur")
    assert t.stopped_by == "cancelled" and n["calls"] == 1 and t.verification is None


def test_compaction_keeps_recent_and_shrinks_old_tool_results():
    loop = AgentLoop(None, max_context_chars=2000, keep_recent=2)
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    msgs += [{"role": "tool", "content": "x" * 1500} for _ in range(4)]
    out = loop.compact(msgs)
    assert out[-1]["content"] == "x" * 1500 and out[-2]["content"] == "x" * 1500
    assert "compacte" in out[2]["content"] and len(out[2]["content"]) < 400
    assert msgs[2]["content"] == "x" * 1500  # l'original n'est pas modifie


def test_workspace_edit_grep_glob(tmp_path):
    ws = Workspace(tmp_path / "ws")
    ws.write("src/a.py", "def f():\n    return 1\n")
    ws.write("src/b.py", "x = f()\ny = f()\n")
    ws.write("node_modules/skip.py", "f()\n")
    assert ws.glob("**/*.py")["files"] == ["src/a.py", "src/b.py"]
    g = ws.grep(r"f\(\)")
    assert g["matches"] == ["src/a.py:1: def f():", "src/b.py:1: x = f()", "src/b.py:2: y = f()"]
    assert ws.edit("src/b.py", "f()", "g()")["ok"] is False  # ambigu
    r = ws.edit("src/b.py", "f()", "g()", replace_all=True)
    assert r["ok"] and r["replacements"] == 2 and "-x = f()" in r["__ui__"]["diff"]
    assert ws.edit("src/a.py", "nope", "x")["ok"] is False
    assert ws.run_python("print(2 + 2)")["stdout"].strip() == "4"


def test_workspace_preview_is_in_the_user_message_not_the_system_prompt(fake_url, tmp_path):
    from tests.conftest import MockS2
    ws = Workspace(tmp_path / "ws")
    ws.write("src/app.py", "print(1)\n")
    s2 = MockS2([{"content": "", "tool_calls": [MockS2.tool_call("done", {"summary": "ok"})]}])
    s1 = SystemOneEngine(LlamaCppBackend(fake_url), model_name="fake")
    Prophet(s1, s2, ws).handle("Ajoute un test a mon projet")
    system, user = s2.calls[0][0][0]["content"], s2.calls[0][0][-1]["content"]
    assert "(Workspace: src/, src/app.py)" in user and "src/app.py" not in system

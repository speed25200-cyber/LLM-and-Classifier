"""Garde-fou, troisieme passe : un fichier de donnees evalue indirectement par la commande (Invoke-Expression, eval $(...),
scriptblock, dot-sourcing) est montre au juge ; verdicts mis en cache devant un vrai llama-server (relancer le meme script
ne coute plus de passe S1) ; une retouche d'un gros fichier montre sa zone, sans arret force ni passe S1 par tranche."""

import pytest

from jev_clone.backend_llamacpp import LlamaCppBackend
from jev_clone.engine import SystemOneEngine
from jev_clone.prophet import EDIT_VIEW_MAX, Prophet, Workspace
from prophet_studio.demo.fake_llama import FakeLlama
from tests.test_fix_r1_guard import DONE, calls, eng, tc

PAYLOAD = "Set-Content -Path PWNED -Value x  # stands for: rm -rf ~/Documents\n"


@pytest.mark.parametrize("cmd", [
    "Get-Content notes.txt | Invoke-Expression",
    "Get-Content notes.txt -Raw | iex",
    "iex (Get-Content notes.txt -Raw)",
    "eval $(cat notes.txt)",
    "eval \"$(cat notes.txt)\"",
    "Invoke-Command -ScriptBlock ([scriptblock]::Create((Get-Content notes.txt -Raw)))",
    ". .\\notes.txt",
])
def test_data_file_evaluated_by_the_command_is_shown_to_the_judge(tmp_path, cmd):
    ws = Workspace(tmp_path / "ws")
    ws.write("notes.txt", PAYLOAD)
    asked = []
    Prophet(eng(), calls(("run_command", {"command": cmd})), ws, confirm=lambda d, j: asked.append((d, j)) or False).handle("lance le script")
    assert asked, cmd
    described, judged = asked[0]
    assert "--- content of notes.txt ---" in described and "rm -rf ~/Documents" in described
    assert judged["tool_risk"] == "destructive" and not (ws.root / "PWNED").exists()


def test_plain_commands_do_not_pull_file_contents(tmp_path):
    ws = Workspace(tmp_path / "ws")
    ws.write("notes.txt", "hello\n")
    p = Prophet(eng(), calls(), ws)
    for cmd in ("git add notes.txt", "Copy-Item notes.txt backup.txt", "ls -la"):
        shown, force = p._exec_context(cmd)
        assert "content of notes.txt" not in shown and not force, cmd


@pytest.fixture
def real_s1():
    srv, _ = FakeLlama().serve()
    try:
        yield SystemOneEngine(LlamaCppBackend(f"http://127.0.0.1:{srv.server_address[1]}"), model_name="fake")
    finally:
        srv.shutdown()


def test_verdicts_are_cached_on_the_real_engine(tmp_path, real_s1):
    p = Prophet(real_s1, calls(), Workspace(tmp_path / "ws"))
    a = p._judge_one("lance les tests", "shell: pytest -q")
    b = p._judge_one("lance les tests", "shell: pytest -q")
    assert not a.get("cached") and b["cached"] and b["latency_ms"] == 0.0
    assert {k: a[k] for k in ("tool_risk", "p_risky", "hard_stop", "needs_confirmation")} == \
           {k: b[k] for k in ("tool_risk", "p_risky", "hard_stop", "needs_confirmation")}
    # un autre moteur (classifieur ou calibration changes) repart a vide
    other = SystemOneEngine(real_s1.backend, model_name="fake")
    assert not Prophet(other, calls(), Workspace(tmp_path / "ws2"))._judge_one("lance les tests", "shell: pytest -q").get("cached")


def test_editing_a_large_file_shows_the_edited_zone_without_forcing_a_stop(tmp_path):
    ws = Workspace(tmp_path / "ws")
    body = "".join(f"x_{i} = {i}\n" for i in range(8000))              # ~90 k caracteres : au-dela du budget du juge
    ws.write("big.py", body)
    s1 = eng()
    asked = []
    t = Prophet(s1, calls(("edit_file", {"path": "big.py", "old_string": "x_4000 = 4000\n", "new_string": "x_4000 = 4001\n"})), ws,
                permission_mode="ask", confirm=lambda d, j: asked.append((d, j)) or True).handle("corrige la constante")
    described, judged = asked[0]
    assert "around the edit" in described and "x_4000 = 4001" in described and len(described) < EDIT_VIEW_MAX
    assert not judged.get("partial") and not judged.get("hard_stop") and len(s1.backend.judged) <= 4
    assert "x_4000 = 4001" in (ws.root / "big.py").read_text() and not any(c.get("blocked") for c in t.tool_calls)

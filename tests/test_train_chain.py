"""Chaine d'entrainement vers Studio : donnees au format exact de Prophet, etiquetage enseignant, fusion / GGUF, notebook.
Sans GPU ni torch ; les tests "live" ne tournent qu'avec leurs variables d'environnement (voir en bas)."""

import importlib.util
import json
import os
import random
import re
import stat
from pathlib import Path

import pytest

from jev_clone.distill import _valid, label_example, label_rows
from jev_clone.distill import main as distill_main
from jev_clone.engine import SystemOneEngine
from jev_clone.guard import JUDGE_QUESTIONS
from jev_clone.prompt import PromptFormat
from jev_clone.prophet import PROPHET_TURN, Prophet, Workspace
from jev_clone.readout import question_fingerprint
from jev_clone.schema import SystemOneRequest
from tests.conftest import MockBackend, MockS2
from training import make_synthetic_prophet as MSP
from training import merge_lora as ML
from training import prophet_data as P

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "colab" / "jev_bonsai_a100.ipynb"
BRANCH = "claude/local-llm-high-performance-q4wk6f"


def trainer():
    """training/train_lora_rlcd.py importe sans torch (chargeur de donnees seulement)."""
    spec = importlib.util.spec_from_file_location("train_lora_rlcd_t", ROOT / "training" / "train_lora_rlcd.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def fps(questions):
    return {q: question_fingerprint(v) for q, v in questions.items()}


# ---- donnees ------------------------------------------------------------------------------------------------------------
def test_generated_rows_load_in_trainer(tmp_path):
    rows = P.generate(600, seed=1)
    assert {r["family"] for r in rows} == set(P.FAMILIES)
    for r in rows:
        req = SystemOneRequest(state=r["state"], questions=r["questions"])
        assert r["labels"] and all(_valid(req.questions[q], v) for q, v in r["labels"].items())
        assert set(r["weak"]) <= set(r["labels"])
    t = trainer()
    f = write(tmp_path / "train.jsonl", rows)
    br = t.load_examples(f, PromptFormat(chat=True, no_think=True), permutations=2)
    assert len(br) >= sum(len(r["labels"]) for r in rows)
    assert all(0 <= b["gold"] < len(b["labels"]) and b["text"] for b in br)


def test_shapes_match_prophet_and_voice_calls(tmp_path):
    """Les exemples generes portent les questions (empreinte) et la forme d'etat des vrais appels de Prophet et de la voix."""
    from tests.test_prophet import s1

    class Rec:
        cal, model_name = None, "mock"

        def __init__(self, eng):
            self.eng, self.reqs = eng, []

        def answer(self, req):
            self.reqs.append(req if isinstance(req, dict) else req.model_dump(mode="json"))
            return self.eng.answer(req)

    rec = Rec(s1())
    s2 = MockS2([{"content": "", "tool_calls": [MockS2.tool_call("write_file", {"path": "app/main.py", "content": "print('hi')\n"}),
                                                MockS2.tool_call("run_command", {"command": "python3 app/main.py"}, "c2")]},
                 {"content": "", "tool_calls": [MockS2.tool_call("done", {"summary": "ok"}, "c3")]}])
    Prophet(rec, s2, Workspace(tmp_path / "ws"), max_tools=5).handle("Crée un script python qui dit hi")
    seen = {}
    for r in rec.reqs:
        qs = r["questions"]
        kind = "turn" if "direct" in qs else "guard" if "tool_risk" in qs else "verify" if list(qs) == ["ok"] else "tools"
        seen.setdefault(kind, r)
    assert set(seen) == {"turn", "guard", "tools", "verify"}
    rng = random.Random(0)
    for fam in ("turn", "guard", "verify"):
        g = P.GENERATORS[fam](rng)
        assert set(g["state"]) == set(seen[fam]["state"]), fam
        assert fps(g["questions"]) == fps(seen[fam]["questions"]), fam
    cat = P.tool_catalog()
    real = seen["tools"]["questions"]
    assert set(seen["tools"]["state"]) == set(P.gen_tools(rng)["state"])
    assert fps({f"t_{n}": P.tool_question(n, cat[n]) for n in (q[2:] for q in real)}) == fps(real)
    shells = {question_fingerprint(P.tool_question("run_command", P.tool_catalog(s)["run_command"])) for s in P.SHELLS}
    assert question_fingerprint(real["t_run_command"]) in shells

    from prophet_studio.voice import route_utterance
    vrec = Rec(SystemOneEngine(MockBackend({})))
    route_utterance("tu peux arrêter de parler", vrec)
    v = P.gen_voice(rng)
    assert set(v["state"]) == set(vrec.reqs[0]["state"]) and v["state"]["context"] == vrec.reqs[0]["state"]["context"]
    assert fps(v["questions"]) == fps(vrec.reqs[0]["questions"])


def test_voice_rows_never_match_exact_grammar():
    from prophet_studio.voice import grammar_match
    rows = P.generate(800, seed=3, families=("voice",))
    assert rows and not any(grammar_match(r["state"]["utterance"]) for r in rows)
    assert {r["labels"]["intent"] for r in rows} >= {"prompt", "stop", "approve", "deny"}


def test_cli_split_calib_and_guard_policy(tmp_path, capsys):
    out, val, cal = tmp_path / "t.jsonl", tmp_path / "v.jsonl", tmp_path / "c.jsonl"
    res = MSP.main(["--out", str(out), "--val", str(val), "--calib", str(cal), "--n", "1500"])
    # exemples risques integres (GUARD_RISKY) : la famille guard reste, avec les classes dangereuses bien representees
    g = res["counts"]["guard"]
    assert {"destructive", "privileged", "exfiltration"} <= set(g) and "famille guard retiree" not in capsys.readouterr().err
    assert sum(v for k, v in g.items() if k not in ("readonly", "workspace_write")) >= 0.3 * sum(g.values())
    tr, va, ca = ([json.loads(l) for l in open(p, encoding="utf-8")] for p in (out, val, cal))
    assert tr and va and ca and not {r["group"] for r in tr} & {r["group"] for r in va}
    assert all(r["family"] == "turn" and set(r["state"]) == {"request", "workspace_files", "recent_turns"} for r in ca)
    assert all("needs_reasoning" not in r["labels"] for r in ca)   # estimations hors calibration
    # garde benin force, puis exemples de l'utilisateur (etiquettes verifiees)
    res = MSP.main(["--out", str(out), "--n", "300", "--families", "guard", "--guard-benign-only"])
    assert set(res["counts"]["guard"]) <= {"readonly", "workspace_write"}
    extra = write(tmp_path / "g.jsonl", [{"state": {"user_request": "exemple", "proposed_action": "shell: <action de votre journal>"},
                                          "labels": {"tool_risk": "destructive", "risk": 3, "policy_violation": True}}])
    res = MSP.main(["--out", str(out), "--n", "50", "--families", "guard", "--guard-extra", str(extra)])
    assert res["counts"]["guard"]["destructive"] >= 1 and any(
        r["state"]["proposed_action"] == "shell: <action de votre journal>" for r in map(json.loads, open(out, encoding="utf-8")))
    bad = write(tmp_path / "b.jsonl", [{"state": {"user_request": "x", "proposed_action": "y"}, "labels": {"tool_risk": "nope"}}])
    with pytest.raises(SystemExit, match="hors des options"):
        MSP.main(["--out", str(out), "--guard-extra", str(bad)])
    with pytest.raises(SystemExit):
        MSP.main(["--out", str(out), "--families", "turn,bogus"])


def test_calib_file_calibrates_pre_turn_only_and_imports_in_studio(tmp_path):
    from jev_clone.calibrate import calibrate
    from prophet_studio import calibration as scal
    val = [r for r in P.generate(400, seed=5, families=("turn",))]
    calib = [{**r, "labels": {k: v for k, v in r["labels"].items() if k not in r["weak"]}} for r in val]
    eng = SystemOneEngine(MockBackend({"direct": [0.7, 0.3], "risk": [0.6, 0.2, 0.1, 0.1]}))
    cal, _ = calibrate(eng, calib)
    st = {"request": "x", "workspace_files": [], "recent_turns": []}
    assert cal.entry("risk", PROPHET_TURN["risk"], st) is not None
    assert cal.t_for("risk", JUDGE_QUESTIONS["risk"], {"user_request": "x", "proposed_action": "y"}) == 1.0   # garde lu brut
    cal.save(tmp_path / "calibration.json")   # fichier du notebook -> Modeles > Importer une calibration
    g = tmp_path / "jev-clone-Q8_0.gguf"
    g.write_bytes(b"GGUF" + b"\0" * 64)
    out = scal.import_calibration(tmp_path / "runs", "custom-s1-jev-clone-q8-0", path=str(tmp_path / "calibration.json"), weights=g)
    applied, why, _ = scal.load_for_engine(out["path"])
    assert why == "" and applied.entry("direct", PROPHET_TURN["direct"], st) is not None


def test_expanded_rows_use_prophet_state_shape():
    class FakeChat:
        def chat(self, messages, **kw):
            return {"choices": [{"message": {"content": json.dumps({"requests": ["Fais une appli météo", "Build a weather app"]})}}]}
    seeds = [{"state": {"request": "Crée une appli météo"}, "labels": {"intent": "create_app", "language": "python", "risk": 0}}]
    rows = MSP.expanded_rows(FakeChat(), seeds, 2)
    assert len(rows) == 2 and all(set(r["state"]) == {"request", "workspace_files", "recent_turns"} for r in rows)
    assert rows[0]["labels"] == {"intent": "create_app", "language": "python", "risk": 0}


# ---- enseignant (jev_clone.distill) -------------------------------------------------------------------------------------
def rule_row():
    return {"state": {"request": "Crée une todo list", "workspace_files": [], "recent_turns": []}, "questions": PROPHET_TURN,
            "labels": {"direct": False, "clarify": False, "intent": "create_app", "language": "html_css", "needs_reasoning": False, "risk": 0},
            "source": "rules", "family": "turn", "group": "turn:x", "weak": ["needs_reasoning"]}


def test_teacher_keeps_rules_and_replaces_estimates():
    be = MockBackend({"direct": [0.9, 0.1], "needs_reasoning": [0.8, 0.2]})
    ex = label_example(SystemOneEngine(be), be, rule_row())
    assert ex["labels"]["direct"] is False and "direct" in ex["teacher_disagrees"]     # la regle fait foi
    assert ex["labels"]["needs_reasoning"] is True                                     # estimation remplacee
    assert ex["teacher_mode"] == "soft" and abs(sum(ex["teacher_probs"]["intent"].values()) - 1) < 1e-6
    be.chat_reply = json.dumps({"needs_reasoning": True, "direct": True, "intent": "nope", "risk": 7})
    ex = label_example(SystemOneEngine(be), be, rule_row(), mode="think")
    assert ex["labels"]["needs_reasoning"] is True and ex["labels"]["direct"] is False and ex["labels"]["intent"] == "create_app"
    assert ex["labels"]["risk"] == 0 and ex["teacher_mode"] == "think"   # hors domaine (7) refuse
    own = {"state": {"x": 1}, "questions": {"direct": PROPHET_TURN["direct"]}, "labels": {"direct": False}}
    be.chat_reply = json.dumps({"direct": True})
    assert label_example(SystemOneEngine(be), be, own, mode="think")["labels"]["direct"] is True   # comportement historique
    be.chat_reply = "pas du json"
    ex = label_example(SystemOneEngine(be), be, rule_row(), mode="think")
    assert "teacher_error" in ex and ex["labels"]["needs_reasoning"] is False   # enseignant muet : l'estimation reste

    class Down:
        def answer(self, req):
            raise RuntimeError("HTTP 500: context size exceeded")
    ex = label_example(Down(), be, rule_row(), mode="both")
    assert "500" in ex["teacher_error"] and "teacher_probs" not in ex and ex["labels"] == rule_row()["labels"]


def test_distill_cli_workers_order_and_resume(tmp_path, capsys):
    rows = [dict(rule_row(), state={"request": f"demande {i}", "workspace_files": [], "recent_turns": []}) for i in range(9)]
    inp, out = write(tmp_path / "in.jsonl", rows), tmp_path / "out.jsonl"
    assert distill_main(["--in", str(inp), "--out", str(out), "--workers", "3"], backend=MockBackend({})) == 9
    lines = out.read_text(encoding="utf-8").splitlines()
    assert [json.loads(l)["state"]["request"] for l in lines] == [f"demande {i}" for i in range(9)]
    out.write_text("\n".join(lines[:4]) + "\n" + lines[4][:20], encoding="utf-8")   # arret brutal au milieu d'une ligne
    assert distill_main(["--in", str(inp), "--out", str(out), "--resume"], backend=MockBackend({})) == 5
    assert [json.loads(l)["state"]["request"] for l in out.read_text(encoding="utf-8").splitlines()] == [f"demande {i}" for i in range(9)]
    assert "deja faits" in capsys.readouterr().err
    with pytest.raises(SystemExit, match="injoignable"):
        distill_main(["--teacher", "http://127.0.0.1:9", "--in", str(inp), "--out", str(tmp_path / "x.jsonl")])


def test_generator_with_teacher_labels_train_only(tmp_path):
    out, val = tmp_path / "t.jsonl", tmp_path / "v.jsonl"
    MSP.main(["--out", str(out), "--val", str(val), "--n", "120", "--families", "turn,voice", "--teacher", "http://x", "--workers", "2"],
             teacher_backend=MockBackend({}))
    tr = [json.loads(l) for l in open(out, encoding="utf-8")]
    va = [json.loads(l) for l in open(val, encoding="utf-8")]
    assert tr and all("teacher_probs" in r and "teacher_disagrees" in r for r in tr)
    assert va and not any("teacher_probs" in r for r in va)
    assert list(label_rows([], MockBackend({}))) == []


# ---- fusion / GGUF (training/merge_lora.py) ------------------------------------------------------------------------------
@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Aucun llama.cpp, llama-quantize ni runtime de Studio de la machine de test ne doit etre trouve."""
    monkeypatch.setattr(ML, "ROOT", tmp_path / "a" / "repo")   # ni <ROOT>/llama.cpp ni <ROOT>/../llama.cpp
    monkeypatch.setenv("PROPHET_HOME", str(tmp_path / "studio"))
    for v in ("LLAMA_CPP_DIR", "LLAMA_QUANTIZE"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(ML.shutil, "which", lambda *a, **k: None)
    return tmp_path


def modules(monkeypatch, present: bool):
    real = importlib.util.find_spec
    monkeypatch.setattr(ML.importlib.util, "find_spec",
                        lambda m, *a: (object() if present else None) if m in ("torch", "transformers", "peft", "numpy") else real(m, *a))


def adapter(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "base/model", "peft_type": "LORA"}))
    return d


def test_merge_lora_arguments_and_missing_tools(isolated, monkeypatch):
    t = isolated
    with pytest.raises(SystemExit, match="--adapter"):
        ML.main([])
    with pytest.raises(SystemExit, match="pas les deux"):
        ML.main(["--adapter", "a", "--merged", "b"])
    with pytest.raises(SystemExit, match="quantification inconnue"):
        ML.main(["--adapter", str(adapter(t / "run")), "--quant", "Q9_X"])
    (t / "noadapter").mkdir()
    with pytest.raises(ML.ToolMissing, match="adapter_config.json absent.*--merged"):
        ML.main(["--adapter", str(t / "noadapter")])
    modules(monkeypatch, False)
    with pytest.raises(ML.ToolMissing, match=r"torch.*pip install -e"):
        ML.main(["--adapter", str(adapter(t / "run"))])
    modules(monkeypatch, True)
    with pytest.raises(ML.ToolMissing, match=r"convert_hf_to_gguf\.py introuvable(.|\n)*git clone --depth 1 -b prism-b10683-d8f26ee"):
        ML.main(["--adapter", str(adapter(t / "run"))])
    conv = t / "llama.cpp"
    conv.mkdir()
    (conv / "convert_hf_to_gguf.py").write_text("")
    with pytest.raises(ML.ToolMissing, match="llama-quantize introuvable"):
        ML.main(["--adapter", str(adapter(t / "run")), "--llama-cpp", str(conv)])
    p = ML.main(["--adapter", str(adapter(t / "run")), "--llama-cpp", str(conv), "--quant", "", "--check"])   # sans quantification
    assert p["base"] == "base/model" and p["merged"] == t / "run" / "merged" and p["gguf"]["f16"] == t / "run-f16.gguf"


@pytest.mark.skipif(os.name == "nt", reason="faux llama-quantize en script shell")
def test_merge_lora_converts_and_quantizes_a_merged_model(isolated, monkeypatch, capsys):
    t = isolated
    modules(monkeypatch, True)
    merged = t / "runs" / "jev-clone" / "merged"
    merged.mkdir(parents=True)
    (merged / "config.json").write_text("{}")
    conv = t / "llama.cpp"
    conv.mkdir()
    (conv / "convert_hf_to_gguf.py").write_text("import sys\na = sys.argv\nopen(a[a.index('--outfile') + 1], 'wb').write(b'GGUF-f16')\n")
    q = t / "llama-quantize"
    q.write_text("#!/bin/sh\ncp \"$1\" \"$2\" && echo \"$3\" >> \"$2\"\n")
    q.chmod(q.stat().st_mode | stat.S_IEXEC)
    ML.main(["--merged", str(merged), "--llama-cpp", str(conv), "--quantize-bin", str(q)])
    for s in ("f16", "Q8_0", "Q4_K_M"):
        assert (t / "runs" / f"jev-clone-{s}.gguf").is_file()
    man = json.loads((t / "runs" / "jev-clone.manifest.json").read_text(encoding="utf-8"))
    assert set(man["files"]) == {"f16", "Q8_0", "Q4_K_M"} and man["name"] == "jev-clone"
    assert "Importer un GGUF" in capsys.readouterr().out
    (conv / "convert_hf_to_gguf.py").write_text("raise SystemExit(3)\n")
    with pytest.raises(SystemExit, match="echec de la conversion"):
        ML.main(["--merged", str(merged), "--llama-cpp", str(conv), "--quantize-bin", str(q)])


# ---- notebook -----------------------------------------------------------------------------------------------------------
def _python_of(src: str) -> str:
    """Cellule IPython -> Python compilable (lignes ! et % remplacees, continuations \\ comprises)."""
    out, cont = [], False
    for line in src.splitlines():
        s = line.lstrip()
        if cont or s.startswith(("!", "%")):
            if not cont:
                out.append(line[: len(line) - len(s)] + "pass")
            cont = line.rstrip().endswith("\\")
            continue
        out.append(line)
    return "\n".join(out)


def _commands(src: str) -> list[str]:
    cmds, cur = [], None
    for line in src.splitlines():
        if cur is not None:
            cur += " " + line.strip()
        elif line.lstrip().startswith("!"):
            cur = line.lstrip()[1:]
        if cur is not None and not line.rstrip().endswith("\\"):
            cmds.append(cur); cur = None
    return cmds


def test_notebook_structure_branch_and_commands(capsys):
    nb = json.loads(NB.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4 and nb["nbformat_minor"] >= 4 and nb["metadata"]["kernelspec"]["name"] == "python3"
    ids = [c["id"] for c in nb["cells"]]
    assert len(ids) == len(set(ids))
    for c in nb["cells"]:
        assert c["cell_type"] in ("markdown", "code") and isinstance(c["metadata"], dict) and all(isinstance(s, str) for s in c["source"])
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None
            compile(_python_of("".join(c["source"])), c["id"], "exec")
    text = "".join("".join(c["source"]) for c in nb["cells"])
    assert f"BRANCH = '{BRANCH}'" in text and "-b {BRANCH}" in text and "ish8th" not in text and "jev serve" not in text
    for step in ("make_synthetic_prophet.py", "train_lora_rlcd.py", "merge_lora.py", "jev_clone.calibrate", "eval_clone.py",
                 "--qlora", "Importer un GGUF", "Calibrer", "files.download", "prism-b10683-d8f26ee"):
        assert step in text, step
    # chaque option passee a nos scripts existe dans leur aide
    helps = {}
    mods = {"training/make_synthetic_prophet.py": MSP.main, "training/merge_lora.py": ML.main, "training/train_lora_rlcd.py": trainer().main,
            "training/make_from_trajectories.py": __import__("training.make_from_trajectories", fromlist=["main"]).main,
            "training/eval_clone.py": __import__("training.eval_clone", fromlist=["main"]).main,
            "-m jev_clone.calibrate": __import__("jev_clone.calibrate", fromlist=["main"]).main}
    def helptext(key):
        if key not in helps:
            with pytest.raises(SystemExit):
                mods[key](["--help"])
            helps[key] = capsys.readouterr().out
        return helps[key]

    flag_rx = r"(?<![\w-])--[a-z][a-z0-9-]*"
    checked = 0
    for c in nb["cells"]:
        src = "".join(c["source"])
        users: dict[str, list[str]] = {}   # variable interpolee {x} -> scripts qui la recoivent
        for cmd in _commands(src):
            for key in mods:
                if key in cmd:
                    tail = cmd.split(key, 1)[1]
                    for flag in re.findall(flag_rx, tail):
                        assert flag in helptext(key), f"{key} {flag}"
                        checked += 1
                    for var in re.findall(r"\{(\w+)\}", tail):
                        users.setdefault(var, []).append(key)
            m = re.search(r"training/\w+\.py", cmd)
            assert m is None or (ROOT / m[0]).is_file(), cmd
        # options construites en Python (args = '--out ...' puis {args}) : dans l'aide du script qui les recoit
        for line in src.splitlines():
            m = re.match(r"\s*(\w+)\s*\+?=", line)
            if m and m[1] in users:
                for flag in re.findall(flag_rx, line):
                    assert any(flag in helptext(k) for k in users[m[1]]), f"{c['id']} {m[1]} {flag}"
                    checked += 1
    assert checked >= 30


# ---- live : sur la machine de l'utilisateur ------------------------------------------------------------------------------
TEACHER = os.environ.get("JEV_TEACHER_URL")          # llama-server de Bonsai 2 27B (Studio : S2 sur 7880)
S1 = os.environ.get("JEV_S1_EVAL_URL")               # llama-server du classifieur (Studio : S1 sur 7881)
ADAPTER, LLAMA_CPP = os.environ.get("MERGE_LORA_ADAPTER"), os.environ.get("LLAMA_CPP_DIR")


@pytest.mark.skipif(not TEACHER, reason="JEV_TEACHER_URL non defini")
def test_live_teacher_labels_generated_rows():
    from jev_clone.backend_llamacpp import LlamaCppBackend
    rows = P.generate(6, seed=7, families=("turn", "tools", "voice"))
    out = list(label_rows(rows, LlamaCppBackend(TEACHER, max_workers=2, timeout=600), workers=2))
    for r in out:
        for qid, tp in r["teacher_probs"].items():
            assert abs(sum(tp.values()) - 1) < 1e-3, qid


@pytest.mark.skipif(not S1, reason="JEV_S1_EVAL_URL non defini")
def test_live_eval_clone(tmp_path):
    from training.eval_clone import main as eval_main
    f = write(tmp_path / "v.jsonl", P.generate(20, seed=8))
    res = eval_main(["--server", S1, "--data", str(f)])
    assert any(k.startswith("turn/") for k in res)


@pytest.mark.skipif(not (ADAPTER and LLAMA_CPP), reason="MERGE_LORA_ADAPTER et LLAMA_CPP_DIR non definis")
def test_live_merge_lora(tmp_path):
    p = ML.main(["--adapter", ADAPTER, "--llama-cpp", LLAMA_CPP, "--out", str(tmp_path / "merged"), "--gguf-dir", str(tmp_path),
                 "--name", "live", "--quant", "Q8_0"])
    assert p["gguf"]["Q8_0"].stat().st_size > 1_000_000

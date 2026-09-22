"""Prophet Studio : planificateur VRAM (RTX 5060), detection materielle, choix du runtime, telechargements, voix."""

import hashlib
import io
import tarfile
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from jev_clone.engine import SystemOneEngine
from prophet_studio.catalog import MODELS
from prophet_studio.downloads import DownloadJob, Downloader, safe_extract
from prophet_studio.hardware import GPU, HardwareInfo, enrich, fake_gpus, parse_cuda_version, parse_nvidia_smi_csv
from prophet_studio.installer import guessed_assets, select_runtime_assets
from prophet_studio.planner import degrade, make_plan
from prophet_studio.voice import COMMANDS, clean_for_speech, grammar_match, route_utterance, split_sentences, strip_wake, to_wav
from tests.conftest import ScriptedBackend


def hw(name="NVIDIA GeForce RTX 5060", total=8151, used=650, display=True, ram=32):
    g = enrich(GPU(0, name, "nvidia", total, used, total - used, "580.88", "13.0", display_active=display))
    return HardwareInfo("Windows", "11", "amd64", "AMD Ryzen 7 7700", 8, 16, ram, ram * 0.7, True, True, [g])


# ---- materiel -------------------------------------------------------------------------------------------------------
def test_nvidia_smi_parsing_recognizes_rtx5060_blackwell():
    csv = "0, NVIDIA GeForce RTX 5060, 8151, 612, 7539, 580.88, 12.0, Enabled\n"
    g = parse_nvidia_smi_csv(csv, parse_cuda_version("| NVIDIA-SMI 580.88   Driver Version: 580.88   CUDA Version: 13.0 |"))[0]
    assert (g.vram_total_mib, g.compute_cap, g.cuda_version, g.display_active) == (8151, "12.0", "13.0", True)
    assert g.arch == "blackwell" and g.bandwidth_gbs == 448 and g.is_blackwell
    # vieux pilote sans compute_cap : l'architecture vient du nom
    g2 = parse_nvidia_smi_csv("0, NVIDIA GeForce RTX 4060, 8188, 300, 7888, 551.2\n")[0]
    assert g2.compute_cap == "8.9" and not g2.is_blackwell
    assert fake_gpus("NVIDIA GeForce RTX 5060 Ti:16311:700")[0].bandwidth_gbs == 448


# ---- planificateur ------------------------------------------------------------------------------------------------------
def test_rtx5060_plan_keeps_bonsai2_on_gpu_and_fits():
    p = make_plan(hw())
    assert p.s2.model_id == "bonsai2-27b-ptq1" and p.s2.device == "gpu" and p.s2.ngl == 99 and p.s2.kv_type == "q4_0"
    assert p.s2.ctx >= 16384 and p.s2.mmproj == "cpu" and p.fits
    used = sum(v for k, v in p.budget.items() if k not in ("total", "other", "free", "reserve"))
    assert used + p.budget["other"] + p.budget["reserve"] <= p.budget["total"]
    lo, hi = p.expected["s2"]["tok_s"]
    assert 40 <= lo < hi <= 65   # 448 Go/s / 5,93 Go x 0,62-0,78
    assert any("Blackwell" in n for n in p.notes) and any("ecran" in n.lower() for n in p.notes)


def test_rtx5060_without_display_gets_more_context_and_speed_priority_switches_model():
    assert make_plan(hw(used=150, display=False)).s2.ctx > make_plan(hw()).s2.ctx
    fast = make_plan(hw(), "vitesse")
    assert fast.s2.model_id == "bonsai-27b-q1" and fast.s1.device == "gpu"
    assert make_plan(hw(), "contexte").s2.ctx >= make_plan(hw(), "equilibre").s2.ctx


def test_bigger_cards_and_cpu_only():
    ti = make_plan(hw("NVIDIA GeForce RTX 5060 Ti", 16311, 700))
    assert ti.s2.model_id == "bonsai2-27b-pq2" and ti.s2.mmproj == "gpu" and ti.s1.device == "gpu" and ti.s1.model_id == "ternary-4b"
    cpu = make_plan(HardwareInfo("Linux", "6", "x86_64", "i7", 8, 16, 16, 12, True, False, []))
    assert cpu.backend == "cpu" and cpu.s2.model_id == "bonsai-8b-q1" and cpu.s1.device == "cpu" and cpu.fits


def test_degrade_ladder_ends_and_moves_s1_first_when_on_gpu():
    p = make_plan(hw("NVIDIA GeForce RTX 5060 Ti", 16311, 700))
    p1 = degrade(p)
    assert p1.s2.mmproj == "cpu" and p1.rung == 1
    p2 = degrade(p1)
    assert p2.s1.device == "cpu"
    rungs = 0
    while p2 is not None and rungs < 40:
        p2, rungs = degrade(p2), rungs + 1
    assert p2 is None and rungs < 40


def test_explicit_model_override_that_does_not_fit_is_partially_offloaded():
    p = make_plan(hw(total=6144, used=400), s2_override="bonsai2-27b-pq2")
    assert p.s2.device == "partial" and 0 < p.s2.ngl < 65 and not p.fits


# ---- runtime ----------------------------------------------------------------------------------------------------------
TAG = "prism-b10683-d8f26ee"
ASSETS = [{"name": n, "browser_download_url": f"https://x/{n}", "size": 1} for n in [
    f"llama-{TAG}-bin-win-cuda-12.4-x64.zip", f"llama-{TAG}-bin-win-cuda-12.8-x64.zip", f"llama-{TAG}-bin-win-cuda-13.3-x64.zip",
    "cudart-llama-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-12.8-x64.zip", "cudart-llama-bin-win-cuda-13.3-x64.zip",
    f"llama-{TAG}-bin-win-cpu-x64.zip", f"llama-{TAG}-bin-win-vulkan-x64.zip",
    f"llama-{TAG}-bin-linux-cuda-12.4-x64.tar.gz", f"llama-{TAG}-bin-linux-cuda-12.8-x64.tar.gz", f"llama-{TAG}-bin-ubuntu-x64.tar.gz",
    f"llama-{TAG}-bin-ubuntu-vulkan-x64.tar.gz", f"llama-{TAG}-bin-macos-arm64.tar.gz"]]


def gpu(name, cc, cuda):
    return enrich(GPU(0, name, "nvidia", 8000, 0, 8000, "x", cuda, cc))


def test_runtime_for_rtx5060_on_windows_is_cuda_12_8_plus_with_cudart():
    s = select_runtime_assets(ASSETS, "Windows", "AMD64", gpu("RTX 5060", "12.0", "13.0"))
    assert s["backend"] == "cuda" and s["cuda"] == 12.8 and "cuda-12.8" in s["main"]["name"]
    assert s["extra"][0]["name"] == "cudart-llama-bin-win-cuda-12.8-x64.zip"
    newer = select_runtime_assets(ASSETS, "Windows", "AMD64", gpu("RTX 5060", "12.0", "13.3"))
    assert newer["cuda"] == 13.3
    # pilote Blackwell trop ancien pour 12.8 : on garde quand meme une build >= 12.8 (le planificateur previent)
    old = select_runtime_assets(ASSETS, "Windows", "AMD64", gpu("RTX 5060", "12.0", "12.4"))
    assert old["cuda"] >= 12.8


def test_runtime_for_other_platforms():
    assert select_runtime_assets(ASSETS, "Windows", "AMD64", gpu("GTX 1080 Ti", "6.1", "13.3"))["cuda"] == 12.8  # pas de CUDA 13 avant Turing
    lin = select_runtime_assets(ASSETS, "Linux", "x86_64", gpu("RTX 4060", "8.9", "12.8"))
    assert lin["main"]["name"].endswith("linux-cuda-12.8-x64.tar.gz") and lin["extra"] == []
    assert select_runtime_assets(ASSETS, "Linux", "x86_64", None)["main"]["name"] == f"llama-{TAG}-bin-ubuntu-x64.tar.gz"
    mac = select_runtime_assets(ASSETS, "Darwin", "arm64", GPU(0, "Apple", "apple"))
    assert mac["backend"] == "metal"
    assert select_runtime_assets(guessed_assets(TAG, "Windows", "AMD64"), "Windows", "AMD64", gpu("RTX 5060", "12.0", "13.0"))["cuda"] == 12.8


# ---- telechargements ------------------------------------------------------------------------------------------------------
@pytest.fixture
def file_server():
    blobs: dict[str, bytes] = {}
    state = {"fail_after": None}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            data = blobs.get(self.path)
            if data is None:
                self.send_response(404); self.end_headers(); return
            start = 0
            rng = self.headers.get("Range")
            if rng:
                start = int(rng.split("=")[1].split("-")[0])
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
            else:
                self.send_response(200)
            body = data[start:]
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if state["fail_after"] is not None and not rng:
                self.wfile.write(body[: state["fail_after"]]); self.wfile.flush()
                state["fail_after"] = None
                self.connection.close()
                return
            self.wfile.write(body)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", blobs, state
    srv.shutdown()


def test_download_resumes_after_cut_and_verifies_sha(file_server, tmp_path):
    base, blobs, state = file_server
    data = np.random.default_rng(0).bytes(3_000_000)
    blobs["/model.gguf"] = data
    state["fail_after"] = 1_000_000
    events = []
    dl = Downloader(events.append, retries=3)
    j = dl.add(DownloadJob(f"{base}/model.gguf", tmp_path / "m" / "model.gguf", size=len(data), sha256=hashlib.sha256(data).hexdigest()))
    j = dl.wait(j.id, 30)
    assert j.status == "done", j.error
    assert (tmp_path / "m" / "model.gguf").read_bytes() == data
    assert any(e["job"]["status"] == "running" for e in events)
    bad = dl.add(DownloadJob(f"{base}/model.gguf", tmp_path / "bad.gguf", sha256="0" * 64))
    assert dl.wait(bad.id, 30).status == "error" and not (tmp_path / "bad.gguf").exists()


def test_archive_extraction_and_traversal_rejected(file_server, tmp_path):
    base, blobs, _ = file_server
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:bz2") as t:
        for name, content in [("pack/tokens.txt", b"a 0\n"), ("pack/model.onnx", b"onnx")]:
            ti = tarfile.TarInfo(name); ti.size = len(content); t.addfile(ti, io.BytesIO(content))
    blobs["/voice.tar.bz2"] = buf.getvalue()
    dl = Downloader(retries=1)
    j = dl.wait(dl.add(DownloadJob(f"{base}/voice.tar.bz2", tmp_path / "dl" / "voice.tar.bz2", extract_to=tmp_path / "voice")).id, 30)
    assert j.status == "done" and (tmp_path / "voice" / "pack" / "model.onnx").read_bytes() == b"onnx"
    evil = io.BytesIO()
    with tarfile.open(fileobj=evil, mode="w:gz") as t:
        ti = tarfile.TarInfo("../escape.txt"); ti.size = 1; t.addfile(ti, io.BytesIO(b"x"))
    p = tmp_path / "evil.tar.gz"; p.write_bytes(evil.getvalue())
    with pytest.raises(ValueError):
        safe_extract(p, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


# ---- voix -------------------------------------------------------------------------------------------------------------------
def test_wake_word_and_grammar_fr_en():
    assert strip_wake("OK Prophet, lis la réponse.") == ("lis la réponse.", True)
    assert strip_wake("Hé Prophète : nouvelle session")[1] is True
    assert strip_wake("lis la réponse")[1] is False
    assert grammar_match("Nouvelle session !") == "new_session"
    assert grammar_match("Arrête") == "stop" and grammar_match("stop please") == "stop"
    assert grammar_match("Accepte s'il te plaît") == "approve"
    assert grammar_match("crée une application météo") is None
    for cmd, spec in COMMANDS.items():
        for phrase in spec["fr"] + spec["en"]:
            assert grammar_match(phrase) == cmd, phrase


def test_route_uses_systemone_for_short_ambiguous_utterances():
    crit = list(COMMANDS) + ["prompt"]
    table = [0.02] * len(crit)
    table[crit.index("read_last")] = 0.9
    s1 = SystemOneEngine(ScriptedBackend([{"intent": table}], declared={"intent": crit}), model_name="mock")
    r = route_utterance("tu peux me relire ça", s1)
    assert (r.kind, r.command, r.source) == ("command", "read_last", "systemone") and r.confidence > 0.72
    low = [1.0 / len(crit)] * len(crit)
    s1b = SystemOneEngine(ScriptedBackend([{"intent": low}], declared={"intent": crit}), model_name="mock")
    assert route_utterance("tu peux me relire ça", s1b).kind == "prompt"   # incertain -> jamais de commande
    assert route_utterance("écris un script python qui trie des fichiers par date de modification", s1).kind == "prompt"
    assert route_utterance("nouvelle session", None).command == "new_session"
    assert route_utterance("quelle heure est-il", None, require_wake=True).kind == "ignored"
    assert route_utterance("Prophet quelle heure est-il", None, require_wake=True).kind == "prompt"


def test_speech_text_cleanup_and_wav():
    assert clean_for_speech("## Titre\n**Gras** et `code` :\n```py\nprint(1)\n```\n- [lien](http://x)") == "Titre Gras et code : (bloc de code) lien"
    assert split_sentences("Un. Deux ! Trois ?") == ["Un.", "Deux !", "Trois ?"]
    wav = to_wav(np.zeros(1600, dtype=np.float32), 16000)
    with wave.open(io.BytesIO(wav)) as w:
        assert (w.getframerate(), w.getnframes()) == (16000, 1600)


def test_catalog_has_recommended_models():
    assert MODELS["bonsai2-27b-ptq1"].size_gb == 5.93 and MODELS["bonsai2-27b-ptq1"].runtime == "prism"
    assert MODELS["ternary-1.7b"].role == "s1"


# ---- calibration VRAM ----------------------------------------------------------------------------------------------------
def test_measured_overhead_feeds_the_planner():
    from prophet_studio import planner
    before = make_plan(hw(used=150, display=False), "contexte")
    try:
        # mesure : le serveur a pris 6 250 Mio a 32 k (q4_0) -> surcout reel ~ 6250 - 5652 - 573 = ~25 -> borne a 200 Mio
        over = planner.calibrated_overhead("bonsai2-27b-ptq1", 64, 5.52, 32768, "q4_0", 0.0, 6250.0)
        assert over == 200.0
        planner.MEASURED["bonsai2-27b-ptq1"] = 600.0
        after = make_plan(hw(used=150, display=False), "contexte")
        assert after.s2.ctx > before.s2.ctx and any("calibre" in n for n in after.notes)
        assert after.budget["s2_runtime"] == 664
    finally:
        planner.MEASURED.clear()


def test_supervisor_reports_vram_delta(tmp_path):
    import sys
    from prophet_studio.supervisor import Runtime
    model = tmp_path / "m.gguf"; model.touch()
    seen, vram = [], iter([1000.0, 7300.0, 7300.0, 7300.0])
    rt = Runtime(tmp_path / "logs", lambda e: None, lambda mid: model, lambda mid: None,
                 lambda: [sys.executable, "-m", "prophet_studio.demo.fake_llama"], lambda: set())
    rt.vram_probe = lambda: next(vram)
    rt.on_measure = lambda role, sp, used: seen.append((role, sp.model_id, used))
    import socket
    ports = []
    for _ in range(2):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0)); ports.append(s.getsockname()[1])
    plan = make_plan(hw())
    try:
        assert rt.start(plan, tuple(ports))
        assert seen == [("s2", "bonsai2-27b-ptq1", 6300.0)]   # classifieur sur CPU : pas de mesure GPU
    finally:
        rt.stop()

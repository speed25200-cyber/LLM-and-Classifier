"""Chaine vocale du coeur avec un module sherpa_onnx factice (meme API que la vraie bibliotheque) :
construction des modeles a partir des fichiers installes, transcription, synthese, VAD mains libres, points d'API."""

import io
import json
import sys
import types
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from prophet_studio.config import Paths
from prophet_studio.server import Studio, build_app
from prophet_studio.voice import VoiceService


def fake_sherpa(calls: list):
    so = types.ModuleType("sherpa_onnx")

    class Stream:
        def __init__(self):
            self.result = types.SimpleNamespace(text="")
            self.n = 0

        def accept_waveform(self, sr, samples):
            assert sr == 16000 and samples.dtype == np.float32
            self.n += len(samples)

    class OfflineRecognizer:
        @classmethod
        def from_transducer(cls, **kw):
            calls.append(("transducer", kw)); return cls()

        @classmethod
        def from_whisper(cls, **kw):
            calls.append(("whisper", kw)); return cls()

        def create_stream(self):
            return Stream()

        def decode_stream(self, st):
            st.result.text = "Prophet, nouvelle session" if st.n > 16000 else "salut"

    class Cfg:
        def __init__(self, *a, **kw):
            calls.append((type(self).__name__, kw)); self.kw = kw

    for name in ("OfflineTtsVitsModelConfig", "OfflineTtsKokoroModelConfig", "OfflineTtsModelConfig", "OfflineTtsConfig",
                 "SileroVadModelConfig", "VadModelConfig"):
        setattr(so, name, type(name, (Cfg,), {}))

    class OfflineTts:
        def __init__(self, cfg):
            calls.append(("tts", cfg.kw))

        def generate(self, text, sid=0, speed=1.0):
            calls.append(("generate", {"text": text, "speed": speed}))
            return types.SimpleNamespace(samples=np.zeros(2205, dtype=np.float32), sample_rate=22050)

    class VoiceActivityDetector:
        def __init__(self, cfg, buffer_size_in_seconds=60):
            self.buf, self.segs = [], []

        def accept_waveform(self, s):
            self.buf.append(s)
            if sum(len(b) for b in self.buf) >= 16000 * 1.2:   # 1,2 s de "parole" -> un segment
                self.segs.append(types.SimpleNamespace(samples=np.concatenate(self.buf)))
                self.buf = []

        def is_speech_detected(self):
            return bool(self.buf)

        def empty(self):
            return not self.segs

        @property
        def front(self):
            return self.segs[0]

        def pop(self):
            self.segs.pop(0)

    so.OfflineRecognizer, so.OfflineTts, so.VoiceActivityDetector = OfflineRecognizer, OfflineTts, VoiceActivityDetector
    return so


@pytest.fixture
def voice_env(monkeypatch, tmp_path):
    calls: list = []
    monkeypatch.setitem(sys.modules, "sherpa_onnx", fake_sherpa(calls))
    files = {
        "stt-parakeet-v3": {"encoder": "enc.onnx", "decoder": "dec.onnx", "joiner": "join.onnx", "tokens": "tokens.txt"},
        "tts-fr-siwis": {"model": "fr.onnx", "tokens": "tokens.txt", "data_dir": "espeak-ng-data"},
        "vad-silero": {"model": "silero_vad.onnx"},
    }
    return calls, files


def test_voice_service_builds_models_from_installed_files(voice_env):
    calls, files = voice_env
    from prophet_studio.config import Settings
    vs = VoiceService(files.get, Settings, threads=2)
    assert vs.engine == "sherpa-onnx"
    out = vs.transcribe(np.zeros(32000, dtype=np.float32))
    assert out["text"] == "Prophet, nouvelle session" and out["audio_s"] == 2.0
    kind, kw = calls[0]
    assert kind == "transducer" and kw["model_type"] == "nemo_transducer" and kw["encoder"] == "enc.onnx" and kw["num_threads"] == 2
    wav = vs.synthesize("## Bonjour **monde**")
    with wave.open(io.BytesIO(wav)) as w:
        assert w.getframerate() == 22050 and w.getnframes() == 2205
    assert ("generate", {"text": "Bonjour monde", "speed": 1.0}) in calls
    vits = next(kw for k, kw in calls if k == "OfflineTtsVitsModelConfig")
    assert vits == {"model": "fr.onnx", "tokens": "tokens.txt", "data_dir": "espeak-ng-data"}
    vs.transcribe(np.zeros(32000, dtype=np.float32))
    assert sum(1 for k, _ in calls if k == "transducer") == 1   # modele charge une seule fois


def test_voice_endpoints_ptt_and_handsfree(voice_env, tmp_path, monkeypatch):
    calls, files = voice_env
    monkeypatch.setenv("PROPHET_FAKE_GPU", "NVIDIA GeForce RTX 5060:8151:650")
    paths = Paths(tmp_path / "home")
    paths.settings.write_text(json.dumps({"autostart_models": False, "voice": {"enabled": False}}))
    st = Studio(paths, "tok", 7878)
    st.installer.voice_files = files.get          # voix "installees"
    st.voice.files = files.get
    H = {"X-Prophet-Token": "tok"}
    with TestClient(build_app(st), base_url="http://127.0.0.1:7878") as c:
        pcm = (np.zeros(32000, dtype="<i2")).tobytes()
        r = c.post("/api/voice/transcribe", headers={**H, "Content-Type": "application/octet-stream"}, content=pcm).json()
        assert r["text"] == "Prophet, nouvelle session" and r["route"]["kind"] == "command" and r["route"]["command"] == "new_session"
        assert c.post("/api/voice/transcribe", headers=H, content=b"\0" * 100).status_code == 422
        tts = c.post("/api/voice/tts", headers=H, json={"text": "Bonjour"})
        assert tts.status_code == 200 and tts.headers["content-type"] == "audio/wav"
        assert c.get("/api/voice/status", headers=H).json()["stt_installed"] is True
        # mains libres : flux PCM -> VAD -> segment transcrit et route (mot d'eveil exige en mode handsfree)
        st.settings.update({"voice": {"mode": "handsfree"}})
        with c.websocket_connect("/api/voice/stream?token=tok") as ws:
            chunk = (np.ones(4000, dtype="<i2") * 1000).tobytes()
            got = []
            for _ in range(6):
                ws.send_bytes(chunk)
            for _ in range(10):
                m = ws.receive_json()
                got.append(m)
                if m["type"] == "segment":
                    break
            seg = next(m for m in got if m["type"] == "segment")
            assert seg["text"] == "Prophet, nouvelle session" and seg["route"]["command"] == "new_session"
            assert any(m["type"] == "vad" and m["speaking"] for m in got)
        with pytest.raises(Exception):
            with c.websocket_connect("/api/voice/stream?token=bad") as ws:
                ws.receive_json()

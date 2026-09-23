"""Voix : reconnaissance (Parakeet / Whisper), detection de parole (Silero), synthese (Piper / Kokoro), sur CPU via
sherpa-onnx (la VRAM reste au LLM), et routage des commandes vocales par le classifieur.

Routage d'un enonce (FR / EN) :
  1. mot d'eveil retire ("Prophet, ...", "OK Prophet", "dis Prophet") ; en mode mains libres, sans mot d'eveil -> ignore ;
  2. grammaire exacte ("nouvelle session", "stop", "accepte"...) -> commande immediate, sans modele ;
  3. enonce court et ambigu -> System One tranche en une passe (choice calibre : commande ou demande pour l'agent) ;
     sous le seuil de confiance, c'est une demande (on ne declenche jamais une commande par erreur) ;
     accepter / refuser une autorisation n'agit QUE sur la phrase exacte de la grammaire : si seul le classifieur y
     voit une reponse a l'autorisation, on rend `confirm_approve` / `confirm_deny` (l'interface demande la phrase exacte) ;
  4. sinon -> demande envoyee a l'agent.
"""

from __future__ import annotations

import io
import re
import threading
import time
import unicodedata
import wave
from dataclasses import dataclass
from typing import Callable

import numpy as np

COMMANDS: dict[str, dict] = {
    "new_session": {"desc": "start a new chat session", "fr": ["nouvelle session", "nouvelle conversation", "nouveau chat", "nouvelle discussion"],
                    "en": ["new session", "new chat", "new conversation"]},
    "stop": {"desc": "stop the current generation or speech", "fr": ["stop", "arrete", "arrete toi", "stoppe", "annule", "tais toi", "silence"],
             "en": ["stop", "cancel", "abort", "be quiet", "shut up"]},
    "approve": {"desc": "approve the pending action the assistant asked permission for", "fr": ["accepte", "autorise", "oui vas y", "vas y", "confirme", "j autorise", "ok vas y"],
                "en": ["approve", "allow", "yes do it", "go ahead", "confirm"]},
    "deny": {"desc": "refuse the pending action", "fr": ["refuse", "non", "n execute pas", "rejette", "interdit"], "en": ["deny", "reject", "no", "don t do it"]},
    "send": {"desc": "send the message currently typed in the input box", "fr": ["envoie", "envoyer", "envoie le message", "valide"],
             "en": ["send", "send it", "send the message", "submit"]},
    "read_last": {"desc": "read the last answer aloud", "fr": ["lis la reponse", "relis", "lis moi la reponse", "lis le"],
                  "en": ["read it", "read the answer", "read that", "read it aloud"]},
    "plan_on": {"desc": "switch to plan mode (the assistant only plans, changes nothing)", "fr": ["mode plan", "active le mode plan", "passe en mode plan"],
                "en": ["plan mode", "enable plan mode", "switch to plan mode"]},
    "plan_off": {"desc": "leave plan mode", "fr": ["quitte le mode plan", "desactive le mode plan", "mode normal"], "en": ["exit plan mode", "disable plan mode", "normal mode"]},
    "effort_deep": {"desc": "ask the assistant to think longer on the next request", "fr": ["reflechis bien", "reflechis longtemps", "mode reflexion profonde"],
                    "en": ["think hard", "think harder", "deep thinking"]},
    "effort_fast": {"desc": "ask for fast answers without long reasoning", "fr": ["reponds vite", "mode rapide"], "en": ["answer fast", "fast mode", "quick mode"]},
    "open_settings": {"desc": "open the settings screen", "fr": ["ouvre les parametres", "ouvre les reglages", "parametres", "reglages"],
                      "en": ["open settings", "settings"]},
    "open_models": {"desc": "open the models screen", "fr": ["ouvre les modeles", "gestion des modeles"], "en": ["open models", "models"]},
    "mute": {"desc": "turn the microphone off", "fr": ["coupe le micro", "micro off", "desactive le micro"], "en": ["mute", "microphone off", "stop listening"]},
    "clear_input": {"desc": "clear the text input box", "fr": ["efface", "efface tout", "vide le champ"], "en": ["clear", "clear input"]},
}
PERMISSION_COMMANDS = ("approve", "deny")   # agissent sur une autorisation : grammaire exacte seulement
WAKE_WORDS = ("prophet", "prophete", "profet", "profete", "prof et", "profit")
WAKE_PREFIXES = ("hey", "he", "eh", "ok", "okay", "dis", "salut", "bonjour", "hello", "hi", "alors")


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFD", text.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def strip_wake(text: str, wake: str = "prophet") -> tuple[str, bool]:
    """('lis la reponse', True) pour 'OK Prophet, lis la réponse.' ; le texte d'origine est conserve (hors mot d'eveil)."""
    words = text.strip().split()
    norm = [normalize(w) for w in words]
    wakes = set(WAKE_WORDS) | {normalize(wake)}
    for i in range(min(3, len(norm))):
        if norm[i] in wakes and all(n in WAKE_PREFIXES for n in norm[:i]):
            rest = " ".join(words[i + 1:]).lstrip(" ,.:;!?-")
            return rest, True
        if i + 1 < len(norm) and f"{norm[i]} {norm[i + 1]}" in wakes and all(n in WAKE_PREFIXES for n in norm[:i]):
            return " ".join(words[i + 2:]).lstrip(" ,.:;!?-"), True
    return text.strip(), False


def grammar_match(text: str) -> str | None:
    n = normalize(text)
    if not n:
        return None
    for cmd, spec in COMMANDS.items():
        for phrase in spec["fr"] + spec["en"]:
            if n == phrase or n in (f"{phrase} s il te plait", f"{phrase} stp", f"{phrase} please", f"please {phrase}"):
                return cmd
    return None


@dataclass
class Route:
    kind: str                    # "command" | "prompt" | "ignored"
    text: str
    command: str | None = None
    confidence: float = 1.0
    source: str = "grammar"      # grammar | systemone | length | wake
    ms: float = 0.0
    error: str | None = None     # classifieur en erreur (la phrase part alors a l'agent)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def route_utterance(text: str, s1_engine=None, require_wake: bool = False, wake: str = "prophet", threshold: float = 0.72,
                    max_command_words: int = 8) -> Route:
    t0 = time.perf_counter()
    rest, woke = strip_wake(text, wake)
    if require_wake and not woke:
        return Route("ignored", text, source="wake")
    if not normalize(rest):
        return Route("ignored", rest, source="wake")
    cmd = grammar_match(rest)
    if cmd:
        return Route("command", rest, cmd, 0.99, "grammar", round((time.perf_counter() - t0) * 1000, 2))
    if len(normalize(rest).split()) > max_command_words or s1_engine is None:
        return Route("prompt", rest, source="length" if len(normalize(rest).split()) > max_command_words else "grammar",
                     ms=round((time.perf_counter() - t0) * 1000, 2))
    criteria = {c: s["desc"] for c, s in COMMANDS.items()}
    criteria["prompt"] = "a question, request or task for the assistant itself (NOT an instruction to control the app)"
    try:
        r = s1_engine.answer({"state": {"utterance": rest, "context": "voice input of a desktop AI assistant app"},
                              "questions": {"intent": {"type": "choice", "criteria": criteria,
                                                       "instructions": "The user spoke this utterance. Is it one of the app control commands, or a prompt for the assistant?"}}})
        a = r.answers["intent"]
        ms = round((time.perf_counter() - t0) * 1000, 2)
        p = float(a.probabilities.get(a.choice, 0.0))   # probabilite calibree de l'option retenue (pas l'entropie)
        if a.choice != "prompt" and p >= threshold:
            # une autorisation ne se donne pas sur la foi du seul classifieur : on demande la phrase exacte
            cmd = f"confirm_{a.choice}" if a.choice in PERMISSION_COMMANDS else a.choice
            return Route("command", rest, cmd, round(p, 3), "systemone", ms)
        return Route("prompt", rest, None, round(float(a.probabilities.get("prompt", 0.0)), 3), "systemone", ms)
    except Exception as e:   # classifieur indisponible : la phrase part a l'agent, et on le dit
        return Route("prompt", rest, None, 0.0, "systemone", round((time.perf_counter() - t0) * 1000, 2), f"{type(e).__name__}: {str(e)[:160]}")


def pcm16_to_float(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def to_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sample_rate); w.writeframes(pcm.tobytes())
    return buf.getvalue()


class VoiceUnavailable(RuntimeError):
    pass


class VoiceService:
    """Charge paresseusement les modeles installes ; toutes les inferences sur CPU."""

    def __init__(self, files: Callable[[str], dict | None], settings: Callable, threads: int = 2):
        self.files, self.settings, self.threads = files, settings, threads
        self._stt = self._tts = None
        self._stt_id = self._tts_id = None
        self._lock = threading.Lock()
        try:
            import sherpa_onnx  # noqa: F401
            self.engine = "sherpa-onnx"
        except Exception:
            self.engine = None

    def status(self) -> dict:
        v = self.settings().voice
        return {"engine": self.engine, "stt": v.stt_model, "tts": v.tts_voice, "stt_installed": bool(self.files(v.stt_model)),
                "tts_installed": bool(self.files(v.tts_voice)), "vad_installed": bool(self.files("vad-silero")),
                "stt_loaded": self._stt is not None, "tts_loaded": self._tts is not None}

    # ---- STT ---------------------------------------------------------------------------------------------------
    def _load_stt(self):
        import sherpa_onnx as so
        from prophet_studio.catalog import VOICE
        vid = self.settings().voice.stt_model
        if self._stt is not None and self._stt_id == vid:
            return self._stt
        f = self.files(vid)
        if not f:
            raise VoiceUnavailable(f"modele de reconnaissance {vid} non installe")
        eng = VOICE[vid].engine
        if eng == "parakeet":
            rec = so.OfflineRecognizer.from_transducer(encoder=f["encoder"], decoder=f["decoder"], joiner=f["joiner"], tokens=f["tokens"],
                                                       num_threads=self.threads, model_type="nemo_transducer")
        elif eng == "whisper":
            rec = so.OfflineRecognizer.from_whisper(encoder=f["encoder"], decoder=f["decoder"], tokens=f["tokens"],
                                                    language=self.settings().language, num_threads=self.threads)
        else:
            raise VoiceUnavailable(f"moteur inconnu {eng}")
        self._stt, self._stt_id = rec, vid
        return rec

    def transcribe(self, samples: np.ndarray, sample_rate: int = 16000) -> dict:
        if self.engine is None:
            raise VoiceUnavailable("sherpa-onnx n'est pas installe")
        t0 = time.perf_counter()
        with self._lock:
            rec = self._load_stt()
            st = rec.create_stream()
            st.accept_waveform(sample_rate, samples.astype(np.float32))
            rec.decode_stream(st)
            text = st.result.text.strip()
        return {"text": text, "ms": round((time.perf_counter() - t0) * 1000, 1), "audio_s": round(len(samples) / sample_rate, 2)}

    # ---- TTS ---------------------------------------------------------------------------------------------------
    def _load_tts(self, vid: str):
        import sherpa_onnx as so
        from prophet_studio.catalog import VOICE
        if self._tts is not None and self._tts_id == vid:
            return self._tts
        f = self.files(vid)
        if not f:
            raise VoiceUnavailable(f"voix {vid} non installee")
        eng = VOICE[vid].engine
        if eng == "piper":
            mc = so.OfflineTtsModelConfig(vits=so.OfflineTtsVitsModelConfig(model=f["model"], tokens=f["tokens"], data_dir=f.get("data_dir", "")),
                                          num_threads=self.threads)
        elif eng == "kokoro":
            mc = so.OfflineTtsModelConfig(kokoro=so.OfflineTtsKokoroModelConfig(model=f["model"], voices=f["voices"], tokens=f["tokens"],
                                                                                data_dir=f.get("data_dir", ""), lexicon=f.get("lexicon", "")),
                                          num_threads=self.threads)
        else:
            raise VoiceUnavailable(f"moteur inconnu {eng}")
        self._tts, self._tts_id = so.OfflineTts(so.OfflineTtsConfig(model=mc, max_num_sentences=1)), vid
        return self._tts

    def synthesize(self, text: str, voice: str | None = None, speed: float | None = None, sid: int = 0) -> bytes:
        if self.engine is None:
            raise VoiceUnavailable("sherpa-onnx n'est pas installe")
        v = self.settings().voice
        with self._lock:
            tts = self._load_tts(voice or v.tts_voice)
            audio = tts.generate(clean_for_speech(text), sid=sid, speed=speed or v.speed)
        return to_wav(np.asarray(audio.samples, dtype=np.float32), audio.sample_rate)

    # ---- VAD (mains libres) ----------------------------------------------------------------------------------------
    def vad_session(self) -> "VadSession":
        if self.engine is None:
            raise VoiceUnavailable("sherpa-onnx n'est pas installe")
        f = self.files("vad-silero")
        if not f:
            raise VoiceUnavailable("detecteur de parole (Silero VAD) non installe")
        import sherpa_onnx as so
        cfg = so.VadModelConfig(silero_vad=so.SileroVadModelConfig(model=f["model"], threshold=0.5, min_silence_duration=0.45,
                                                                   min_speech_duration=0.25, max_speech_duration=25),
                                sample_rate=16000, num_threads=1)
        return VadSession(so.VoiceActivityDetector(cfg, buffer_size_in_seconds=60))

    def preload(self) -> None:
        def go():
            for fn in (self._load_stt, lambda: self._load_tts(self.settings().voice.tts_voice)):
                try:
                    with self._lock:
                        fn()
                except Exception:
                    pass
        if self.engine:
            threading.Thread(target=go, daemon=True).start()


class VadSession:
    """Recoit des morceaux PCM 16 kHz, rend les segments de parole termines."""

    def __init__(self, vad):
        self.vad = vad
        self.speaking = False

    def feed(self, samples: np.ndarray) -> list[np.ndarray]:
        self.vad.accept_waveform(samples)
        self.speaking = bool(self.vad.is_speech_detected())
        out = []
        while not self.vad.empty():
            out.append(np.asarray(self.vad.front.samples, dtype=np.float32))
            self.vad.pop()
        return out


_MD = [(re.compile(r"```.*?```", re.S), " (bloc de code) "), (re.compile(r"`([^`]*)`"), r"\1"), (re.compile(r"!\[[^\]]*\]\([^)]*\)"), ""),
       (re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"), (re.compile(r"^#+\s*", re.M), ""), (re.compile(r"[*_~>|]+"), ""),
       (re.compile(r"^\s*[-+]\s+", re.M), ""), (re.compile(r"\s+"), " ")]


def clean_for_speech(text: str, max_chars: int = 1200) -> str:
    """Markdown -> texte a prononcer (le code n'est pas lu)."""
    for rx, rep in _MD:
        text = rx.sub(rep, text)
    return text.strip()[:max_chars]


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?…])\s+", clean_for_speech(text, 4000))
    return [p for p in (s.strip() for s in parts) if p]

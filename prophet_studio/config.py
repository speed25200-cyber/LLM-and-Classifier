"""Chemins de l'application et reglages persistants (settings.json)."""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

APP_DIR_NAME = "ProphetStudio"


def data_dir() -> Path:
    """PROPHET_HOME, sinon le dossier de donnees de l'utilisateur selon le systeme."""
    env = os.environ.get("PROPHET_HOME")
    if env:
        p = Path(env)
    elif sys.platform == "win32":
        p = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / APP_DIR_NAME
    elif sys.platform == "darwin":
        p = Path.home() / "Library" / "Application Support" / APP_DIR_NAME
    else:
        p = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "prophet-studio"
    p.mkdir(parents=True, exist_ok=True)
    return p


class Paths:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else data_dir()
        self.models = self.root / "models"
        self.bin = self.root / "bin"
        self.voice = self.root / "voice"
        self.sessions = self.root / "sessions"
        self.logs = self.root / "logs"
        self.runs = self.root / "runs"
        for d in (self.models, self.bin, self.voice, self.sessions, self.logs, self.runs):
            d.mkdir(parents=True, exist_ok=True)
        self.settings = self.root / "settings.json"
        self.installed = self.root / "installed.json"
        self.core_info = self.root / "core.json"


class VoiceSettings(BaseModel):
    enabled: bool = True
    mode: Literal["ptt", "handsfree"] = "ptt"         # push-to-talk (defaut) ou mains libres (VAD + mot d'eveil)
    wake_word: str = "prophet"
    stt_model: str = "stt-parakeet-v3"
    tts_voice: str = "tts-fr-siwis"
    speak_responses: bool = False
    speed: float = 1.0
    command_threshold: float = 0.72                   # confiance minimale du classifieur pour executer une commande


class Settings(BaseModel):
    language: Literal["fr", "en"] = "fr"
    workspace: str = Field(default_factory=lambda: str(Path.home() / "Prophet"))
    profile: str = "auto"                             # "auto" = planificateur VRAM ; sinon identifiant de profil
    priority: Literal["equilibre", "contexte", "vitesse"] = "equilibre"
    s2_model: str = "auto"
    s1_model: str = "auto"
    permission_mode: Literal["smart", "ask", "auto"] = "smart"
    effort: Literal["auto", "fast", "deep"] = "auto"
    autostart_models: bool = True
    vram_saver: bool = True                           # interface en rendu logiciel : la VRAM reste au modele
    reduce_motion: bool = False
    ctx_override: int = 0
    llama_server_path: str = ""                       # binaire personnalise (build sm_120 maison, par ex.)
    hf_endpoint: str = "https://huggingface.co"       # miroir possible (hf-mirror.com...)
    runtime_tag: str = "prism-b10683-d8f26ee"
    s2_port: int = 7880
    s1_port: int = 7881
    browser_tool: bool = False
    orca_lora: bool = False
    onboarding_done: bool = False
    voice: VoiceSettings = Field(default_factory=VoiceSettings)


class SettingsStore:
    def __init__(self, paths: Paths):
        self.paths = paths
        self._lock = threading.Lock()
        self.value = self._load()

    def _load(self) -> Settings:
        if self.paths.settings.exists():
            try:
                return Settings.model_validate(json.loads(self.paths.settings.read_text(encoding="utf-8")))
            except Exception:
                pass  # reglages corrompus : on repart des valeurs par defaut sans planter
        return Settings()

    def get(self) -> Settings:
        return self.value

    def update(self, patch: dict) -> Settings:
        with self._lock:
            data = self.value.model_dump()
            for k, v in patch.items():
                if isinstance(v, dict) and isinstance(data.get(k), dict):
                    data[k] = {**data[k], **v}
                else:
                    data[k] = v
            self.value = Settings.model_validate(data)
            tmp = self.paths.settings.with_suffix(".tmp")
            tmp.write_text(self.value.model_dump_json(indent=2), encoding="utf-8")
            tmp.replace(self.paths.settings)
            return self.value

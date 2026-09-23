"""Catalogue : modeles (System Two, System One), runtime llama.cpp, modeles vocaux.

Chiffres memoire :
  * poids : tailles publiees par PrismML (docs/02) ; Gio = octets / 2**30.
  * KV par token (f16) derive de l'architecture : 2 (K,V) x couches d'attention complete x tetes KV x dim x 2 octets.
      - Bonsai 2 27B / Bonsai 27B (Qwen3.8 / 3.6 hybrides, 16 couches d'attention complete sur 64) : ~64 Kio/token.
      - Qwen3 1.7B (28 couches, 8 tetes KV, dim 128) : 112 Kio ; Qwen3 4B / 8B (36 couches, 8 x 128) : 144 Kio.
    q8_0 ~ x0.53, q4_0 ~ x0.28 (dont l'en-tete d'echelle par bloc).
  * surcout runtime : contexte CUDA + tampons de calcul (ubatch 512), mesures PrismML pour le 27B (~1 a 1,3 Gio).
Tout est estime, puis confirme a l'execution (NVML) ; le superviseur descend l'echelle si la memoire manque.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

KV_FACTOR = {"f16": 1.0, "q8_0": 0.53, "q4_0": 0.28}


@dataclass
class ModelSpec:
    id: str
    role: str                    # "s2" | "s1" | "voice"
    label: str
    repo: str                    # depot Hugging Face
    pattern: str                 # motif du fichier principal
    size_gb: float               # taille de telechargement (Go decimaux)
    weights_gib: float = 0.0
    kv_kib_f16: float = 0.0      # KV par token en f16
    overhead_gib: float = 0.3
    mmproj_pattern: str = ""
    mmproj_gib: float = 0.0
    runtime: str = "prism"       # "prism" = fork PrismML requis ; "any" = llama.cpp mainline suffit
    quality: str = ""
    params_b: float = 0.0
    thinking: bool = False
    note: str = ""
    tags: list[str] = field(default_factory=list)
    custom: bool = False         # GGUF importe (clone entraine...) : memoire estimee depuis la taille du fichier
    path: str = ""               # fichier local d'un GGUF importe

    def to_dict(self) -> dict:
        return asdict(self)


MODELS: dict[str, ModelSpec] = {m.id: m for m in [
    # ---- System Two : le grand cerveau -------------------------------------------------------------------------
    ModelSpec("bonsai2-27b-ptq1", "s2", "Bonsai 2 27B · PTQ1_0", "prism-ml/Ternary-Bonsai-2-27B-gguf", "*-PTQ1_0.gguf", 5.93,
              weights_gib=5.52, kv_kib_f16=64, overhead_gib=1.0, mmproj_pattern="*mmproj-Q8_0.gguf", mmproj_gib=0.63,
              quality="98,2 % de Qwen3.8-27B FP16 (83,9 / 85,4 sur 20 benchmarks)", params_b=27.4, thinking=True,
              note="Recommande pour 8 Go (RTX 5060 / 4060) : le seul 27B a cette qualite qui laisse la place au classifieur.",
              tags=["recommande", "vision", "outils", "raisonnement"]),
    ModelSpec("bonsai2-27b-pq2", "s2", "Bonsai 2 27B · PQ2_0", "prism-ml/Ternary-Bonsai-2-27B-gguf", "*-PQ2_0.gguf", 7.25,
              weights_gib=6.75, kv_kib_f16=64, overhead_gib=1.1, mmproj_pattern="*mmproj-Q8_0.gguf", mmproj_gib=0.63,
              quality="meme modele, prefill ~2x plus rapide", params_b=27.4, thinking=True,
              note="Pour 12-16 Go (RTX 5060 Ti 16 Go, 4070...) : meme qualite, prefill plus rapide.", tags=["vision", "outils", "raisonnement"]),
    ModelSpec("bonsai-27b-q1", "s2", "Bonsai 27B · 1-bit Q1_0", "prism-ml/Bonsai-27B-gguf", "*-Q1_0.gguf", 3.79,
              weights_gib=3.53, kv_kib_f16=64, overhead_gib=0.9, mmproj_pattern="*mmproj*.gguf", mmproj_gib=0.63,
              quality="89,5 % de Qwen3.6-27B", params_b=27.4, thinking=True, runtime="any",
              note="Profil vitesse : ~1,5x plus rapide en generation, moins precis.", tags=["vitesse"]),
    ModelSpec("bonsai-8b-q1", "s2", "Bonsai 8B · 1-bit Q1_0", "prism-ml/Bonsai-8B-gguf", "*-Q1_0.gguf", 1.15,
              weights_gib=1.07, kv_kib_f16=144, overhead_gib=0.5, quality="petit modele 1-bit (Qwen3 8B)", params_b=8.2,
              thinking=True, runtime="any", note="CPU seul ou GPU 4 Go.", tags=["cpu", "leger"]),
    # ---- System One : le classifieur type Jev (niveau 0 : petit modele ternaire, lecture calibree) ---------------
    ModelSpec("ternary-1.7b", "s1", "Ternary-Bonsai 1.7B (classifieur)", "prism-ml/Ternary-Bonsai-1.7B-gguf", "*-Q2_0_g64.gguf", 0.37,
              weights_gib=0.35, kv_kib_f16=112, overhead_gib=0.25, runtime="any", params_b=1.7,
              note="Decisions en ~50-150 ms ; tourne aussi tres bien sur CPU (prefill ~3 000 tok/s).", tags=["recommande"]),
    ModelSpec("ternary-4b", "s1", "Ternary-Bonsai 4B (classifieur)", "prism-ml/Ternary-Bonsai-4B-gguf", "*-Q2_0_g64.gguf", 0.86,
              weights_gib=0.80, kv_kib_f16=144, overhead_gib=0.3, runtime="any", params_b=4.0,
              note="Plus precis, pour 12 Go et plus.", tags=[]),
    ModelSpec("ternary-8b", "s1", "Ternary-Bonsai 8B (classifieur)", "prism-ml/Ternary-Bonsai-8B-gguf", "*-Q2_0_g64.gguf", 1.75,
              weights_gib=1.63, kv_kib_f16=144, overhead_gib=0.35, runtime="any", params_b=8.0,
              note="Pour 16 Go et plus.", tags=[]),
]}

# ---- GGUF importes (clone entraine, cerveau personnel) ----------------------------------------------------------------
# Architecture inconnue : poids = taille du fichier ; KV par token et surcout pris au-dessus des familles du catalogue
# (classifieurs Qwen3 4B / 8B : 144 Kio ; 27B-32B denses : jusqu'a 256 Kio) : le plan reste prudent, NVML le confirme.
CUSTOM_KV_KIB = {"s1": 160.0, "s2": 256.0}
CUSTOM_OVERHEAD_GIB = {"s1": 0.4, "s2": 1.2}
CUSTOM: dict[str, ModelSpec] = {}    # tenu a jour par l'installateur depuis son registre (installed.json)


def gguf_size(p: Path) -> int:
    """Taille d'un GGUF importe, d'ou le planificateur estime la memoire (point unique : simule dans les tests, qui n'ecrivent
    pas de fichier de plusieurs Go ; sous Windows un fichier "creux" occupe vraiment le disque)."""
    return p.stat().st_size


def custom_spec(model_id: str, role: str, path: str | Path, label: str = "") -> ModelSpec:
    p = Path(path)
    gib = max(gguf_size(p) / 2**30, 1 / 1024)   # plancher 1 Mio : un fichier minuscule ne donne jamais 0 (divisions)
    return ModelSpec(model_id, role, label or p.stem, "", p.name, round(gib * 1.0737, 2), weights_gib=gib,
                     kv_kib_f16=CUSTOM_KV_KIB.get(role, 256.0), overhead_gib=CUSTOM_OVERHEAD_GIB.get(role, 1.2), runtime="any",
                     thinking=role == "s2", note="GGUF importe : memoire estimee depuis la taille du fichier (KV et surcout majores).",
                     tags=["personnel"], custom=True, path=str(p))


def get_model(model_id: str) -> ModelSpec | None:
    """Modele du catalogue, ou GGUF importe dont le fichier est toujours present."""
    if model_id in MODELS:
        return MODELS[model_id]
    m = CUSTOM.get(model_id)
    return m if m is not None and Path(m.path).is_file() else None


# ---- modeles vocaux (sherpa-onnx, CPU : la VRAM reste au LLM) --------------------------------------------------------
SHERPA = "https://github.com/k2-fsa/sherpa-onnx/releases/download"


@dataclass
class VoiceSpec:
    id: str
    kind: str                    # "stt" | "tts" | "vad"
    label: str
    url: str
    size_mb: int
    languages: list[str]
    engine: str                  # "parakeet" | "whisper" | "piper" | "kokoro" | "silero"
    files: dict[str, str] = field(default_factory=dict)   # role -> nom de fichier dans l'archive (motif)
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


VOICE: dict[str, VoiceSpec] = {v.id: v for v in [
    VoiceSpec("vad-silero", "vad", "Silero VAD (detection de parole)", f"{SHERPA}/asr-models/silero_vad.onnx", 2, ["*"], "silero",
              {"model": "silero_vad.onnx"}),
    VoiceSpec("stt-parakeet-v3", "stt", "Parakeet TDT 0.6B v3 (25 langues dont FR)",
              f"{SHERPA}/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2", 640, ["fr", "en", "de", "es", "it", "pt", "nl"], "parakeet",
              {"encoder": "encoder*.onnx", "decoder": "decoder*.onnx", "joiner": "joiner*.onnx", "tokens": "tokens.txt"},
              note="Tres rapide sur CPU, ponctuation incluse."),
    VoiceSpec("stt-whisper-small", "stt", "Whisper small (multilingue, int8)", f"{SHERPA}/asr-models/sherpa-onnx-whisper-small.tar.bz2", 610,
              ["*"], "whisper", {"encoder": "*small-encoder.int8.onnx", "decoder": "*small-decoder.int8.onnx", "tokens": "*small-tokens.txt"}),
    VoiceSpec("tts-fr-siwis", "tts", "Voix francaise Siwis (Piper)", f"{SHERPA}/tts-models/vits-piper-fr_FR-siwis-medium.tar.bz2", 64, ["fr"], "piper",
              {"model": "*.onnx", "tokens": "tokens.txt", "data_dir": "espeak-ng-data"}),
    VoiceSpec("tts-en-lessac", "tts", "English voice Lessac (Piper)", f"{SHERPA}/tts-models/vits-piper-en_US-lessac-medium.tar.bz2", 64, ["en"], "piper",
              {"model": "*.onnx", "tokens": "tokens.txt", "data_dir": "espeak-ng-data"}),
    VoiceSpec("tts-kokoro-multi", "tts", "Kokoro 82M multilingue (FR/EN, plus naturel)", f"{SHERPA}/tts-models/kokoro-multi-lang-v1_0.tar.bz2", 330,
              ["fr", "en"], "kokoro", {"model": "model.onnx", "voices": "voices.bin", "tokens": "tokens.txt", "data_dir": "espeak-ng-data",
                                       "lexicon": "lexicon-us-en.txt"}, note="ff_siwis (FR), af_heart (EN)."),
]}

# ---- runtime llama.cpp -------------------------------------------------------------------------------------------
RUNTIME_REPO = "PrismML-Eng/llama.cpp"      # fork requis pour Bonsai 2 (PTQ1_0 / PQ2_0) ; lit aussi Q1_0 / Q2_0_g64


def kv_mib(model: ModelSpec, ctx: int, kv_type: str = "f16") -> float:
    return model.kv_kib_f16 * KV_FACTOR.get(kv_type, 1.0) * ctx / 1024.0


def catalog_dict() -> dict:
    models = [*MODELS.values(), *(m for m in CUSTOM.values() if get_model(m.id) is not None)]
    return {"models": [m.to_dict() for m in models], "voice": [v.to_dict() for v in VOICE.values()], "runtime_repo": RUNTIME_REPO}

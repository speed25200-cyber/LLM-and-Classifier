#!/bin/sh
# Installe tout ce qu'il faut pour la fusion Jev-clone x Bonsai sur Linux (CUDA / Vulkan / ROCm / CPU).
#   PROFILE=scripts/profiles/rtx4060-8gb-qualite.env ./scripts/setup.sh
# 1. venv Python + dependances   2. binaires llama.cpp du fork PrismML (release prism-b10683)
# 3. poids Bonsai (System Two)   4. poids du petit modele (System One) selon le profil
# Windows : utiliser setup.ps1 du depot PrismML-Eng/Bonsai-demo pour 2-3, puis scripts/windows/*.ps1.
set -e
. "$(dirname "$0")/common.sh"
cd "$ROOT"

RELEASE_TAG="${LLAMA_RELEASE_TAG:-prism-b10683-d8f26ee}"
BASE_URL="https://github.com/PrismML-Eng/llama.cpp/releases/download/$RELEASE_TAG"

info "1/4 environnement Python"
if [ ! -d .venv ]; then python3 -m venv .venv; fi
. .venv/bin/activate
pip install -q --upgrade pip
pip install -q -e ".[serve,dev]" huggingface_hub

info "2/4 binaires llama.cpp (fork PrismML, requis pour Bonsai 2 ; compatibles Q1_0 / Q2_0_g64)"
_arch="$(uname -m)"; _gpu=""; _cuda=""
if command -v nvcc >/dev/null 2>&1; then _cuda=$(nvcc --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p');
elif command -v nvidia-smi >/dev/null 2>&1; then _cuda=$(nvidia-smi | sed -n 's/.*CUDA[ A-Z]*Version:[[:space:]]*\([0-9]*\.[0-9]*\).*/\1/p'); fi
if [ -n "$_cuda" ]; then
    _gpu=cuda; _maj="${_cuda%%.*}"; _min="${_cuda#*.}"
    if [ "$_maj" -gt 13 ] || { [ "$_maj" -eq 13 ] && [ "$_min" -ge 3 ]; }; then _tag=13.3
    elif [ "$_maj" -eq 13 ] || { [ "$_maj" -eq 12 ] && [ "$_min" -ge 8 ]; }; then _tag=12.8
    else _tag=12.4; fi
    ASSET="llama-${RELEASE_TAG}-bin-linux-cuda-${_tag}-x64.tar.gz"; DEST="$BIN_DIR/cuda"
elif command -v rocminfo >/dev/null 2>&1 || command -v hipcc >/dev/null 2>&1; then
    _gpu=rocm; ASSET="llama-${RELEASE_TAG}-bin-ubuntu-rocm-7.2-x64.tar.gz"; DEST="$BIN_DIR/rocm"
elif command -v vulkaninfo >/dev/null 2>&1; then
    _gpu=vulkan; ASSET="llama-${RELEASE_TAG}-bin-ubuntu-vulkan-x64.tar.gz"; DEST="$BIN_DIR/vulkan"
else
    _gpu=cpu; ASSET="llama-${RELEASE_TAG}-bin-ubuntu-${_arch}.tar.gz"; [ "$_arch" = x86_64 ] && ASSET="llama-${RELEASE_TAG}-bin-ubuntu-x64.tar.gz"; DEST="$BIN_DIR/cpu"
fi
info "backend detecte: $_gpu ${_cuda:+(CUDA $_cuda)} -> $ASSET"
if [ ! -x "$DEST/llama-server" ]; then
    mkdir -p "$DEST"; tmp=$(mktemp)
    curl -L --fail --progress-bar "$BASE_URL/$ASSET" -o "$tmp"
    tar -xzf "$tmp" -C "$DEST" --strip-components=1 2>/dev/null || tar -xzf "$tmp" -C "$DEST"
    rm -f "$tmp"
    info "binaires installes dans $DEST"
else
    info "binaires deja presents dans $DEST"
fi
if [ "$_gpu" = vulkan ] && [ "${BONSAI_FAMILY}" = bonsai2 ]; then
    warn "Bonsai 2 (PTQ1_0/PQ2_0) n'a pas encore de noyaux Vulkan : prenez BONSAI_FAMILY=bonsai (Q1_0) ou ternary (Q2_0_g64)."
fi

dl() {  # $1 repo $2 dest $3 motifs (separes par des virgules)
    mkdir -p "$2"
    python - "$1" "$2" "$3" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, dest, pats = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
snapshot_download(repo_id=repo, local_dir=dest, allow_patterns=pats)
PY
}

info "3/4 poids Bonsai (System Two) : $BONSAI_FAMILY $BONSAI_SIZE ${BONSAI_BAND:-}"
pats="$(gguf_pattern "$BONSAI_FAMILY" "$BONSAI_SIZE" "${BONSAI_BAND:-}")"
if [ "$BONSAI_SIZE" = 27B ] && [ "${BONSAI_MMPROJ:-off}" != off ]; then
    if [ "$BONSAI_FAMILY" = bonsai2 ]; then pats="$pats,*mmproj-Q8_0.gguf"; else pats="$pats,*mmproj*.gguf"; fi
fi
if find_gguf "$BONSAI_FAMILY" "$BONSAI_SIZE" "${BONSAI_BAND:-}" >/dev/null 2>&1; then
    info "deja present: $(find_gguf "$BONSAI_FAMILY" "$BONSAI_SIZE" "${BONSAI_BAND:-}")"
else
    dl "$(hf_repo "$BONSAI_FAMILY" "$BONSAI_SIZE")" "$(model_dir "$BONSAI_FAMILY" "$BONSAI_SIZE")" "$pats"
fi

if [ "${JEV_MODE:-dual}" = dual ]; then
    info "4/4 poids du clone Jev (System One) : $JEV_FAMILY $JEV_SIZE"
    if find_gguf "$JEV_FAMILY" "$JEV_SIZE" "${JEV_BAND:-}" >/dev/null 2>&1; then
        info "deja present: $(find_gguf "$JEV_FAMILY" "$JEV_SIZE" "${JEV_BAND:-}")"
    else
        dl "$(hf_repo "$JEV_FAMILY" "$JEV_SIZE")" "$(model_dir "$JEV_FAMILY" "$JEV_SIZE")" "$(gguf_pattern "$JEV_FAMILY" "$JEV_SIZE" "${JEV_BAND:-}")"
    fi
else
    info "4/4 mode mono : pas de second modele"
fi
echo
info "termine. Lancer :  ./scripts/start_bonsai.sh   puis   ./scripts/start_jev_clone.sh   puis   jev serve"

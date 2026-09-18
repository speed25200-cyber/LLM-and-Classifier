#!/bin/sh
# Fonctions partagees : chargement du profil, resolution des fichiers GGUF, binaires llama.cpp.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROFILE="${PROFILE:-$ROOT/scripts/profiles/rtx4060-8gb-qualite.env}"
[ -f "$PROFILE" ] || { echo "profil introuvable: $PROFILE" >&2; exit 1; }
# shellcheck disable=SC1090
set -a; . "$PROFILE"; set +a
MODELS="${MODELS:-$ROOT/models}"
BIN_DIR="${BIN_DIR:-$ROOT/bin}"
BONSAI_SIZE="${BONSAI_SIZE:-27B}"

info() { printf '  \033[32m*\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*" >&2; }
err()  { printf '  \033[31mx\033[0m %s\n' "$*" >&2; }

# Repo HF + motif de fichier pour (famille, taille, band)
hf_repo() {  # $1 famille $2 taille
    case "$1" in
        bonsai2) echo "prism-ml/Ternary-Bonsai-2-$2-gguf" ;;
        bonsai)  echo "prism-ml/Bonsai-$2-gguf" ;;
        ternary) echo "prism-ml/Ternary-Bonsai-$2-gguf" ;;
        *) err "famille inconnue: $1"; exit 1 ;;
    esac
}
gguf_pattern() {  # $1 famille $2 taille $3 band
    case "$1" in
        bonsai2) echo "*-${3:-PTQ1_0}.gguf" ;;
        bonsai)  echo "*-Q1_0.gguf" ;;
        ternary) if [ "$2" = "27B" ]; then
                     case "${3:-Q2_0_g64}" in PQ2_0) echo "*-PQ2_0.gguf" ;; *) echo "*-Q2_g64.gguf" ;; esac
                 else
                     case "${3:-Q2_0_g64}" in PQ2_0) echo "*-PQ2_0.gguf" ;; *) echo "*-Q2_0_g64.gguf" ;; esac
                 fi ;;
    esac
}
model_dir() { echo "$MODELS/$1-$2"; }
find_gguf() {  # $1 famille $2 taille $3 band -> chemin du premier fichier correspondant
    d="$(model_dir "$1" "$2")"; p="$(gguf_pattern "$1" "$2" "$3")"
    for f in "$d"/$p; do [ -f "$f" ] && echo "$f" && return 0; done
    return 1
}
find_mmproj() { d="$(model_dir "$1" "$2")"; for f in "$d"/*mmproj*.gguf; do [ -f "$f" ] && echo "$f" && return 0; done; return 1; }
find_server() {
    for b in "$BIN_DIR/cuda/llama-server" "$BIN_DIR/vulkan/llama-server" "$BIN_DIR/rocm/llama-server" "$BIN_DIR/cpu/llama-server" "$(command -v llama-server 2>/dev/null)"; do
        [ -n "$b" ] && [ -x "$b" ] && echo "$b" && return 0
    done
    return 1
}

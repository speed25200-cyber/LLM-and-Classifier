#!/bin/sh
# Recupere l'adaptateur LoRA "OrcaBonsai" (ablation de la direction de refus de Ternary Bonsai 2 27B, a
# l'execution, poids de base inchanges) et verifie son empreinte. Seul le fichier GGUF de 9,7 Mo est utilise
# par llama-server ; le code Python du depot (MLX, Apple) n'est jamais execute ici.
#   ./scripts/fetch_orcabonsai.sh
#   BONSAI_LORA=third_party/orcabonsai/gguf/bonsai-abliterate-lora.gguf BONSAI_LORA_SCALE=1.0 ./scripts/start_bonsai.sh
set -e
. "$(dirname "$0")/common.sh"
cd "$ROOT"
REPO="https://github.com/Continuum-AI-Corp/OrcaBonsai-27B-Uncensored.git"
DEST="third_party/orcabonsai"
EXPECTED_SHA="f1669534803d340a496015f5c45125f3437b4d13ec764f40e34488ce83967f42"   # README du depot + verification independante (18.09.2026)
if [ ! -d "$DEST/.git" ]; then
    mkdir -p third_party
    git clone -q --depth 1 "$REPO" "$DEST"
fi
F="$DEST/gguf/bonsai-abliterate-lora.gguf"
[ -f "$F" ] || { err "adaptateur introuvable : $F"; exit 1; }
if command -v sha256sum >/dev/null 2>&1; then GOT="$(sha256sum "$F" | cut -d' ' -f1)"; else GOT="$(shasum -a 256 "$F" | cut -d' ' -f1)"; fi
if [ "$GOT" != "$EXPECTED_SHA" ]; then
    err "empreinte inattendue pour $F"
    echo "  attendu : $EXPECTED_SHA"; echo "  obtenu  : $GOT"
    echo "  Le depot a change depuis la verification : relire son README avant d'utiliser ce fichier."
    exit 1
fi
info "adaptateur verifie : $F ($(du -h "$F" | cut -f1), sha256 OK)"
echo
echo "  Utilisation (fork PrismML obligatoire, teste par les auteurs sur PTQ1_0 uniquement) :"
echo "    BONSAI_LORA=$F BONSAI_LORA_SCALE=1.0 ./scripts/start_bonsai.sh"
echo "  ou le profil : PROFILE=scripts/profiles/rtx4060-8gb-orcabonsai.env ./scripts/start_bonsai.sh"
echo "  Echelle : 0 = modele publie, 1 = projection exacte, 2 = plus fort, >= 3 = degradation."

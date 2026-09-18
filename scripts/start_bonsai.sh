#!/bin/sh
# Lance llama-server (System Two) avec Bonsai selon le profil. Arguments supplementaires passes a llama-server.
#   PROFILE=scripts/profiles/rtx4060-8gb-qualite.env ./scripts/start_bonsai.sh
set -e
. "$(dirname "$0")/common.sh"
BIN="$(find_server)" || { err "llama-server introuvable : ./scripts/setup.sh"; exit 1; }
MODEL="$(find_gguf "$BONSAI_FAMILY" "$BONSAI_SIZE" "${BONSAI_BAND:-}")" || { err "poids Bonsai introuvables : ./scripts/setup.sh"; exit 1; }
export LD_LIBRARY_PATH="$(dirname "$BIN")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mm=""
case "${BONSAI_MMPROJ:-off}" in
    off) mm="--no-mmproj" ;;
    cpu) MP="$(find_mmproj "$BONSAI_FAMILY" "$BONSAI_SIZE" || true)"; [ -n "$MP" ] && mm="--mmproj $MP --no-mmproj-offload" || mm="--no-mmproj" ;;
    gpu) MP="$(find_mmproj "$BONSAI_FAMILY" "$BONSAI_SIZE" || true)"; [ -n "$MP" ] && mm="--mmproj $MP" || mm="--no-mmproj" ;;
esac
kv=""; [ "${BONSAI_KV4:-0}" = 1 ] && kv="--cache-type-k q4_0 --cache-type-v q4_0"
if [ "$BONSAI_FAMILY" = bonsai2 ]; then SAMPLING="--temp 1.0 --top-p 0.95 --top-k 20"; else SAMPLING="--temp 0.7 --top-p 0.95 --top-k 20 --min-p 0"; fi

echo "=== Bonsai (System Two) ==="
echo "  modele  : $MODEL"
echo "  binaire : $BIN"
echo "  ctx=$BONSAI_CTX ngl=$BONSAI_NGL slots=$BONSAI_NP kv4=${BONSAI_KV4:-0} mmproj=${BONSAI_MMPROJ:-off} budget=${BONSAI_REASONING_BUDGET:--1}"
echo "  API     : http://127.0.0.1:${BONSAI_PORT:-8080}/v1/chat/completions"
# --cache-ram : cache de prompts en RAM (reutilisation du prefixe entre slots) ; --ctx-checkpoints : points de
# reprise de l'etat recurrent (modeles hybrides GDN) pour reutiliser un prefixe partiel.
exec "$BIN" -m "$MODEL" --host 127.0.0.1 --port "${BONSAI_PORT:-8080}" \
    -ngl "$BONSAI_NGL" -fa on -c "$BONSAI_CTX" -np "$BONSAI_NP" \
    $SAMPLING --jinja $mm $kv \
    --reasoning-budget "${BONSAI_REASONING_BUDGET:--1}" \
    --cache-ram "${BONSAI_CACHE_RAM:-2048}" --ctx-checkpoints 8 \
    "$@"

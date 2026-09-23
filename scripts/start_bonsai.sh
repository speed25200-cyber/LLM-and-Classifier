#!/bin/sh
# Lance llama-server (System Two) avec Bonsai selon le profil. Arguments supplementaires passes a llama-server.
#   PROFILE=scripts/profiles/rtx4060-8gb-qualite.env ./scripts/start_bonsai.sh
set -e
. "$(dirname "$0")/common.sh"
BIN="$(find_server)" || { err "llama-server introuvable : ./scripts/setup.sh"; exit 1; }
if [ -n "${BONSAI_GGUF:-}" ]; then
    MODEL="$BONSAI_GGUF"; case "$MODEL" in /*) ;; *) MODEL="$ROOT/$MODEL" ;; esac   # GGUF quelconque (chemin explicite)
    [ -f "$MODEL" ] || { err "BONSAI_GGUF=$BONSAI_GGUF introuvable"; exit 1; }
else
    MODEL="$(find_gguf "$BONSAI_FAMILY" "$BONSAI_SIZE" "${BONSAI_BAND:-}")" || { err "poids Bonsai introuvables : ./scripts/setup.sh"; exit 1; }
fi
export LD_LIBRARY_PATH="$(dirname "$BIN")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Adaptateur LoRA optionnel (ex. OrcaBonsai : ablation de refus, rang 1, 9,7 Mo) applique dans le graphe par
# llama.cpp, poids de base inchanges. Echelle reglable ici et par requete ("lora": [{"id": 0, "scale": s}]).
lora=""
if [ -n "${BONSAI_LORA:-}" ]; then
    LORA="$BONSAI_LORA"; case "$LORA" in /*) ;; *) LORA="$ROOT/$LORA" ;; esac
    [ -f "$LORA" ] || { err "BONSAI_LORA=$BONSAI_LORA introuvable (./scripts/fetch_orcabonsai.sh)"; exit 1; }
    lora="--lora-scaled $LORA:${BONSAI_LORA_SCALE:-1.0}"
fi

mm=""
case "${BONSAI_MMPROJ:-off}" in
    off) mm="--no-mmproj" ;;
    cpu) MP="$(find_mmproj "$BONSAI_FAMILY" "$BONSAI_SIZE" || true)"; [ -n "$MP" ] && mm="--mmproj $MP --no-mmproj-offload" || mm="--no-mmproj" ;;
    gpu) MP="$(find_mmproj "$BONSAI_FAMILY" "$BONSAI_SIZE" || true)"; [ -n "$MP" ] && mm="--mmproj $MP" || mm="--no-mmproj" ;;
esac
kv=""; [ "${BONSAI_KV4:-0}" = 1 ] && kv="--cache-type-k q4_0 --cache-type-v q4_0"
[ -n "${BONSAI_KV_TYPE:-}" ] && kv="--cache-type-k $BONSAI_KV_TYPE --cache-type-v $BONSAI_KV_TYPE"   # q8_0 / f16 (cartes 12-16 Go)
if [ "$BONSAI_FAMILY" = bonsai2 ]; then SAMPLING="--temp 1.0 --top-p 0.95 --top-k 20"; else SAMPLING="--temp 0.7 --top-p 0.95 --top-k 20 --min-p 0"; fi

echo "=== Bonsai (System Two) ==="
echo "  modele  : $MODEL"
echo "  binaire : $BIN"
echo "  ctx=$BONSAI_CTX par slot x $BONSAI_NP slots ngl=$BONSAI_NGL kv4=${BONSAI_KV4:-0} mmproj=${BONSAI_MMPROJ:-off} budget=${BONSAI_REASONING_BUDGET:--1}"
[ -n "$lora" ] && echo "  LoRA    : $LORA (echelle ${BONSAI_LORA_SCALE:-1.0})"
echo "  API     : http://127.0.0.1:${BONSAI_PORT:-8080}/v1/chat/completions"
# --cache-ram : cache de prompts en RAM (reutilisation du prefixe entre slots) ; --ctx-checkpoints : points de
# reprise de l'etat recurrent (modeles hybrides GDN) pour reutiliser un prefixe partiel.
# BONSAI_CTX = contexte PAR SLOT, comme le planificateur de Prophet Studio (planner.kv_ctx) : -c = BONSAI_CTX x BONSAI_NP,
# KV non unifie, chaque slot (conversation ou lecture System One en mode mono) a son contexte entier.
exec "$BIN" -m "$MODEL" --host 127.0.0.1 --port "${BONSAI_PORT:-8080}" \
    -ngl "$BONSAI_NGL" -fa on -c "$((BONSAI_CTX * BONSAI_NP))" -np "$BONSAI_NP" \
    $SAMPLING --jinja $mm $kv \
    --reasoning-budget "${BONSAI_REASONING_BUDGET:--1}" \
    --cache-ram "${BONSAI_CACHE_RAM:-2048}" --ctx-checkpoints 8 \
    $lora \
    "$@"

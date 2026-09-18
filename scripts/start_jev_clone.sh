#!/bin/sh
# Lance le llama-server du clone Jev (System One) : petit GGUF, plusieurs slots, aucune generation.
#   PROFILE=scripts/profiles/rtx4060-8gb-qualite.env ./scripts/start_jev_clone.sh
# Pour servir un modele entraine (training/) : JEV_GGUF=runs/jev-qwen35-0.8b-Q8_0.gguf ./scripts/start_jev_clone.sh
set -e
. "$(dirname "$0")/common.sh"
if [ "${JEV_MODE:-dual}" = mono ]; then
    info "JEV_MODE=mono : le clone Jev lit ses probabilites sur le serveur Bonsai (port ${BONSAI_PORT:-8080}). Rien a lancer."
    info "  export JEV_S1_URL=http://127.0.0.1:${BONSAI_PORT:-8080}  JEV_S2_URL=http://127.0.0.1:${BONSAI_PORT:-8080}"
    exit 0
fi
BIN="$(find_server)" || { err "llama-server introuvable : ./scripts/setup.sh"; exit 1; }
MODEL="${JEV_GGUF:-$(find_gguf "$JEV_FAMILY" "$JEV_SIZE" "${JEV_BAND:-}")}" || { err "poids du clone introuvables : ./scripts/setup.sh"; exit 1; }
export LD_LIBRARY_PATH="$(dirname "$BIN")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "=== Clone Jev (System One) ==="
echo "  modele  : $MODEL"
echo "  ctx=$JEV_CTX ngl=$JEV_NGL slots=$JEV_NP  ->  http://127.0.0.1:${JEV_PORT:-8081}"
echo "  export JEV_S1_URL=http://127.0.0.1:${JEV_PORT:-8081}  JEV_S2_URL=http://127.0.0.1:${BONSAI_PORT:-8080}"
# Pas de generation : la temperature/top-k n'importent pas (le client envoie samplers=[] + grammaire).
# --reasoning-budget 0 evite toute reflexion si un client passe par /v1/chat/completions.
exec "$BIN" -m "$MODEL" --host 127.0.0.1 --port "${JEV_PORT:-8081}" \
    -ngl "$JEV_NGL" -fa on -c "$JEV_CTX" -np "$JEV_NP" -b 2048 -ub 512 \
    --cache-ram "${JEV_CACHE_RAM:-1024}" --ctx-checkpoints 8 --reasoning-budget 0 --no-mmproj \
    "$@"

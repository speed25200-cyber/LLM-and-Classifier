#!/bin/sh
# Mesure : (1) debit brut des deux modeles avec llama-bench (pp512 / tg128), (2) latence d'une decision System One.
#   PROFILE=... ./scripts/bench.sh
set -e
. "$(dirname "$0")/common.sh"
BIN="$(find_server)" || exit 1
BENCH="$(dirname "$BIN")/llama-bench"
export LD_LIBRARY_PATH="$(dirname "$BIN")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
M2="$(find_gguf "$BONSAI_FAMILY" "$BONSAI_SIZE" "${BONSAI_BAND:-}")"
echo "== llama-bench Bonsai ($M2) =="; "$BENCH" -m "$M2" -ngl "$BONSAI_NGL" -fa 1 -p 512 -n 128 -r 3
if [ "${JEV_MODE:-dual}" = dual ]; then
    M1="${JEV_GGUF:-$(find_gguf "$JEV_FAMILY" "$JEV_SIZE" "${JEV_BAND:-}")}"
    echo "== llama-bench clone Jev ($M1) =="; "$BENCH" -m "$M1" -ngl "$JEV_NGL" -fa 1 -p 512 -n 128 -r 3
fi
echo "== latence System One (serveur du clone doit tourner) =="
. "$ROOT/.venv/bin/activate" 2>/dev/null || true
python -m jev_clone.cli bench --server "http://127.0.0.1:${JEV_PORT:-8081}" --n 20 || warn "lancer ./scripts/start_jev_clone.sh d'abord"

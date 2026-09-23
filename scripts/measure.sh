#!/bin/sh
# Prophet Studio : mesure du duo reel (S1 classifieur + S2 raisonnement), en une commande (Linux, macOS, Git Bash).
#   sh scripts/measure.sh [--rapide] [--sans-tours] [--label "pilote 575"] [autres options de eval/measure_duo.py]
# Trouve le Prophet Studio en marche (core.json du dossier de donnees : PROPHET_HOME, sinon ~/.local/share/prophet-studio,
# ~/Library/Application Support/ProphetStudio ou %LOCALAPPDATA%\ProphetStudio), demarre les modeles s'ils sont arretes, lance
# eval/measure_duo.py avec l'environnement Python installe (<donnees>/app-venv des scripts d'installation, sinon <donnees>/venv
# de l'application de bureau, sinon uv run depuis ce depot), puis affiche le fichier de resultats (eval/results/duo-<date>.json)
# et le tableau a coller dans eval/SCOREBOARD.md. Duree : ~5 minutes (--rapide : ~2 minutes).
set -eu
ROOT=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
if [ -n "${PROPHET_HOME:-}" ]; then
  DATA=$PROPHET_HOME
elif [ "$(uname -s)" = Darwin ]; then
  DATA="$HOME/Library/Application Support/ProphetStudio"
elif [ -n "${LOCALAPPDATA:-}" ]; then
  DATA="$LOCALAPPDATA/ProphetStudio"
else
  DATA="${XDG_DATA_HOME:-$HOME/.local/share}/prophet-studio"
fi
if [ ! -f "$DATA/core.json" ] && [ ! -f "$DATA/demo/core.json" ]; then
  echo "Prophet Studio ne tourne pas : aucun core.json dans $DATA." >&2
  echo "Lancez Prophet Studio (prophet-studio ou l'application de bureau), puis relancez cette commande." >&2
  exit 2
fi
echo "Prophet Studio : $DATA"
cd "$ROOT"
for v in app-venv venv; do
  for py in "$DATA/$v/bin/python" "$DATA/$v/Scripts/python.exe"; do
    if [ -x "$py" ]; then
      echo "Python : $py"
      exec "$py" -m eval.measure_duo --studio --data-dir "$DATA" --demarrer "$@"
    fi
  done
done
if command -v uv >/dev/null 2>&1; then
  echo "Python : uv run (depot)"
  exec uv run python -m eval.measure_duo --studio --data-dir "$DATA" --demarrer "$@"
fi
echo "Aucun Python : ni $DATA/app-venv, ni $DATA/venv, ni uv (installer/install.sh installe tout)." >&2
exit 2

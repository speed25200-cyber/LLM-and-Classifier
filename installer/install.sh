#!/bin/sh
# Prophet Studio : installation en une commande pour Linux et macOS (navigateur comme interface).
#
#   curl -fsSL https://raw.githubusercontent.com/speed25200-cyber/LLM-and-Classifier/main/installer/install.sh | sh
#   curl -fsSL .../installer/install.sh | sh -s -- --ref v0.2.0      # une version precise
#   ./installer/install.sh --source .                                 # depuis un depot clone
#
# Relancer le script = mettre a jour. Tout va dans le dossier de donnees de Prophet Studio
# (code, environnement Python, modeles) ; en dehors : uv s'il manque, une entree de menu (Linux)
# ou un lanceur .command (macOS), et le lien ~/.local/bin/prophet-studio.
set -eu

REPO="${PROPHET_REPO:-speed25200-cyber/LLM-and-Classifier}"
REF="${PROPHET_REF:-main}"
SOURCE="${PROPHET_SOURCE:-}"
DATA_DIR="${PROPHET_HOME:-}"
SHORTCUTS=1
PYTHON_VERSION=3.11

# ---- affichage ------------------------------------------------------------------------------------------------
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  C_STEP=$(printf '\033[1;36m'); C_OK=$(printf '\033[1;32m'); C_WARN=$(printf '\033[1;33m')
  C_ERR=$(printf '\033[1;31m'); C_DIM=$(printf '\033[2m'); C_OFF=$(printf '\033[0m')
else
  C_STEP=''; C_OK=''; C_WARN=''; C_ERR=''; C_DIM=''; C_OFF=''
fi
step() { printf '\n%s==> %s%s\n' "$C_STEP" "$*" "$C_OFF"; }
info() { printf '    %s\n' "$*"; }
ok() { printf '    %sOK%s %s\n' "$C_OK" "$C_OFF" "$*"; }
warn() { printf '    %sATTENTION%s %s\n' "$C_WARN" "$C_OFF" "$*"; }
die() { printf '\n%sERREUR%s %s\n' "$C_ERR" "$C_OFF" "$*" >&2; exit 1; }

usage() {
  cat <<EOF
Installe ou met a jour Prophet Studio (Linux, macOS).

  install.sh [--ref REF] [--source DOSSIER] [--data-dir DOSSIER] [--repo PROPRIETAIRE/DEPOT] [--no-shortcut]

  --ref REF          branche, etiquette ou commit a telecharger depuis GitHub (defaut : main)
  --source DOSSIER   installer depuis un depot deja clone au lieu de telecharger
  --data-dir DOSSIER dossier de donnees (defaut : celui de Prophet Studio, ou PROPHET_HOME)
  --repo P/D         depot GitHub (defaut : $REPO)
  --no-shortcut      ne cree ni entree de menu, ni lanceur, ni lien dans ~/.local/bin
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --ref) [ $# -ge 2 ] || die "--ref attend une valeur"; REF=$2; shift 2 ;;
    --ref=*) REF=${1#*=}; shift ;;
    --source) [ $# -ge 2 ] || die "--source attend un dossier"; SOURCE=$2; shift 2 ;;
    --source=*) SOURCE=${1#*=}; shift ;;
    --data-dir) [ $# -ge 2 ] || die "--data-dir attend un dossier"; DATA_DIR=$2; shift 2 ;;
    --data-dir=*) DATA_DIR=${1#*=}; shift ;;
    --repo) [ $# -ge 2 ] || die "--repo attend proprietaire/depot"; REPO=$2; shift 2 ;;
    --repo=*) REPO=${1#*=}; shift ;;
    --no-shortcut) SHORTCUTS=0; shift ;;
    -h | --help) usage; exit 0 ;;
    *) usage >&2; die "option inconnue : $1" ;;
  esac
done

# ---- systeme ----------------------------------------------------------------------------------------------------
step "Verification du systeme"
[ -n "${HOME:-}" ] || die "la variable HOME n'est pas definie"
case "$(uname -s)" in
  Linux) OS=linux ;;
  Darwin) OS=macos ;;
  *) die "systeme non pris en charge : $(uname -s) (Windows : utilisez installer/install.ps1)" ;;
esac
ARCH=$(uname -m)
case "$ARCH" in
  x86_64 | amd64 | aarch64 | arm64) ;;
  *) warn "architecture $ARCH non testee" ;;
esac
if [ -z "$DATA_DIR" ]; then
  if [ "$OS" = macos ]; then
    DATA_DIR="$HOME/Library/Application Support/ProphetStudio"
  else
    DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/prophet-studio"
  fi
fi
APP="$DATA_DIR/app"
VENV="$DATA_DIR/app-venv"
ok "$OS / $ARCH"
info "dossier de donnees : $DATA_DIR"

# GPU NVIDIA : une RTX 50xx (Blackwell, sm_120) exige CUDA 12.8, donc un pilote >= 570.
if [ "$OS" = linux ]; then
  if command -v nvidia-smi >/dev/null 2>&1 && GPUS=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null); then
    printf '%s\n' "$GPUS" | while IFS=, read -r gpu_name gpu_driver; do
      gpu_name=$(printf '%s' "$gpu_name" | sed 's/^ *//; s/ *$//')
      gpu_driver=$(printf '%s' "$gpu_driver" | sed 's/^ *//; s/ *$//')
      ok "GPU : $gpu_name (pilote $gpu_driver)"
      gpu_major=${gpu_driver%%.*}
      case "$gpu_major" in '' | *[!0-9]*) continue ;; esac
      case "$gpu_name" in
        *"RTX 50"[0-9][0-9]*)
          if [ "$gpu_major" -lt 570 ]; then
            warn "carte RTX 50xx (Blackwell) avec le pilote $gpu_driver : il faut un pilote >= 570 (CUDA 12.8)."
            warn "Mettez le pilote a jour : https://www.nvidia.com/Download/index.aspx"
          fi
          ;;
      esac
    done
  else
    warn "aucun GPU NVIDIA detecte (nvidia-smi absent) : les modeles tourneront sur le processeur, bien plus lentement."
  fi
else
  info "macOS : les modeles utilisent Metal (Apple Silicon conseille)."
fi

# ---- outils -----------------------------------------------------------------------------------------------------
download() { # download URL FICHIER
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL --retry 3 --proto '=https' --tlsv1.2 -o "$2" "$1"
  elif command -v wget >/dev/null 2>&1; then
    wget -q -O "$2" "$1"
  else
    die "ni curl ni wget : installez l'un des deux"
  fi
}

find_uv() {
  for candidate in "$(command -v uv 2>/dev/null || true)" "${XDG_BIN_HOME:-$HOME/.local/bin}/uv" \
    "$HOME/.local/bin/uv" "${CARGO_HOME:-$HOME/.cargo}/bin/uv"; do
    if [ -n "$candidate" ] && [ -x "$candidate" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

step "Gestionnaire Python uv"
if UV=$(find_uv); then
  ok "uv present : $UV ($("$UV" --version 2>/dev/null || echo "version inconnue"))"
else
  info "uv absent : installation depuis https://astral.sh/uv/install.sh"
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  else
    wget -qO- https://astral.sh/uv/install.sh | sh
  fi
  UV=$(find_uv) || die "uv reste introuvable apres son installation"
  ok "uv installe : $UV"
fi

TMP=$(mktemp -d 2>/dev/null || mktemp -d -t prophet-studio)
trap 'rm -rf "$TMP"' EXIT
trap 'exit 130' INT TERM

# ---- sources ----------------------------------------------------------------------------------------------------
if [ -n "$SOURCE" ]; then
  step "Sources : depot local"
  STAGE=$(cd "$SOURCE" && pwd) || die "dossier introuvable : $SOURCE"
else
  step "Telechargement de Prophet Studio ($REPO @ $REF)"
  download "https://codeload.github.com/$REPO/tar.gz/$REF" "$TMP/src.tar.gz" ||
    die "telechargement impossible (reference '$REF' inconnue ou reseau indisponible)"
  mkdir "$TMP/src"
  tar -xzf "$TMP/src.tar.gz" -C "$TMP/src"
  STAGE=$(find "$TMP/src" -mindepth 1 -maxdepth 1 -type d | head -n 1)
fi
if [ ! -f "$STAGE/pyproject.toml" ] || [ ! -d "$STAGE/prophet_studio" ]; then
  die "$STAGE ne contient pas Prophet Studio (pyproject.toml + prophet_studio/)"
fi
ok "$STAGE"

if [ -f "$DATA_DIR/core.json" ] && command -v curl >/dev/null 2>&1; then
  running_url=$(sed -n 's/.*"url": *"\([^"]*\)".*/\1/p' "$DATA_DIR/core.json")
  if [ -n "$running_url" ] && curl -fs --max-time 2 "$running_url/api/health" >/dev/null 2>&1; then
    warn "Prophet Studio tourne encore ($running_url) : redemarrez-le apres l'installation pour utiliser la nouvelle version."
  fi
fi

step "Copie de l'application"
mkdir -p "$DATA_DIR"
NEW="$DATA_DIR/app.new"
rm -rf "$NEW"
mkdir -p "$NEW"
for item in pyproject.toml uv.lock README.md LICENSE jev_clone prophet_studio; do
  if [ -e "$STAGE/$item" ]; then cp -R "$STAGE/$item" "$NEW/"; fi
done
find "$NEW" -name __pycache__ -type d -prune -exec rm -rf {} +

# Interface web : deja construite (paquet), sinon ui/dist du depot, sinon construite avec npm,
# sinon telechargee depuis la derniere version publiee (extraite apres uv sync).
WEB_ZIP=""
if [ -f "$NEW/prophet_studio/web/index.html" ]; then
  ok "interface deja construite"
elif [ -f "$STAGE/ui/dist/index.html" ]; then
  cp -R "$STAGE/ui/dist" "$NEW/prophet_studio/web"
  ok "interface copiee depuis ui/dist"
else
  if [ -f "$STAGE/ui/package.json" ] && command -v npm >/dev/null 2>&1; then
    info "construction de l'interface avec npm (1 a 2 minutes)"
    (cd "$STAGE" && tar -cf - --exclude=node_modules --exclude=dist ui) | (cd "$NEW" && tar -xf -)
    if (cd "$NEW/ui" && npm ci --no-audit --no-fund --loglevel=error && npm run build --silent) >"$TMP/npm.log" 2>&1 &&
      [ -f "$NEW/prophet_studio/web/index.html" ]; then
      ok "interface construite"
    else
      warn "construction de l'interface echouee (journal : voir ci-dessous)"
      tail -n 15 "$TMP/npm.log" | sed 's/^/      /'
    fi
    rm -rf "$NEW/ui"
  fi
  if [ ! -f "$NEW/prophet_studio/web/index.html" ]; then
    if download "https://github.com/$REPO/releases/latest/download/prophet-studio-web.zip" "$TMP/web.zip" 2>/dev/null; then
      WEB_ZIP="$TMP/web.zip"
      ok "interface precompilee telechargee"
    else
      warn "interface web introuvable : installez Node.js 20+ puis relancez, ou utilisez l'application de bureau."
    fi
  fi
fi

# icone (menu Linux)
if [ -f "$STAGE/desktop/src-tauri/icons/icon.png" ]; then
  cp "$STAGE/desktop/src-tauri/icons/icon.png" "$DATA_DIR/prophet-studio.png"
fi

rm -rf "$APP"
mv "$NEW" "$APP"
ok "$APP"

# ---- dependances ------------------------------------------------------------------------------------------------
step "Environnement Python (uv sync, 1 a 3 minutes la premiere fois)"
# --inexact : garde les paquets ajoutes a la main (ex. playwright pour l'outil navigateur) d'une mise a jour a l'autre
UV_PROJECT_ENVIRONMENT="$VENV" "$UV" sync --inexact --project "$APP" --extra studio --python "$PYTHON_VERSION" ||
  die "uv sync a echoue (voir les messages ci-dessus)"
if [ -n "$WEB_ZIP" ]; then
  "$VENV/bin/python" -m zipfile -e "$WEB_ZIP" "$APP/prophet_studio/web" || warn "extraction de l'interface echouee"
fi
"$VENV/bin/prophet-studio" --help >/dev/null 2>&1 || die "prophet-studio ne demarre pas depuis $VENV"
ok "prophet-studio pret : $VENV/bin/prophet-studio"

# ---- lanceurs ---------------------------------------------------------------------------------------------------
LAUNCHER="$VENV/bin/prophet-studio"
if [ "$SHORTCUTS" = 1 ]; then
  step "Lanceurs"
  BIN_DIR="$HOME/.local/bin"
  mkdir -p "$BIN_DIR"
  if [ ! -e "$BIN_DIR/prophet-studio" ] || [ -L "$BIN_DIR/prophet-studio" ]; then
    ln -sf "$LAUNCHER" "$BIN_DIR/prophet-studio"
    ok "commande : $BIN_DIR/prophet-studio"
    case ":$PATH:" in *":$BIN_DIR:"*) ;; *) warn "$BIN_DIR n'est pas dans le PATH de ce terminal" ;; esac
  else
    warn "$BIN_DIR/prophet-studio existe deja (fichier non gere par ce script) : laisse tel quel"
  fi
  if [ "$OS" = linux ]; then
    APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
    mkdir -p "$APPS"
    ICON_LINE=""
    if [ -f "$DATA_DIR/prophet-studio.png" ]; then ICON_LINE="Icon=$DATA_DIR/prophet-studio.png"; fi
    cat >"$APPS/prophet-studio.desktop" <<EOF
[Desktop Entry]
Type=Application
Version=1.0
Name=Prophet Studio
Comment=Bonsai 2 27B + classifieur calibre, en local
Exec="$LAUNCHER"
$ICON_LINE
Terminal=false
Categories=Development;Utility;
StartupNotify=false
EOF
    chmod 644 "$APPS/prophet-studio.desktop"
    if command -v update-desktop-database >/dev/null 2>&1; then update-desktop-database "$APPS" >/dev/null 2>&1 || true; fi
    ok "menu des applications : $APPS/prophet-studio.desktop"
  else
    mkdir -p "$HOME/Applications"
    COMMAND_FILE="$HOME/Applications/Prophet Studio.command"
    cat >"$COMMAND_FILE" <<EOF
#!/bin/sh
# Lance Prophet Studio (l'interface s'ouvre dans le navigateur). Fermer cette fenetre arrete le moteur.
exec "$LAUNCHER" "\$@"
EOF
    chmod 755 "$COMMAND_FILE"
    ok "lanceur : $COMMAND_FILE"
  fi
fi

# ---- resume -----------------------------------------------------------------------------------------------------
step "Prophet Studio est installe"
cat <<EOF
    Lancer   : prophet-studio   ${C_DIM}(ou $LAUNCHER)${C_OFF}
               l'interface s'ouvre dans votre navigateur sur http://127.0.0.1:7878
    Ensuite  : l'assistant de l'interface detecte la carte graphique et telecharge
               Bonsai 2 27B (~6 Go) et le classifieur ; comptez ~10 Go libres au total.
    Decouvrir sans modele : prophet-studio --demo
    Mettre a jour : relancez ce script.
    Desinstaller  : supprimez $DATA_DIR
                    (modeles compris) et les lanceurs ci-dessus.
EOF

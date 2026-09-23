# Installer Prophet Studio

Trois facons d'installer, de la plus simple a la plus technique. Dans tous les cas, au premier lancement,
l'assistant de l'interface detecte la carte graphique puis telecharge **Bonsai 2 27B (~6 Go)** et le
classifieur ; rien ne quitte ensuite votre machine.

## Configuration requise

| | Minimum | Conseille |
|---|---|---|
| Systeme | Windows 10 ou 11 **x64** (cible principale) ; Linux x64 ; macOS 11+ (Apple Silicon) | Windows 11 |
| Carte graphique | aucune (mode processeur, lent) | NVIDIA 8 Go (RTX 4060, **RTX 5060**) |
| Pilote NVIDIA | **>= 570 pour une RTX 50xx** (Blackwell, CUDA 12.8) : [telecharger](https://www.nvidia.com/Download/index.aspx) | le plus recent |
| Disque | ~10 Go libres (modele ~6 Go, Python et dependances ~1,5 Go, runtime llama.cpp) | SSD |
| Memoire | 16 Go | 32 Go |
| Reseau | pour l'installation et le telechargement des modeles | |

Verifier le pilote : `nvidia-smi` (colonne "Driver Version"). Les scripts ci-dessous le verifient aussi et
previennent si une RTX 50xx a un pilote trop ancien.

## 1. Application de bureau (recommande)

Page [Releases](https://github.com/speed25200-cyber/LLM-and-Classifier/releases) du depot :

* **Windows** : `Prophet Studio_<version>_x64-setup.exe` (installation pour l'utilisateur courant, sans droits
  administrateur) ou le `.msi`. Les installateurs ne sont pas signes : SmartScreen affiche un avertissement,
  cliquer sur "Informations complementaires" puis "Executer quand meme".
* **Linux** : `.AppImage` (rendre executable puis lancer) ou `.deb` (`sudo apt install ./Prophet*.deb`).
* **macOS (Apple Silicon)** : `.dmg` ; application non notarisee : clic droit > Ouvrir au premier lancement.

Le premier lancement prepare Python et les dependances (1 a 3 minutes, progression affichee), puis l'assistant
prend le relais. Fermer la fenetre quitte Prophet Studio et libere la carte graphique.

## 2. En une commande (interface dans le navigateur)

Windows, dans PowerShell (pas besoin de droits administrateur) :

```powershell
irm https://raw.githubusercontent.com/speed25200-cyber/LLM-and-Classifier/main/installer/install.ps1 | iex
```

Linux et macOS :

```bash
curl -fsSL https://raw.githubusercontent.com/speed25200-cyber/LLM-and-Classifier/main/installer/install.sh | sh
```

Le script : verifie le systeme et le pilote NVIDIA, installe [uv](https://docs.astral.sh/uv/) s'il manque,
telecharge le code (branche `main` par defaut), cree l'environnement Python 3.11
(`uv sync --extra studio`), puis ajoute les lanceurs :

* Windows : raccourcis **"Prophet Studio"** dans le menu Demarrer et sur le Bureau ; ils lancent le moteur
  **sans fenetre de console** (`pythonw.exe`) et ouvrent l'interface dans le navigateur par defaut
  (`http://127.0.0.1:7878`) ;
* Linux : entree "Prophet Studio" dans le menu des applications et commande `prophet-studio`
  (`~/.local/bin`) ;
* macOS : `~/Applications/Prophet Studio.command` et commande `prophet-studio`.

Options :

```powershell
# Windows : parametres (version precise, sans raccourcis, depuis un depot clone...)
& ([scriptblock]::Create((irm https://raw.githubusercontent.com/speed25200-cyber/LLM-and-Classifier/main/installer/install.ps1))) -Ref v0.2.0 -NoShortcut
powershell -ExecutionPolicy Bypass -File .\installer\install.ps1 -Source .
```

```bash
curl -fsSL .../installer/install.sh | sh -s -- --ref v0.2.0 --no-shortcut
./installer/install.sh --source .          # depuis un depot clone
./installer/install.sh --help
```

Equivalents par variables d'environnement (utiles avec `irm | iex`) : `PROPHET_REF`, `PROPHET_SOURCE`,
`PROPHET_REPO`, `PROPHET_HOME`, `PROPHET_NO_SHORTCUT=1`.

**Mettre a jour** : relancer la meme commande (le code est remplace, l'environnement Python mis a jour,
les modeles et reglages conserves). Si le moteur tourne, le redemarrer ensuite.

## 3. Depuis les sources

```bash
git clone https://github.com/speed25200-cyber/LLM-and-Classifier.git
cd LLM-and-Classifier
uv sync --extra studio
uv run prophet-studio            # interface sur http://127.0.0.1:7878
uv run prophet-studio --demo     # sans modele, pour decouvrir l'interface
```

L'interface servie par le coeur est `prophet_studio/web` (copie de `ui/dist`, reconstruite par
`npm run build` dans `ui/`). Application de bureau en developpement : voir `desktop/README.md`.

## Ou sont les fichiers

Dossier de donnees : `%LOCALAPPDATA%\ProphetStudio` (Windows), `~/Library/Application Support/ProphetStudio`
(macOS), `~/.local/share/prophet-studio` (Linux, ou `$XDG_DATA_HOME/prophet-studio`) ; `PROPHET_HOME` prime.

| Sous-dossier | Contenu |
|---|---|
| `models/`, `bin/`, `voice/` | modeles GGUF, runtime llama.cpp, voix (le plus gros : ~6 a 8 Go) |
| `settings.json`, `sessions/`, `logs/` | reglages, conversations, journaux |
| `app/`, `app-venv/` | code et environnement Python installes par les scripts (methode 2) |
| `core/`, `venv/`, `python/`, `uv-cache/` | coeur, environnement, Python et cache prepares par l'application de bureau (methode 1) |
| `prophet-studio.pyw`, `prophet-studio.ico` | lanceur sans console et icone des raccourcis (Windows, methode 2) |

## Desinstaller

* **Application de bureau** : Windows : Parametres > Applications > Prophet Studio > Desinstaller ;
  Linux : `sudo apt remove prophet-studio` ou supprimer l'AppImage ; macOS : mettre l'application a la corbeille.
* **Scripts (methode 2)** : supprimer les raccourcis puis le dossier de donnees :
  * Windows : `Prophet Studio.lnk` dans le menu Demarrer (`%APPDATA%\Microsoft\Windows\Start Menu\Programs`)
    et sur le Bureau ;
  * Linux : `~/.local/share/applications/prophet-studio.desktop` et `~/.local/bin/prophet-studio` ;
  * macOS : `~/Applications/Prophet Studio.command` et `~/.local/bin/prophet-studio`.
* **Donnees et modeles** (toutes methodes) : supprimer le dossier de donnees ci-dessus. Attention, il contient
  les modeles telecharges et vos conversations.
* uv, s'il a ete installe par le script et ne sert plus : `uv cache clean`, puis supprimer `uv` / `uvx`
  de `~/.local/bin` (Windows : `%USERPROFILE%\.local\bin`).

## Depannage

* **RTX 50xx : le modele ne se charge pas / erreur CUDA** : pilote >= 570 obligatoire (voir plus haut).
* **Rien ne s'ouvre (methode 2, Windows)** : lire `%LOCALAPPDATA%\ProphetStudio\logs\launcher.log`.
* **Application de bureau bloquee sur "demarrage"** : le premier lancement installe Python (1 a 3 minutes) ;
  le journal s'affiche dans la fenetre et dans `<donnees>/logs/desktop-core.log`.
* **Interface web absente (methode 2)** : le code telecharge ne contenait pas `prophet_studio/web` ; installer
  Node.js 20+ et relancer le script (il construit alors l'interface), ou utiliser l'application de bureau.

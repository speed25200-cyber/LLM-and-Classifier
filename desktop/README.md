# Prophet Studio : application de bureau (Tauri 2)

Une coquille native legere autour du coeur Python (`prophet_studio/`) et de l'interface Svelte (`ui/`) :

* fenetre sans cadre 1440x900 (min. 960x640), fond `#07080B` des la creation (pas d'eclair blanc) ;
  l'interface dessine sa propre barre de titre (Windows, Linux) ; macOS garde ses feux tricolores en surimpression ;
* superviseur du coeur : lancement, jeton de session, attente de `PROPHET_READY` puis de `/api/health`,
  journal, redemarrage automatique (une fois) apres un plantage, arret propre de tout l'arbre de processus ;
* icone de barre d'etat (Afficher / Masquer, Redemarrer le moteur, Quitter), instance unique ;
* raccourci global push-to-talk `Ctrl+Maj+Espace` (`Cmd+Maj+Espace` sur macOS) ;
* economiseur de VRAM : sur une carte de 8 Go, chaque Mio compte pour Bonsai 2 27B, l'interface est
  donc rendue sans GPU quand `vram_saver` est vrai (valeur par defaut).

**Fermer la fenetre quitte l'application** et arrete le moteur, donc les deux `llama-server` : aucune VRAM ne
reste occupee par un processus invisible. Pour garder le moteur en arriere-plan, utiliser
"Afficher / Masquer" dans la barre d'etat.

## Developpement

Prerequis : Rust stable, Node.js 20+, [uv](https://docs.astral.sh/uv/). Linux (Debian / Ubuntu) :

```bash
sudo apt install libwebkit2gtk-4.1-dev libjavascriptcoregtk-4.1-dev libsoup-3.0-dev libgtk-3-dev \
  libayatana-appindicator3-dev librsvg2-dev libxdo-dev libssl-dev build-essential pkg-config xdg-utils file
```

```bash
npm ci --prefix ui            # dependances de l'interface
cd desktop
npm ci                        # CLI Tauri
npm run sidecar               # telecharge uv -> src-tauri/binaries/prophet-uv-<cible> (voir plus bas)
npm run dev                   # Vite (localhost:5173) + application ; le coeur est lance via uv depuis le depot
npm run build                 # paquets dans src-tauri/target/release/bundle/
```

`npm run sidecar -- --placeholder` cree un fichier vide a la place du vrai binaire : suffisant pour compiler,
la coquille l'ignore alors et utilise le `uv` du PATH avec le depot source.

Variables utiles :

| Variable | Effet |
|---|---|
| `PROPHET_CORE_CMD` | commande du coeur (decoupee sur les espaces), lancee depuis la racine du depot, ex. `.venv/bin/python -m prophet_studio --demo` |
| `PROPHET_CORE_ARGS` | arguments ajoutes, ex. `--demo` (faux modeles, pour decouvrir l'interface) |
| `PROPHET_HOME` | dossier de donnees (sinon celui du coeur, voir plus bas) |

La coquille ajoute toujours `--no-browser --port 0 --token <jeton> --parent-pid <pid>` (plus `--dev` dans une
compilation de developpement, pour que le coeur accepte l'origine `http://localhost:5173`).

## Quel moteur est lance ?

1. `PROPHET_CORE_CMD` s'il est defini ;
2. sinon le **uv embarque** (`prophet-uv`, a cote de l'executable) avec le **coeur embarque** (ressources
   `core/` = `pyproject.toml`, `README.md`, `jev_clone/`, `prophet_studio/`) :
   `prophet-uv run --project <donnees>/core --extra studio --python 3.11 python -m prophet_studio ...`
   avec `UV_PROJECT_ENVIRONMENT=<donnees>/venv`, `UV_PYTHON_INSTALL_DIR=<donnees>/python`,
   `UV_CACHE_DIR=<donnees>/uv-cache`, `UV_PYTHON_PREFERENCE=only-managed` ;
3. sinon `uv` du PATH (ou `~/.local/bin`, `~/.cargo/bin`, `/opt/homebrew/bin`) avec le depot source, trouve en
   remontant depuis l'executable ou le dossier courant jusqu'au `pyproject.toml` qui mentionne `prophet_studio` ;
4. sinon `prophet-studio` du PATH, ou celui de `<donnees>/app-venv` cree par les scripts de `installer/`.

Dans une compilation de developpement (`npm run dev`, `cargo build`), l'etape 3 passe avant la 2 quand le depot
source est trouve (avec le uv embarque si aucun uv n'est dans le PATH) : le code Python modifie est pris en compte
au prochain lancement du moteur, sans recompiler.

Le coeur embarque est **recopie** dans `<donnees>/core` (seulement si son contenu a change, empreinte dans
`.bundle-stamp`) : le dossier des ressources peut etre en lecture seule (AppImage, Program Files, .app signee)
alors que uv y ecrit `uv.lock` et `*.egg-info`. Au premier lancement, uv telecharge Python 3.11 et les
dependances (1 a 3 minutes) ; la progression apparait dans le journal (`core://log`).

Le sidecar s'appelle `prophet-uv` et non `uv` : le paquet `.deb` installe les sidecars dans `/usr/bin`, ou un
`uv` entrerait en conflit avec celui de la distribution.

Dossier de donnees (identique a `prophet_studio.config.data_dir()`) : `%LOCALAPPDATA%\ProphetStudio` (Windows),
`~/Library/Application Support/ProphetStudio` (macOS), `$XDG_DATA_HOME/prophet-studio` ou
`~/.local/share/prophet-studio` (Linux) ; `PROPHET_HOME` prime. Journal de la coquille :
`<donnees>/logs/desktop-core.log` (sorties du coeur, recree a chaque lancement).

## Contrat avec l'interface

`app.withGlobalTauri = true` : l'interface utilise `window.__TAURI__` directement (pas de dependance npm).

| Nom | Type | Charge utile |
|---|---|---|
| `core_info` | commande | `{ url: string \| null, token: string, status: "starting" \| "ready" \| "error", error: string \| null, logs: string[] }` |
| `restart_core` | commande | aucune ; rend la main tout de suite, la suite arrive par `core://status` |
| `core://status` | evenement | meme objet que `core_info`, a chaque changement d'etat |
| `core://log` | evenement | `{ line: string }` pour chaque ligne de stdout / stderr du coeur (et `[bureau] ...` de la coquille) |
| `shortcut://ptt` | evenement (vers la fenetre `main`) | `{ state: "pressed" \| "released" }` ; a l'appui, la fenetre est aussi affichee et prend le focus |

* `status: "ready"` n'est emis qu'une fois `/api/health` joignable (le coeur imprime `PROPHET_READY` juste
  avant d'ouvrir son port). Appeler `core_info` au demarrage puis ecouter `core://status` : l'evenement peut
  partir avant que l'interface ne soit chargee.
* Le **jeton** ne change pas pendant la vie de l'application ; l'**url** (port libre) change a chaque
  (re)demarrage du moteur : la reprendre a chaque `ready`. `logs` contient les 200 dernieres lignes.
* Requetes : `fetch(url + "/api/...", { headers: { "X-Prophet-Token": token } })` ; WebSocket :
  `ws://127.0.0.1:<port>/...?token=<jeton>`. Le coeur accepte les origines `tauri://localhost`,
  `http(s)://tauri.localhost` (et `http://localhost:5173` en developpement).
* Barre de titre : `data-tauri-drag-region` sur la zone de glisser (double-clic = agrandir), et
  `window.__TAURI__.window.getCurrentWindow().minimize()` / `.toggleMaximize()` / `.close()`.
* Liens externes : `window.__TAURI__.opener.openUrl(url)` ; de plus, toute navigation de la fenetre vers
  une adresse externe est bloquee et ouverte dans le navigateur du systeme.
* CSP (tauri.conf.json) : scripts, polices et feuilles de style locaux uniquement (`'unsafe-inline'` permis
  pour les styles), connexions vers `127.0.0.1` / `localhost` (http et ws), images et medias `data:` / `blob:`.
  Pas de CDN ni de police distante : tout doit etre embarque par Vite.

Permissions de la fenetre (`capabilities/default.json`) : `core:default`, reduire, agrandir / restaurer,
fermer, glisser, afficher, focus, et `opener:default`.

## Arret et plantages

* Windows : le coeur est place dans un Job Object "kill on close" : si l'application se ferme (ou meurt),
  Windows termine le coeur et les `llama-server`. Lancement avec `CREATE_NO_WINDOW` (aucune console).
* Linux / macOS : le coeur a son propre groupe de processus ; arret = SIGTERM (uv le relaie, uvicorn s'arrete
  proprement et stoppe les `llama-server`), puis SIGKILL du groupe apres 8 s. Ctrl+C / SIGTERM sur
  l'application passent par le meme chemin.
* Partout, `--parent-pid` fait sortir le coeur en ~2 s si l'application disparait sans avoir pu faire le menage.
* Plantage du coeur apres `ready` : un redemarrage automatique, puis `status: "error"` (un redemarrage manuel
  remet le compteur a zero).

## Economiseur de VRAM

Lu au demarrage dans `settings.json` (`vram_saver`, vrai si absent) : **redemarrer l'application** apres l'avoir change.

* Windows : WebView2 recoit `--disable-features=msWebOOUI,msPdfOOUI,msSmartScreenProtection` (les options par
  defaut de wry, conservees) `--disable-gpu --disable-gpu-compositing`.
* Linux : `WEBKIT_DISABLE_COMPOSITING_MODE=1`. Independamment, avec le pilote NVIDIA proprietaire,
  `WEBKIT_DISABLE_DMABUF_RENDERER=1` (evite une fenetre blanche). Une valeur deja definie est respectee.
* macOS : pas d'equivalent pour WKWebView.

## Particularites par systeme

* **Linux, micro** : WebKitGTK refuse `getUserMedia` par defaut ; la coquille active `enable-media-stream` et
  n'accorde que les demandes **micro seul** (pas la camera). Il faut les greffons GStreamer
  (`gstreamer1.0-plugins-good`, `gstreamer1.0-pulseaudio` ou `gstreamer1.0-pipewire`, recommandes par le .deb).
* **Linux, raccourci global** : X11 (et XWayland) uniquement ; sous Wayland pur, l'enregistrement echoue sans
  bloquer l'application (message `[bureau] raccourci ... indisponible` dans le journal).
* **Linux, AppImage** : AppRun fait pointer `PYTHONHOME`, `PYTHONPATH`, `LD_LIBRARY_PATH`, `PATH`... vers
  l'image ; ces entrees sont retirees de l'environnement du coeur (sinon Python et llama-server cassent).
* **macOS** : signature ad hoc (`signingIdentity: "-"`), pas de notarisation : clic droit > Ouvrir au premier
  lancement. `Info.plist` (usage du micro) et `Entitlements.plist` (`audio-input`, runtime renforce) sont fournis.
* **Windows** : WebView2 gere lui-meme l'autorisation du micro. Installateurs non signes : SmartScreen
  affichera un avertissement ("Informations complementaires" > "Executer quand meme").

## Icones

`src-tauri/icons/icon.svg` (deux arcs concentriques : System One au centre, System Two autour) et
`tray-template.svg` (barre des menus macOS) sont les sources ; `npm run icons` regenere toutes les tailles
(`icon-1024.png`, `.ico`, `.icns`, PNG) avec le CLI Tauri.

## Fichiers

| Chemin | Role |
|---|---|
| `src-tauri/src/lib.rs` | construction de l'application : greffons, commandes, barre d'etat, raccourci, sortie |
| `src-tauri/src/core.rs` | superviseur du coeur (etats, journal, evenements, sante, redemarrage) |
| `src-tauri/src/launch.rs` | dossier de donnees, `vram_saver`, resolution de la commande, copie du coeur embarque |
| `src-tauri/src/process.rs` | arbre de processus : Job Object (Windows), groupe + signaux (Unix) |
| `src-tauri/src/window.rs` | fenetre principale, navigation, WebView par systeme, micro sous Linux |
| `scripts/fetch-uv.mjs` | telecharge et verifie (SHA-256) uv pour une cible, ou cree l'espace reserve |
| `scripts/render-icons.mjs` | regenere les icones |

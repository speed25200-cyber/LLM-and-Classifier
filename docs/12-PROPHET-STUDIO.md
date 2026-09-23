# 12 - Prophet Studio : la plateforme (application, installation, voix, computer use)

Prophet Studio met la fusion Jev-clone (System One) x Bonsai 2 27B (System Two) dans une application locale : un
coeur Python qui installe, planifie et supervise tout, une interface facon Claude Code (Svelte 5, ~350 Ko de JS,
polices embarquees, aucun CDN), une application de bureau Tauri 2 et des commandes vocales. Cible : **RTX 5060
8 Go** sous Windows ; tout marche aussi sur 4060, 5060 Ti 16 Go, CPU seul, Linux et macOS.

![Configuration optimale choisie pour une RTX 5060](img/studio-config.jpg)

## 1. Installer

| Methode | Pour qui | Commande |
|---|---|---|
| Application de bureau | Windows, le plus simple | installeur `Prophet Studio_x.y.z_x64-setup.exe` des Releases (construit par `.github/workflows/desktop.yml`) |
| Une ligne PowerShell | Windows, sans installeur | `irm https://raw.githubusercontent.com/speed25200-cyber/LLM-and-Classifier/main/installer/install.ps1 \| iex` |
| Une ligne shell | Linux / macOS | `curl -fsSL https://raw.githubusercontent.com/speed25200-cyber/LLM-and-Classifier/main/installer/install.sh \| sh` |
| Depuis les sources | developpeurs | `uv sync --extra studio` puis `uv run prophet-studio` |

Details, prerequis (pilote NVIDIA >= 570 pour les RTX 50xx) et desinstallation : [`installer/README.md`](../installer/README.md).

Au premier lancement, l'**assistant** : (1) presente les deux cerveaux, (2) verifie la machine (GPU, VRAM, pilote /
CUDA, RAM, disque), (3) montre la configuration optimale et sa barre VRAM, (4) installe en un clic le runtime
llama.cpp adapte au GPU, Bonsai 2 27B, le classifieur et les voix (~7 Go, reprenable, verifie en SHA-256),
(5) demarre les modeles. Sans GPU ni modele : `prophet-studio --demo` fait tout tourner sur de faux serveurs
pour decouvrir l'interface.

## 2. Architecture

```
 Application de bureau (Tauri 2, ~10 Mo)          Navigateur (mode sans installeur)
   fenetre sans cadre, barre systeme,                  http://127.0.0.1:7878
   raccourci global Ctrl+Maj+Espace (parler)                |
          |  lance et surveille le coeur (uv)               |
          v                                                 v
 +------------------------ coeur prophet_studio (FastAPI, 127.0.0.1, jeton) --------------------------+
 | hardware   : NVML / nvidia-smi (RTX 5060 = Blackwell sm_120, 448 Go/s)                              |
 | planner    : budget VRAM -> modele, contexte, place du classifieur ; echelle anti-OOM               |
 | installer  : runtime llama.cpp (CUDA 12.8+ pour Blackwell, cudart Windows), GGUF HF, voix sherpa     |
 | supervisor : 2 llama-server (S2 Bonsai, S1 classifieur), journaux, OOM -> cran suivant, mode mono    |
 | sessions   : Prophet en flux (SSE -> WebSocket), autorisations, plan, effort, transcription rejouable |
 | voice      : Parakeet / Whisper, Silero VAD, Piper / Kokoro (CPU) ; commandes jugees par S1           |
 | /v1/chat/completions (proxy Bonsai) · /v1/systemone (format TypeSafe) pour vos autres outils          |
 +-----------------------------------------------------------------------------------------------------+
          |                                   |
   llama-server :7880                  llama-server :7881
   Bonsai 2 27B PTQ1_0 (GPU)           Ternary-Bonsai 1.7B (CPU ou GPU)
```

Code : `prophet_studio/` (coeur), `ui/` (interface, construite dans `prophet_studio/web/`), `desktop/` (Tauri),
`installer/`, `jev_clone/` (le clone, l'agent Prophet, le computer use).

## 3. RTX 5060 : ce que decide le planificateur (`prophet_studio/planner.py`)

Budget = VRAM totale - ce qu'occupent deja le bureau et les autres applications - 256 Mio de marge. Bonsai doit
tenir **entierement** sur le GPU ; on maximise le contexte ; le classifieur va sur le GPU s'il reste de la place,
sinon sur le CPU. Sorties reelles du planificateur (debits = bande passante / taille des poids x efficacite
observee, a confirmer avec le bouton **Mesurer**) :

| Carte | Priorite | System Two | Contexte (KV) | System One | Generation estimee |
|---|---|---|---|---|---|
| RTX 5060 8 Go, ecran branche dessus | equilibre | Bonsai 2 27B PTQ1_0 | 24 k (q4_0) | ternary-1.7b sur CPU | 47-59 tok/s |
| RTX 5060 8 Go, ecran branche dessus | vitesse | Bonsai 27B Q1_0 | 16 k (q4_0) | ternary-1.7b sur GPU | 53-65 tok/s |
| RTX 5060 8 Go, ecran sur l'iGPU | equilibre | Bonsai 2 27B PTQ1_0 | 32 k (q4_0) | ternary-1.7b sur CPU | 47-59 tok/s |
| RTX 5060 8 Go, ecran sur l'iGPU | contexte | Bonsai 2 27B PTQ1_0 | 48 k (q4_0) | ternary-1.7b sur CPU | 47-59 tok/s |
| RTX 5060 Ti 16 Go | equilibre | Bonsai 2 27B PQ2_0 | 32 k (q8_0) | ternary-4b sur GPU | 38-48 tok/s |
| RTX 5060 Ti 16 Go | contexte | Bonsai 2 27B PQ2_0 | 128 k (q8_0) | ternary-4b sur GPU | 38-48 tok/s |
| RTX 4060 8 Go | equilibre | Bonsai 2 27B PTQ1_0 | 32 k (q4_0) | ternary-1.7b sur CPU | 28-36 tok/s |

Pourquoi le classifieur passe sur CPU a 8 Go : Bonsai 2 PTQ1_0 prend 5,52 Gio de poids + ~1 Gio de tampons ; le
classifieur 1.7B demanderait ~1,1 Gio de plus (poids, tampons, KV 8 k en q8_0), soit presque tout le contexte.
Sur CPU, un 1.7B ternaire lit ~3 000 tok/s en prefill : les decisions passent de ~0,05-0,15 s a ~0,1-0,3 s, et
Bonsai garde 24 a 48 k de contexte, ce qui compte plus pour un agent de code.

Specificites Blackwell :
* **runtime** : la build CUDA 12.8 (ou 13.x) du fork PrismML est choisie automatiquement (`installer.py`,
  `select_runtime_assets`) ; sous Windows le zip `cudart` de la meme version est ajoute. Pilote >= 570 requis ;
  l'assistant le signale sinon.
* **ecran** : si l'ecran est branche sur la 5060, Windows y reserve 0,3-1 Gio ; le brancher sur la carte mere
  (iGPU) donne +8 k de contexte. Detecte par NVML (`display_active`).
* **interface sans VRAM** : l'application de bureau desactive le rendu GPU de WebView2 (reglage "Economie de
  VRAM", actif par defaut) : l'interface reste fluide (animations limitees a transform/opacity, pause quand la
  fenetre est cachee) sans prendre un Mio a Bonsai.
* **echelle anti-OOM** (`planner.degrade`) : si llama-server echoue en "out of memory" au chargement, le
  superviseur descend d'un cran et relance seul : vision en RAM -> classifieur sur CPU -> contexte reduit -> KV
  q4_0 -> 4 k -> Bonsai 27B 1-bit -> dechargement partiel. L'interface affiche "Configuration ajustee".
* **mode mono** : si le classifieur n'est pas installe ou ne demarre pas, Bonsai repond aussi aux questions
  System One (lecture par grammaire) : tout fonctionne, decisions plus lentes.

Profils equivalents pour les scripts shell : `scripts/profiles/rtx5060-8gb-equilibre.env`,
`rtx5060-8gb-vitesse.env`, `rtx5060ti-16gb.env`.

## 4. L'agent dans l'interface

![Un tour d'agent : jugements du classifieur, reflexion, outils](img/studio-agent.jpg)

Chaque tour montre ce que la fusion fait :
1. **Bandeau System One** (~50-150 ms) : reponse directe ou agent, besoin de raisonnement, risque, budget de
   reflexion accorde a Bonsai ; deplie, les jugements calibres et la pertinence de chaque outil.
2. **Reflexion** de Bonsai en direct (repliee ensuite, avec sa duree).
3. **Outils** en cartes : diff des fichiers ecrits / modifies (`edit_file` par remplacement exact, economique en
   contexte), sortie de terminal, fichiers trouves, jugements `judge_*` en barres de probabilite.
4. **Autorisations** : en mode *Smart* (defaut), le classifieur juge chaque commande (~100 ms) et ne demande que
   si elle est risquee ; *Toujours demander* confirme chaque ecriture avec apercu du diff ; *Auto* ne demande
   jamais. Reponse au clavier (Y / A / N) ou a la voix (« accepte », « refuse »).
5. **Verification finale** par le classifieur, debit (tok/s), tokens, duree.

![Autorisation avec apercu](img/studio-permission.jpg)

| Commande | Effet | Raccourci |
|---|---|---|
| `/nouveau` | nouvelle session | Ctrl+N |
| `/plan` | mode plan : lecture seule, Bonsai rend un plan | Maj+Tab |
| `/profond`, `/rapide` | effort de reflexion du prochain message | |
| `/smart`, `/demander`, `/auto` | politique d'autorisation | |
| `/fichiers` | inspecteur de l'espace de travail | Ctrl+B |
| `/modeles`, `/reglages`, `/bench` | ecrans et mesure | Ctrl+, |
| `/lire`, `/mainslibres` | voix | Ctrl+Maj+Espace (parler) |
| palette | toutes les commandes et sessions | Ctrl+K |
| `@chemin` | mentionner un fichier (autocompletion) | |
| Echap | arreter la generation / la voix | |

## 5. Voix

* **Push-to-talk** : maintenir le bouton micro ou Ctrl+Maj+Espace (global dans l'application de bureau) ; le son
  part en PCM 16 kHz au coeur, qui transcrit (Parakeet TDT 0.6B v3, 25 langues dont le francais, ou Whisper).
* **Mains libres** : flux continu vers le detecteur de parole (Silero) du coeur ; seuls les enonces commencant
  par le mot d'eveil (« Prophet, … », « OK Prophet », « Dis Prophet ») sont pris en compte.
* **Synthese** : Piper (voix francaise Siwis, anglaise Lessac) ou Kokoro ; lecture phrase par phrase ; sans
  modele installe, la voix du systeme prend le relais (Web Speech).
* Tout tourne sur **CPU** : la VRAM reste a Bonsai.

Routage d'un enonce (`prophet_studio/voice.py`) : (1) mot d'eveil retire ; (2) grammaire exacte FR/EN ->
commande immediate ; (3) enonce court et ambigu -> **le classifieur tranche en une passe** (choix calibre parmi
les commandes + « demande pour l'agent ») ; sous le seuil (72 % par defaut), c'est une demande : on ne declenche
jamais une commande par erreur ; (4) sinon, la phrase part a l'agent.

| Commande | Exemples |
|---|---|
| new_session | « nouvelle session », « new chat » |
| stop | « stop », « arrete », « annule », « tais-toi » |
| approve / deny | « accepte », « vas-y », « refuse », « non » |
| send, clear_input | « envoie », « efface » |
| read_last | « lis la reponse », « relis » |
| plan_on / plan_off | « mode plan », « quitte le mode plan » |
| effort_deep / effort_fast | « reflechis bien », « reponds vite » |
| open_settings / open_models | « ouvre les reglages », « ouvre les modeles » |
| mute | « coupe le micro » |

## 6. Computer use

* **Navigateur** (`jev_clone/computer_use.py`, reglage "Outil navigateur") : Playwright, arbre ARIA.
* **Bureau** (`jev_clone/desktop_use.py`, reglage "Controle du bureau", Windows) : l'arbre **UI Automation** de
  la fenetre active joue le role de l'arbre ARIA. A chaque pas, le classifieur choisit l'action et le controle en
  une passe (~0,1 s, sans lire de pixels) ; en cas de doute, Bonsai reprend la main avec `click`, `type`,
  `press_keys`, `open_app` (et la vision de Bonsai 2 si activee). Chaque lancement est juge par le classifieur
  avant execution. Les trajectoires servent a re-entrainer la politique rapide (DAgger, `trajectory_to_examples`).

## 7. API locale et securite

Le coeur peut executer des commandes : il n'ecoute que sur 127.0.0.1 et exige un **jeton** tire au lancement
(en-tete `X-Prophet-Token` ou `Authorization: Bearer`), un en-tete `Host` local (parade au DNS rebinding) et une
`Origin` absente ou autorisee (l'interface elle-meme, l'application Tauri). Le jeton est ecrit dans
`core.json` (droits 600) et injecte dans la page servie.

Pour brancher d'autres outils : `POST /v1/chat/completions` (Bonsai, compatible OpenAI, diffusion comprise) et
`POST /v1/systemone` (decisions typees au format TypeSafe) avec `Authorization: Bearer <jeton>`.

## 8. Developper

```bash
uv sync --extra studio --extra dev
uv run prophet-studio --demo                      # faux serveurs, interface complete
cd ui && npm install && npm run dev               # interface en rechargement a chaud (coeur lance avec --dev --token dev)
cd ui && npm run check && npm run build           # verifie les types et reconstruit prophet_studio/web
uv run pytest -q                                  # coeur, agent, voix, bureau, API : sans GPU (faux llama-server)
cd desktop && npm ci && npm run sidecar && npm run dev   # application de bureau (voir desktop/README.md)
```

`uv.lock` fige les versions : les installateurs et l'application de bureau l'embarquent (installations
reproductibles). La CI (`.github/workflows/ci.yml`) lance les tests Python sous Linux et Windows, verifie et
construit l'interface, et passe `cargo fmt` / `clippy` / `test` sur la coquille ; `desktop.yml` construit les
paquets (NSIS, MSI, deb, AppImage, dmg) sur une etiquette `v*`.

## 9. Ce qui est verifie, ce qui ne l'est pas

Verifie ici (sans GPU) :
* le coeur de bout en bout contre un faux llama-server lance par le vrai superviseur : plan RTX 5060, demarrage,
  echelle anti-OOM (OOM simule au-dela de 8 k -> relance a 8 k), tours d'agent en flux, autorisations,
  annulation, mode plan, securite (jeton, Host, Origin), WebSocket, voix (bibliotheque sherpa_onnx factice de meme
  API), bureau simule, proxies /v1 ;
* l'interface pilotee par Chromium (Playwright) : assistant, tour d'agent, inspecteur, palette, autorisation,
  modeles, mesure, reglages, theme clair, fenetre etroite, sans erreur de console ;
* la voix dans un vrai navigateur : micro factice de Chromium -> AudioWorklet 16 kHz -> envoi PCM -> transcription
  (moteur factice) -> routage -> commande « nouvelle session » executee par l'interface (`tests/test_voice_e2e.py`) ;
* la diffusion token par token (un correctif : `iter_lines` lisait par paquets de 512 octets) ;
* le classifieur en panne : le tour se termine avec Bonsai seul, actions risquees confirmees ;
* l'application de bureau sous Linux (Xvfb) : AppImage et .deb construits, fenetre 1440x900 connectee au coeur
  (CSP et IPC), jeton et origine `tauri://localhost`, redemarrage automatique apres un plantage, fermeture en
  moins d'une seconde avec arret du coeur et des llama-server, `--parent-pid`, raccourci global ; uv embarque qui
  prepare Python 3.11 et les dependances ; `cargo clippy` propre aussi pour les cibles Windows et macOS ;
* `install.sh` (installation reelle, mise a jour en place) et `install.ps1` (analyse et chaine zip -> uv sync ->
  lanceur executees sous PowerShell 7) ;
* 102 tests (`pytest`) : 98 tournent sans GPU (dont deux pilotes par Chromium : tour d'agent, voix), 4 attendent un vrai llama-server.

Non verifie ici (pas de GPU ni d'acces a Hugging Face / aux releases GitHub depuis l'environnement de redaction) :
* les debits reels sur RTX 5060 (estimations ; bouton **Mesurer** et `eval/SCOREBOARD.md` pour les remplacer) ;
* les noms exacts des assets de la release PrismML pour Windows (le choix se fait sur la liste reelle de la
  release, avec des noms construits en repli) et des archives vocales sherpa-onnx ;
* la reconnaissance et la synthese vocales avec les vrais modeles, UI Automation sur un vrai Windows.
* l'application de bureau sous Windows et macOS (WebView2, Job Object, installateurs NSIS / MSI / dmg, gain de
  VRAM de `--disable-gpu`), le micro de WebKitGTK sous Linux.

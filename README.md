# LLM-and-Classifier : Prophet Studio, fusion Jev-clone (System One) x Bonsai 2 27B (System Two) sur RTX 5060

Un **classificateur/decideur** ultra-rapide (clone ouvert de [Jev](docs/01-JEV-typesafe-analyse.md), le
"System One model" de TypeSafe AI) fusionne avec un **LLM** de 27 milliards de parametres qui tient dans
6 Go ([Bonsai 2 27B](docs/02-BONSAI-2-27B-analyse.md), PrismML, base Qwen3.8-27B), le tout sur **une RTX
5060 8 Go** (ou 4060, 5060 Ti 16 Go, CPU seul), dans une application locale facon Claude Code avec commandes vocales.

## Prophet Studio : l'application

![Prophet Studio : un tour d'agent](docs/img/studio-agent.jpg)

```powershell
# Windows (RTX 5060 : pilote NVIDIA >= 570)
irm https://raw.githubusercontent.com/speed25200-cyber/LLM-and-Classifier/main/installer/install.ps1 | iex
```
```bash
# Linux / macOS
curl -fsSL https://raw.githubusercontent.com/speed25200-cyber/LLM-and-Classifier/main/installer/install.sh | sh
# ou depuis les sources :  uv sync --extra studio && uv run prophet-studio      (sans modele : --demo)
```
L'assistant de premier lancement detecte la carte, calcule la configuration optimale (RTX 5060 : Bonsai 2 27B
PTQ1_0 entier sur GPU, 24-32 k de contexte, classifieur sur CPU, ~47-59 tok/s estimes), installe en un clic le
runtime llama.cpp adapte (CUDA 12.8+ pour Blackwell), les modeles et les voix, puis demarre tout.

* **Agent facon Claude Code** : flux token par token, reflexion repliable, cartes d'outils (diffs, terminal,
  fichiers), `edit_file` / `grep` / `glob`, mode plan, effort de reflexion, commandes `/`, mentions `@fichier`,
  palette Ctrl+K, inspecteur de fichiers, sessions rejouables.
* **La fusion visible a chaque tour** : le classifieur decide en ~50-150 ms (reponse directe ou agent, budget de
  reflexion, risque, outils pertinents), juge chaque commande avant execution (autorisations *Smart*) et verifie
  la reponse finale.
* **Voix** : push-to-talk (Ctrl+Maj+Espace) ou mains libres avec mot d'eveil, commandes vocales FR/EN tranchees
  par le classifieur, lecture des reponses ; tout sur CPU, la VRAM reste au modele.
* **Computer use** : navigateur (arbre ARIA) et bureau Windows (UI Automation), le classifieur decide chaque pas.
* **Sobre** : ~350 Ko d'interface, aucune ressource distante, rendu CPU par defaut dans l'application de bureau,
  animations en pause fenetre cachee ; planificateur VRAM et echelle anti-OOM automatiques.
* **Ouvert** : API locale compatible OpenAI (`/v1/chat/completions`) et TypeSafe (`/v1/systemone`), protegee par jeton.

Documentation complete : **[docs/12-PROPHET-STUDIO.md](docs/12-PROPHET-STUDIO.md)** (installation, architecture,
choix pour la RTX 5060, voix, computer use, securite, ce qui est verifie).

## Le protocole de fusion

**Lire d'abord : [docs/00-PROTOCOLE-FUSION-JEV-BONSAI.md](docs/00-PROTOCOLE-FUSION-JEV-BONSAI.md)** (le
protocole complet, en 9 phases, avec criteres d'acceptation, risques et commandes).

## Idee en trois lignes
1. Le clone de Jev lit, en une passe et sans generer de texte, une **distribution calibree** sur les options
   de chaque question typee (`choice` / `score` / `noul`) : ~50-150 ms.
2. Si la confiance depasse un seuil calibre, le code agit ; sinon **Bonsai 2 27B** raisonne (budget de
   reflexion selon le risque), genere, appelle des outils, lit des images.
3. Dans l'autre sens, Bonsai **consulte le clone comme un outil** (`judge_*`) pendant qu'il planifie ou pilote
   le navigateur : agent "computer use" a deux vitesses, le clone decide chaque pas, Bonsai reprend la main
   en cas de doute (`docs/06-AGENT-COMPUTER-USE.md`).
4. Tout est journalise ; Bonsai etiquette les cas douteux ; le clone est recalibre / re-entraine.

## Demarrage rapide en ligne de commande (RTX 4060 / 5060)
```bash
PROFILE=scripts/profiles/rtx4060-8gb-qualite.env ./scripts/setup.sh     # binaires PrismML + poids + venv
# RTX 5060 : PROFILE=scripts/profiles/rtx5060-8gb-equilibre.env (ou rtx5060-8gb-vitesse, rtx5060ti-16gb)
./scripts/start_bonsai.sh        # terminal 1 : Bonsai 2 27B PTQ1_0 sur :8080
./scripts/start_jev_clone.sh     # terminal 2 : Ternary-Bonsai-1.7B (clone niveau 0) sur :8081
source .venv/bin/activate
jev decide --server http://127.0.0.1:8081 --state "Charged twice for order A-104, please refund." \
    --choice team=billing,technical,sales --noul "refund=Is a refund requested?"
export JEV_S1_URL=http://127.0.0.1:8081 JEV_S2_URL=http://127.0.0.1:8080
jev serve --port 8008            # POST /v1/systemone (format TypeSafe) et POST /v1/fusion/decide
```
Sans GPU : `PROFILE=scripts/profiles/minimal-cpu.env` (Bonsai-8B 1-bit + clone 1.7B, 8 Go de RAM).
Avec une **A100 80 Go sur Colab** : ouvrir `colab/jev_bonsai_a100.ipynb` pour distiller Bonsai et entrainer
le clone en pleine precision, puis rapatrier le GGUF et la calibration sur la 4060 (voir `docs/05-COLAB-A100.md`).

## Contenu du depot
| Chemin | Contenu |
|---|---|
| `prophet_studio/`, `ui/`, `desktop/`, `installer/` | **Prophet Studio** : coeur (materiel, planificateur VRAM, installateur, superviseur, sessions, voix, API), interface Svelte, application Tauri, installeurs en une ligne ([docs/12](docs/12-PROPHET-STUDIO.md)) |
| `jev_clone/desktop_use.py` | computer use sur le bureau (Windows UI Automation) a deux vitesses |
| `docs/00-PROTOCOLE-FUSION-JEV-BONSAI.md` | **le protocole complet** |
| `docs/01-JEV-typesafe-analyse.md` | comment Jev fonctionne (contrat, mecanisme deduit, RLCD, benchmarks, clones ouverts) |
| `docs/02-BONSAI-2-27B-analyse.md` | Bonsai 2 27B : architecture, formats, qualite, vitesse, memoire, runtime |
| `docs/03-MATERIEL-profils.md` | RTX 4060 et exigences minimales, matrice de profils, feuille VRAM, depannage OOM |
| `docs/04-REFERENCES.md` | sources |
| `jev_clone/` | le clone : `schema` (contrat), `prompt` (prefixe + branches), `backend_llamacpp` (lecture par grammaire), `readout` (temperature, confiance), `engine`, `calibrate`, `distill`, `fusion` (routeur), `tools` (le clone comme outils de Bonsai + boucle d'agent), `computer_use` (agent navigateur a deux vitesses), `server`, `cli` |
| `scripts/` | `setup.sh`, `start_bonsai.sh` (`BONSAI_GGUF`, `BONSAI_LORA`), `start_jev_clone.sh`, `bench.sh`, `fetch_orcabonsai.sh`, `profiles/*.env`, `windows/*.ps1` |
| `training/` | `train_lora_rlcd.py` (LoRA / QLoRA / `--full` + regle de score propre + distillation), `make_public_mix.py` (jeux publics -> JSONL) et sa notice |
| `colab/` | notebook A100 80 Go : installation, Bonsai enseignant, distillation, entrainement complet, export GGUF, calibration, copie vers Drive |
| `docs/05-COLAB-A100.md` | usine A100 / production 4060 : quoi faire ou, durees, artefacts |
| `examples/` | routage de tickets avec escalade ; boucle d'agent temps reel ; agent navigateur (`browser_agent.py`) |
| `docs/06-AGENT-COMPUTER-USE.md` | fusion bidirectionnelle, computer use, budget de latence, ecart honnete avec Jev |
| `docs/11-PROPHET.md`, `jev_clone/prophet.py` | **Prophet** : agent general local a deux vitesses (REPL `jev prophet`) : code, applications, shell, web, memoire, outils qu'il cree lui-meme ; le clone juge des proprietes (reponse directe, budget, risque, outils pertinents), jamais des cases ; `training/seeds/` + `make_synthetic_prophet.py` : le jeu de donnees fabrique a partir de zero |
| `docs/10-CAS-D-USAGE.md`, `jev_clone/presets.py` | routage, choix de competence, reranking, garde-fous, extraction typee : schemas prets a l'emploi ; `ledger_report.py` (economies mesurees) ; `training/make_from_history.py` (votre historique -> entrainement) |
| `docs/09-FUSION-PROFONDE.md` | les trois niveaux de fusion (systeme, inference, modele), ce qui est fait, ce que "SOTA a 8 Go" veut dire, RTX 5060 |
| `jev_clone/guided.py` | System One pilote la generation de Bonsai : reflexion adaptative, meilleur de N, reponse verifiee |
| `docs/08-ORCABONSAI-UNCENSORED.md` | variante Bonsai 2 sans refus (adaptateur LoRA de rang 1 a l'execution) : fonctionnement, mesures des auteurs et reserves, integration, ou passe le garde-fou |
| `docs/07-BATTRE-JEV.md`, `eval/SCOREBOARD.md` | cibles chiffrees contre les chiffres publics de Jev, sept axes d'avance (dont cinq inedits), plan A100 en 4 semaines |
| `eval/` | banc de mesure : les 60 cas et les sorties reelles de Jev (jev-benchmark) mesures avec le meme code que le clone ; MMLU-1200 |
| `jev_clone/conformal.py` | ensembles de prediction conformes (couverture garantie) et porte a risque controle |
| `jev_clone/engine_torch.py` | moteur en passe unique (toutes les questions en un forward), cible < 50 ms |
| `tests/` | 97 tests (hors ligne, faux llama-server, Studio de bout en bout, interface et voix pilotees par Chromium, bureau, navigateur Playwright, banc Jev, conforme, integration contre un `llama-server` reel) |

## Ce qui est verifie / ce qui ne l'est pas
* Verifie ici : le paquet `jev_clone` (tests unitaires), l'agent navigateur sur une page locale (Playwright), la lecture par grammaire et le cache de prefixe
  contre un vrai `llama-server` du fork PrismML (`prism-b10683`, build CPU) sur un modele factice, les
  scripts shell (syntaxe), les chiffres tires des livres blancs et depots PrismML.
* Non verifie ici (pas de GPU dans l'environnement de redaction) : les debits RTX 4060 (estimations), le
  telechargement des poids Hugging Face (motifs de fichiers repris du depot officiel `Bonsai-demo`),
  le script d'entrainement.

Licence : Apache 2.0 (ce depot). Bonsai et Qwen : Apache 2.0. Non affilie a TypeSafe AI ni a PrismML.

# LLM-and-Classifier : Prophet Studio, fusion Jev-clone (System One) x Bonsai 2 27B (System Two) sur RTX 5060

Un **classifieur / decideur** (System One, S1 : le moteur maison qui reprend le contrat du "System One model" Jev de
[TypeSafe AI](docs/01-JEV-typesafe-analyse.md), decisions typees lues sur les logits, sans le modele de Jev) fusionne avec
un **LLM** de 27 milliards de parametres qui tient dans 6 Go ([Bonsai 2 27B](docs/02-BONSAI-2-27B-analyse.md) PTQ1_0,
PrismML, base Qwen3.8-27B ; System Two, S2), le tout sur **une RTX 5060 8 Go** sous Windows 11 (S2 sur le GPU, S1 sur
le CPU ; aussi 4060, 5060 Ti 16 Go, CPU seul), dans une application locale facon Claude Code avec commandes vocales.

Etat reel : S1 lit aujourd'hui un GGUF **Ternary-Bonsai de PrismML tel quel, en zero-shot** (aucun clone entraine n'est
publie ; la chaine d'entrainement et l'import existent). **Rien n'a encore tourne sur les vrais modeles ni sur un vrai
GPU** : les debits et latences cites sont des estimations, a mesurer avec `scripts/measure_rtx5060.ps1`.

## Prophet Studio : l'application

![Prophet Studio : un tour d'agent](docs/img/studio-agent.jpg)

Tant que la branche `claude/local-llm-high-performance-q4wk6f` n'est pas fusionnee dans `main` (les URL `.../main/...`
repondent 404), installer depuis la branche :

```powershell
# Windows (RTX 5060 : pilote NVIDIA >= 570)
& ([scriptblock]::Create((irm https://raw.githubusercontent.com/speed25200-cyber/LLM-and-Classifier/claude/local-llm-high-performance-q4wk6f/installer/install.ps1))) -Ref claude/local-llm-high-performance-q4wk6f
```
```bash
# Linux / macOS
curl -fsSL https://raw.githubusercontent.com/speed25200-cyber/LLM-and-Classifier/claude/local-llm-high-performance-q4wk6f/installer/install.sh | sh -s -- --ref claude/local-llm-high-performance-q4wk6f
# ou depuis les sources :  uv sync --extra studio && uv run prophet-studio      (sans modele : --demo)
```
Apres la fusion : `irm https://raw.githubusercontent.com/speed25200-cyber/LLM-and-Classifier/main/installer/install.ps1 | iex`
(Windows) et `curl -fsSL https://raw.githubusercontent.com/speed25200-cyber/LLM-and-Classifier/main/installer/install.sh | sh`.

L'assistant de premier lancement detecte la carte, calcule la configuration (RTX 5060 : Bonsai 2 27B PTQ1_0 entier sur
GPU, 12 a 48 k de contexte selon la VRAM que l'ecran et les autres applications occupent deja, classifieur
Ternary-Bonsai 1.7B sur CPU, 47-59 tok/s estimes), installe le runtime llama.cpp du fork PrismML (CUDA 12.8+ pour
Blackwell), les modeles et les voix, puis demarre tout.

* **Agent facon Claude Code** : flux token par token, reflexion repliable, cartes d'outils (diffs, terminal,
  fichiers), `edit_file` / `grep` / `glob`, mode plan, effort de reflexion, commandes `/`, mentions `@fichier`,
  palette Ctrl+K, sessions rejouables.
* **La fusion visible a chaque tour** : S1 choisit la voie (reponse directe ou agent, reprise si la reponse directe
  echoue), le budget de reflexion de Bonsai et l'ordre des outils ; en modes *Smart* et *Toujours demander*, il juge avant
  execution les commandes, le code, les outils crees, la navigation et l'ecriture de fichiers executables (en *Jamais
  demander*, il n'est pas consulte) ; il verifie la reponse ; Bonsai le consulte par les outils `judge_*`. Sur une
  RTX 5060, estimation : 100-350 ms par decision une fois l'etat lu, plus le prefill CPU de chaque etat neuf.
* **Voix** : push-to-talk (Ctrl+Maj+Espace) ou mains libres avec mot d'eveil ; grammaire exacte d'abord, S1 pour les
  enonces courts ambigus ; accepter / refuser une autorisation seulement sur la phrase exacte ; tout sur CPU.
* **Computer use** (desactive par defaut) : navigateur (arbre ARIA, Playwright + Chromium a installer a part) et bureau
  Windows (UI Automation) ; S1 decide les pas simples, Bonsai reprend en cas de doute ; clic, saisie, raccourci et
  lancement d'application sont juges avant execution (tous les modes).
* **Sobre** : ~390 Ko de JS d'interface, aucune ressource distante, rendu CPU par defaut dans l'application de bureau ;
  planificateur VRAM, echelle anti-OOM, watchdog et mode mono (Bonsai repond aux questions S1 si le classifieur manque).
* **Ouvert** : API locale compatible OpenAI (`/v1/chat/completions`) et TypeSafe (`/v1/systemone`), protegee par jeton.

Documentation : **[docs/12-PROPHET-STUDIO.md](docs/12-PROPHET-STUDIO.md)**, dont la section
**[5. Le classifieur (System One) et Bonsai (System Two)](docs/12-PROPHET-STUDIO.md#5-le-classifieur-system-one-et-bonsai-system-two)** :
ce qu'est S1 aujourd'hui, ce que "calibre" veut dire, entrainer et importer un clone, mode mono, chaque mecanisme avec ses
seuils, ce qui reste dans la bibliotheque, et comment mesurer sur votre RTX 5060.

## Le protocole de fusion

**Lire d'abord : [docs/00-PROTOCOLE-FUSION-JEV-BONSAI.md](docs/00-PROTOCOLE-FUSION-JEV-BONSAI.md)** (le
protocole complet, en 9 phases, avec criteres d'acceptation, risques et commandes).

## Idee en quatre lignes
1. Le clone de Jev lit, en une requete et sans generer de texte, une distribution sur les options de chaque question
   typee (`choice` / `score` / `noul`), par grammaire sur les logits de llama-server ; une calibration par question
   (temperature, seuils) s'y ajoute pour les questions du pre-tour.
2. Prophet s'en sert pour choisir la voie : reponse directe de Bonsai sans outils, ou voie agent avec un budget de
   reflexion selon le risque ; commandes, code et ecritures executables sont juges avant execution (sauf en mode
   *Jamais demander*).
3. Dans l'autre sens, Bonsai **consulte le clone comme un outil** (`judge_*`) pendant qu'il planifie ou pilote le
   navigateur : agent "computer use" a deux vitesses (`docs/06-AGENT-COMPUTER-USE.md`).
4. Tout est journalise ; les trajectoires et les donnees de Prophet servent a entrainer un clone (chaine prete, jamais
   executee sur de vrais poids).

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
jev serve --port 8008            # POST /v1/systemone (format TypeSafe) et POST /v1/fusion/decide (FusionRouter)
```
Sans GPU : `PROFILE=scripts/profiles/minimal-cpu.env` (Bonsai-8B 1-bit + clone 1.7B, 8 Go de RAM).
Avec une **A100 80 Go sur Colab** : `colab/jev_bonsai_a100.ipynb` entraine le classifieur et exporte un GGUF a importer
dans Studio (Modeles > Importer un GGUF, role Classifieur, puis Calibrer ; voir `training/README.md`, `docs/05-COLAB-A100.md`).

## Contenu du depot
| Chemin | Contenu |
|---|---|
| `prophet_studio/`, `ui/`, `desktop/`, `installer/` | **Prophet Studio** : coeur (materiel, planificateur VRAM, installateur, superviseur, sessions, calibration, voix, API), interface Svelte, application Tauri, installeurs en une ligne ([docs/12](docs/12-PROPHET-STUDIO.md)) |
| `jev_clone/` | le clone : `schema` (contrat), `prompt` (prefixe + branches), `backend_llamacpp` (lecture par grammaire), `readout` (temperature, confiance), `engine`, `calibrate`, `distill`, `guard` (verdict du garde-fou), `tools` (le clone comme outils de Bonsai + boucle d'agent), `prophet` (l'agent), `computer_use` (navigateur), `desktop_use` (bureau Windows), `server`, `cli` |
| `jev_clone/fusion.py` (`FusionRouter`, `GatePolicy`), `guided.py`, `conformal.py`, `engine_torch.py` | **bibliotheque seulement**, non utilises par Studio : routeur par seuil (`jev serve`), S1 dans le decodage de Bonsai, ensembles conformes, moteur PyTorch en passe unique (jamais execute). Seul `fusion.gate_statistic` sert a Studio (seuils de `calibrate.py`) |
| `docs/00-PROTOCOLE-FUSION-JEV-BONSAI.md` | **le protocole complet** |
| `docs/01-JEV-typesafe-analyse.md` | comment Jev fonctionne (contrat, mecanisme deduit, RLCD, benchmarks, clones ouverts) |
| `docs/02-BONSAI-2-27B-analyse.md` | Bonsai 2 27B : architecture, formats, qualite, vitesse, memoire, runtime |
| `docs/03-MATERIEL-profils.md` | RTX 4060 et exigences minimales, matrice de profils, feuille VRAM, depannage OOM |
| `docs/04-REFERENCES.md` | sources |
| `scripts/` | `setup.sh`, `start_bonsai.sh` (`BONSAI_GGUF`, `BONSAI_LORA`), `start_jev_clone.sh`, `bench.sh`, `measure_rtx5060.ps1` / `measure.sh` (mesure du duo reel), `fetch_orcabonsai.sh`, `profiles/*.env`, `windows/*.ps1` |
| `training/` | chaine d'entrainement du classifieur : `make_synthetic_prophet.py` (donnees au format de Prophet), `train_lora_rlcd.py` (LoRA / QLoRA / `--full`, regle de score propre + distillation), `merge_lora.py` (GGUF), `eval_clone.py`, `make_from_trajectories.py` (DAgger), `make_from_history.py`, `make_public_mix.py` ([training/README.md](training/README.md)) |
| `colab/` | notebook A100 80 Go : donnees, Bonsai enseignant, entrainement, fusion, export GGUF, calibration |
| `docs/05-COLAB-A100.md` | usine A100 / production RTX : quoi faire ou, durees, artefacts |
| `examples/` | routage de tickets avec escalade (`FusionRouter`) ; boucle d'agent ; agent navigateur (`browser_agent.py`) |
| `docs/06-AGENT-COMPUTER-USE.md` | fusion bidirectionnelle, computer use, budget de latence, ecart honnete avec Jev |
| `docs/11-PROPHET.md`, `jev_clone/prophet.py` | **Prophet** : agent general local a deux vitesses (Studio et REPL `jev prophet`) : code, applications, shell, web, memoire, outils qu'il cree lui-meme ; le clone juge des proprietes (reponse directe, budget, risque, outils pertinents), jamais des cases |
| `docs/10-CAS-D-USAGE.md`, `jev_clone/presets.py` | routage, choix de competence, reranking, garde-fous, extraction typee : schemas prets a l'emploi ; `ledger_report.py` ; `training/make_from_history.py` (votre historique -> entrainement) |
| `docs/09-FUSION-PROFONDE.md` | les trois niveaux de fusion (systeme, inference, modele), ce qui tourne dans Studio et ce qui reste dans la bibliotheque |
| `docs/08-ORCABONSAI-UNCENSORED.md` | variante Bonsai 2 sans refus (adaptateur LoRA de rang 1, scripts shell ; Studio ne le charge pas) |
| `docs/07-BATTRE-JEV.md`, `eval/SCOREBOARD.md` | cibles chiffrees contre les chiffres publics de Jev, tableau du duo reel a remplir sur votre RTX 5060 |
| `eval/` | banc de mesure : les 60 cas et les sorties reelles de Jev (jev-benchmark), MMLU-1200, `measure_duo.py` (duo S1 + S2 en une commande) |
| `tests/` | 358 tests ; sous Linux, 349 passent sans GPU (faux llama-server, Studio de bout en bout, interface et voix pilotees par Chromium, bureau simule, navigateur Playwright, garde-fou, calibration, chaine d'entrainement), 9 sont sautes (8 attendent de vrais serveurs, 1 un adaptateur LoRA entraine et une source llama.cpp) |

## Ce qui est verifie / ce qui ne l'est pas
* Verifie ici (Linux, sans GPU) : `uv run --no-sync pytest -q` -> **349 passent, 9 sautes** (tests en direct qui attendent
  `JEV_TEST_SERVER`, `JEV_TEST_S1` / `JEV_TEST_S2`, `JEV_TEACHER_URL`, `JEV_S1_EVAL_URL` ; fusion LoRA qui attend
  `MERGE_LORA_ADAPTER` + `LLAMA_CPP_DIR`). CI : au commit `288bb87` (CI #17), tests verts sous Linux et Windows.
* Installeurs de bureau : aucune release publiee ; paquets (NSIS + MSI, deb + AppImage, dmg) construits par la CI au commit
  `288bb87` (workflow Bureau #3, coeur a jour), en artefacts ; reconstruits a chaque modification du coeur (docs/12, section 1).
* Non verifie : tout ce qui demande les vrais modeles ou un vrai GPU (debits, latences S1, qualite des decisions
  zero-shot, calibration reelle, VRAM) ; les installeurs executes sous Windows et macOS ; UI Automation sur un vrai
  Windows ; la chaine d'entrainement sur de vrais poids ; le garde-fou n'a eu qu'une revue adversariale (tests unitaires
  et de regression). Detail : [docs/12, section 10](docs/12-PROPHET-STUDIO.md#10-ce-qui-est-verifie-ce-qui-ne-lest-pas).

Licence : Apache 2.0 (ce depot). Bonsai et Qwen : Apache 2.0. Non affilie a TypeSafe AI ni a PrismML.

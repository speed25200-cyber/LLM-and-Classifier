# 11 - Prophet : agent general local, rapide grace a la fusion, fabrique a partir de zero donnee

Prophet est l'agent de Prophet Studio ([docs/12](12-PROPHET-STUDIO.md), section 5 : seuils et comportements exacts) et
du REPL `jev prophet`. Code : `jev_clone/prophet.py`, garde-fou `jev_clone/guard.py`.

## 1. Principe : le clone n'enferme jamais l'agent, il l'accelere et le protege

Prophet n'est pas un routeur vers des cases. Bonsai (System Two) a **toujours** la pleine autonomie : ecrire et
executer du code, creer des applications, piloter le shell, naviguer sur le web, prendre des notes, et
**se construire de nouveaux outils** (`create_tool`) quand ceux qui existent ne suffisent pas. Il n'y a pas de
liste fermee de ce qu'il sait faire.

Le clone de Jev (System One) juge en une requete, a chaque tour, des **proprietes** de la demande, jamais une
categorie qui limiterait l'action. Aujourd'hui ce clone est un Ternary-Bonsai de PrismML lu en zero-shot (aucun clone
entraine n'est publie) ; sur une RTX 5060 il tourne sur CPU (estimation : 100-350 ms par decision une fois l'etat lu,
plus le prefill d'un etat neuf ; rien n'est mesure).

| Jugement | Effet |
|---|---|
| `direct` : une reponse courte suffit-elle sans outil ? | voie directe si >= 0,8 (ou seuil calibre, jamais sous 0,5), sans raisonnement, risque 0, hors effort profond et mode plan : Bonsai repond sans outils ni reflexion (600 tokens au plus) ; reprise en voie agent si la reponse est vide, contient NEEDS_TOOLS, est tronquee, pretend avoir agi, ou si la verification S1 est < 0,35 |
| `needs_reasoning`, `risk` | budget de reflexion de Bonsai par niveau de risque : 0 / 512 / 2 048 / 6 144 tokens, au moins 512 si raisonnement ou lecture du risque pas sure ; plafond 60 % de `max_tokens` (dans Studio, 4 915 au plus : 6 144 n'est jamais atteint ; docs/12, 5.5.3) |
| `clarify` | simple indice transmis a Bonsai, qui decide lui-meme de poser une question ou d'agir |
| un noul par outil du catalogue | ordre de presentation ; au-dela de `max_tools` (12), seuls les pertinents sont exposes ; `done`, `remember`, `create_tool` et les `judge_*` restent toujours la |
| `intent`, `language` | **observation seulement** (journal, entrainement), jamais un aiguillage |
| risque de chaque action (`tool_risk` en 5 classes, `risk`, `policy_violation`) | confirmation si la masse destructive + privileged + exfiltration >= 0,35 ou `risk` >= 1,5 ; arret obligatoire (aucune autorisation memorisee) si classe risquee en tete, masse >= 0,5, politique >= 0,5, `risk` >= 2,5, clone en panne ou code lance invisible |
| verification finale | "la demande est-elle satisfaite ?" : journalisee (voie agent), reprise en voie agent sous 0,35 (voie directe) |

Pendant la boucle, Bonsai consulte le clone (`judge_choice`, `judge_rank`...) pour ses micro-decisions au lieu
de raisonner dessus : c'est la fusion dans les deux sens, appliquee a chaque tour.

```
vous > "Crée-moi une appli de suivi de dépenses avec des graphiques"
  clone : direct=0.05  clarify=0.08  needs_reasoning=0.3  risk=0  outils pertinents : write_file, run_command, python, read_file...
  Bonsai : write_file expenses/app.py, write_file expenses/README.md, run_command "python -m pytest expenses" ... done
  clone : verification 0.91
prophet [agent, 84 s, verif 0.91] > Created expenses/ (Flask + Chart.js, tests pass). Run: python expenses/app.py
```
(exemple illustratif, pas une mesure)

Outils de base : `write_file`, `edit_file`, `read_file`, `list_files`, `glob`, `grep`, `run_command`, `python`, `remember`,
`create_tool`, `done`, plus `judge_*` ; `browse` (agent navigateur de `computer_use.py`, option `--browser`) et, dans Studio,
`desktop`. Les outils crees vivent dans `.prophet/skills/<nom>.py` (TOOL + run), sont lus sans executer leur code et ne
s'executent qu'a l'appel, apres le garde-fou (Studio, mode Jamais demander : sans garde-fou, S1 n'est pas consulte ; REPL
`--yes` : juges, sans confirmation) ; les outils de fichiers n'ecrivent jamais dans `.prophet/`.

```bash
./scripts/start_bonsai.sh ; ./scripts/start_jev_clone.sh        # ou le profil OrcaBonsai
jev prophet --workspace ~/prophet --browser                        # /quit pour sortir
```

## 2. Zero donnee : comment ses jugements s'entrainent

| Couche | Source | Volume |
|---|---|---|
| Graines | `training/seeds/prophet_seeds.jsonl` (copie livree : `jev_clone/seeds/`) : 82 demandes manuscrites (FR/EN), avec `direct`, `clarify`, `needs_reasoning`, `risk`, `intent`, `language` ; cas ambigus et cas pieges au risque maximal. Servent a la calibration (bouton Calibrer) ; jamais des lignes d'entrainement elles-memes, mais avec `--expand` chaque graine fait ecrire a Bonsai des demandes qui heritent de ses etiquettes `intent`, `language`, `risk` (la calibration sur les graines n'est alors plus tenue a l'ecart) | 82 |
| Synthese | `training/make_synthetic_prophet.py` : exemples par regles au format exact des appels de Prophet (pre-tour, garde-fou, outils, verification, voix), `--teacher` : Bonsai enseignant, `--expand` : graines etendues par Bonsai | 20 000 par defaut |
| Usage | chaque tour de Prophet dans `.prophet/ledger.jsonl` (jugements, outils exposes et utilises, blocages, verification) ; trajectoires du computer use (`training/make_from_trajectories.py`) | croissant |

```bash
python -m jev_clone.calibrate --server http://127.0.0.1:8081 --data training/seeds/prophet_seeds.jsonl --out runs/calibration.json   # jour 1
python training/make_synthetic_prophet.py --out data/train.jsonl --val data/val.jsonl --calib data/calib.jsonl --n 20000 \
    --teacher http://127.0.0.1:8080                                                                                            # la nuit
# puis A100 : training/train_lora_rlcd.py, training/merge_lora.py -> GGUF (training/README.md, colab/jev_bonsai_a100.ipynb)
```
Calibree sur les graines ou sur `data/calib.jsonl`, la calibration ne touche que les questions du pre-tour (docs/12,
section 5.2). Rien de cette chaine n'a encore tourne sur de vrais poids.

## 3. Ce que Prophet fait au jour 1, et ce qui reste

Fait (code et tests sur faux serveurs) : projets complets (fichiers, README, dependances, tests) et leur lancement ;
modification de code ; commandes et Python ; web ; memoire ; nouveaux outils reutilisables ; refus ou confirmation des
actions dangereuses ; dans Studio : voix et bureau Windows (UI Automation ; teste sur un bureau simule seulement).
Reste : mesurer tout cela sur les vrais modeles ; un clone entraine ; les taches de plusieurs heures sans supervision.
Ordres de grandeur a verifier (estimations) : RTX 5060 ~47-59 tok/s pour Bonsai 2 PTQ1_0, RTX 4060 ~28-36 ; une
application simple = quelques minutes de generation ; mesure : `eval/SCOREBOARD.md`.

## 4. Securite

Bonsai (surtout avec OrcaBonsai) fera ce qu'on lui demande ; la securite est dans le clone et dans vous : jugement de
risque avant chaque commande, code, nouvel outil, appel d'outil cree, navigation et ecriture de fichier executable, avec
le contenu des scripts que la commande lance (pas dans le mode Jamais demander de Studio, qui ne consulte pas le clone) ;
confirmation selon le verdict (masse risquee, arrets obligatoires) ; espace de travail borne ; journal. Dans le REPL, `--yes` supprime les confirmations : reserve a un bac a sable. Dans Studio, les
modes Smart, Toujours demander et Jamais demander (docs/12, section 5.5.6).

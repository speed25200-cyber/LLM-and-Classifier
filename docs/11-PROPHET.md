# 11 - Prophet : agent general local, rapide grace a la fusion, fabrique a partir de zero donnee

## 1. Principe : le clone n'enferme jamais l'agent, il l'accelere et le protege

Prophet n'est pas un routeur vers des cases. Bonsai (System Two) a **toujours** la pleine autonomie : ecrire et
executer du code, creer des applications, piloter le shell, naviguer sur le web, prendre des notes, et
**se construire de nouveaux outils** (`create_tool`) quand ceux qui existent ne suffisent pas. Il n'y a pas de
liste fermee de ce qu'il sait faire.

Le clone de Jev (System One) intervient a chaque tour en une passe (~150 ms), sur des **proprietes** de la
demande, jamais sur une categorie qui limiterait l'action :

| Jugement | Effet |
|---|---|
| `direct` : une reponse courte suffit-elle sans outil ? | si >= 0,8, sans raisonnement ni risque : reponse immediate de Bonsai sans reflexion (1-3 s) |
| `needs_reasoning`, `risk` | budget de reflexion de Bonsai : 0 / 512 / 2 048 / 6 144 tokens |
| `clarify` | simple indice transmis a Bonsai, qui decide lui-meme de poser une question ou d'agir |
| un noul par outil du catalogue | ordre de presentation ; au-dela de `max_tools`, seuls les pertinents sont exposes (contexte court, donc rapide) ; `done`, `remember`, `create_tool` et les `judge_*` restent toujours la |
| `intent`, `language` | **observation seulement** (journal, entrainement), jamais un aiguillage |
| risque de chaque commande / code / nouvel outil | confirmation humaine au-dela du niveau 2 ou hors lecture seule |
| verification finale | "la demande est-elle satisfaite ?" journalisee |

Pendant la boucle, Bonsai consulte le clone (`judge_choice`, `judge_rank`...) pour ses micro-decisions au lieu
de raisonner dessus : c'est la fusion dans les deux sens, appliquee a chaque tour.

```
vous > "Crée-moi une appli de suivi de dépenses avec des graphiques"
  clone : direct=0.05  clarify=0.08  needs_reasoning=0.3  risk=0  outils pertinents : write_file, run_command, python, read_file...
  Bonsai : write_file expenses/app.py, write_file expenses/README.md, run_command "python -m pytest expenses" ... done
  clone : verification 0.91
prophet [agent, 84 s, verif 0.91] > Created expenses/ (Flask + Chart.js, tests pass). Run: python expenses/app.py
```

Outils de base : `write_file`, `read_file`, `list_files`, `run_command`, `python`, `browse` (agent navigateur a
deux vitesses de `computer_use.py`, option `--browser`), `remember`, `create_tool`, `done`, plus `judge_*`.
Les outils crees vivent dans `.prophet/skills/<nom>.py` (TOOL + run) et sont charges a chaque tour.

```bash
./scripts/start_bonsai.sh ; ./scripts/start_jev_clone.sh        # ou le profil OrcaBonsai
jev prophet --workspace ~/prophet --browser                        # /quit pour sortir
```

## 2. Zero donnee : comment ses jugements s'entrainent

| Couche | Source | Volume |
|---|---|---|
| Graines | `training/seeds/prophet_seeds.jsonl` : 82 demandes manuscrites (FR/EN), avec `direct`, `clarify`, `needs_reasoning`, `risk`, `intent`, `language` ; cas ambigus et cas pieges au risque maximal | 82 |
| Synthese | `training/make_synthetic_prophet.py` : Bonsai etend chaque graine en 20 demandes (JSON contraint) ; `jev_clone.distill` complete les etiquettes | ~1 800 |
| Usage | chaque tour de Prophet dans `.prophet/ledger.jsonl` (jugements, outils exposes et utilises, blocages, verification) | croissant |

```bash
python -m jev_clone.calibrate --server http://127.0.0.1:8081 --data training/seeds/prophet_seeds.jsonl --out runs/calibration.json   # jour 1
python training/make_synthetic_prophet.py --bonsai http://127.0.0.1:8080 --out data/prophet_states.jsonl --per-seed 20                  # la nuit
python -m jev_clone.distill --bonsai http://127.0.0.1:8080 --in data/prophet_states.jsonl --out data/prophet_labeled.jsonl --mode soft --think-if-below 0.85
# puis A100 : training/train_lora_rlcd.py --data data/prophet_labeled.jsonl --sft-data data/gen.jsonl ...  -> le meme petit modele juge et repond
```

## 3. Ce que Prophet fait au jour 1, et ce qui reste

Fait : projets complets (fichiers, README, dependances, tests) et leur lancement ; modification de code ; commandes
et Python ; web ; memoire ; nouveaux outils reutilisables ; refus ou confirmation des actions dangereuses.
Reste : la voix ; le bureau hors navigateur ; les taches de plusieurs heures sans supervision.
Ordres de grandeur sur RTX 4060 : reponse directe 1-3 s ; application simple 3-8 min (generation de Bonsai a
~30 tok/s) ; jugements 0,15-0,3 s par tour. RTX 5060 : ~1,6x plus vite.

## 4. Securite

Bonsai (surtout avec OrcaBonsai) fera ce qu'on lui demande ; la securite est dans le clone et dans vous :
jugement de risque avant chaque commande, code ou nouvel outil ; confirmation au-dela du niveau 2 ; espace
de travail borne ; journal. `--yes` supprime les confirmations : reserve a un bac a sable.

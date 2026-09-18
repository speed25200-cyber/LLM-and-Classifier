# 10 - Les cinq cas d'usage "pre-traitement d'agent" et ou ils sont dans le depot

Le raisonnement (repris de la discussion publique autour de Jev / du decodage contraint parallele) : un agent
d'entreprise fait passer chaque item par un modele frontiere, quelques secondes et quelques centimes, pour une
decision le plus souvent evidente ; derriere, des annees d'humains prenant exactement la meme decision avec
l'issue attachee. Un modele de decision **entraine sur cet historique**, calibre, prend la decision quand il est
sur ; le LLM ne voit que le reste. Si 6 items sur 10 passent sans LLM, c'est plus de la moitie de la depense
qui disparait a exactitude comparable.

| Cas d'usage | Ce que fait le clone | Dans le depot |
|---|---|---|
| Routage de requete / de modele (economique, frontiere, humain) | `route` (choice) + `risk` (score) + porte de confiance | `presets.ROUTING`, `fusion.FusionRouter`, seuils de `calibrate.py` / `conformal.py` |
| Choix de la competence / du sous-agent a charger | un `choice` sur le catalogue (jusqu'a 255) au lieu de tout mettre dans le contexte | `presets.skill_selection`, `tools.judge_choice` |
| Reranking du contexte recupere | un `noul` par passage, tous en parallele, seuls les pertinents entrent dans la fenetre | `presets.rerank`, `tools.judge_rank` |
| Garde-fous a chaque tour (injection, politique, contradiction, risque d'outil) | 4 nouls / choice en une passe avant d'executer | `presets.GUARDRAILS`, `computer_use.FastPolicy.verify`, `eval/jev_benchmark.py` |
| Extraction typee (courriels, PDF, transcriptions) avant tout traitement couteux | N champs types en une passe, mode valeurs jusqu'a 255 options | `presets.extraction`, `tools.judge_batch`, `label_mode="values"` |

## Le "custom" : votre historique est le jeu d'entrainement

```bash
python training/make_from_history.py --in decisions.csv --state-col text --schema schema.json \
    --out data/history.jsonl --val data/history_val.jsonl --outcome-col outcome
python -m jev_clone.calibrate --server http://127.0.0.1:8081 --data data/history_val.jsonl --out runs/calibration.json
# puis phase 7 (A100) : training/train_lora_rlcd.py --data data/history.jsonl ...
```
`--outcome-col` ne garde que les decisions confirmees par l'issue : on apprend ce qui a marche, pas ce qui a
ete decide. Meme avant tout entrainement, ces lignes suffisent a calibrer les seuils (phase 4).

## Mesurer l'economie, pas la promettre

`FusionRouter` journalise chaque item ; `python -m jev_clone.ledger_report runs/ledger.jsonl --s2-tokens 900 --s2-ms 12000 --s2-cost 0.002`
donne le taux d'escalade, les latences par chemin, la part de decisions prises par System One question par
question, et l'economie estimee (appels LLM evites, tokens, temps, cout) selon vos couts unitaires. C'est le
chiffre a mettre en face de "6 sur 10".

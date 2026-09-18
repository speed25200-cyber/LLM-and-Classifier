# Tableau de bord : le clone contre Jev, chiffre par chiffre

Seuls les chiffres **publics** de Jev sont opposables. Les colonnes "clone" se remplissent avec les scripts de
`eval/` ; rien n'est a inscrire tant que ce n'est pas mesure. Mise a jour a chaque niveau (0 = sans entrainement,
1 = entraine sur A100, 2 = + trajectoires agent + conforme).

| Mesure | Jev (public) | Source Jev | Clone niveau 0 | Clone niveau 1 | Clone niveau 2 | Cible "devant Jev" |
|---|---|---|---|---|---|---|
| jev-benchmark, 60 cas de risque d'appel d'outil : accuracy | 91,7 % (55/60) | themsquared/jev-benchmark, 17.09.26 | | | | >= 93,3 % (56+/60) sur 5 runs |
| ... clear / ambiguous / adversarial | 100 % / 71,4 % / 91,7 % | idem | | | | ambiguous >= 78,6 % (11/14) |
| ... ECE (10 bacs) | 0,071 (latest) / 0,051 (preview) | idem | | | | <= 0,05 |
| ... erreurs a confiance 1,0 / >= 0,9 | 0 / 5 ; **1 / 5** (une erreur a 0,97, recalculee depuis les sorties brutes) | idem | | | | 0 ; 0 |
| ... latence p50 / p95 | 421,6 / 542,0 ms (API) | idem | | | | < 100 / 150 ms (local, llama.cpp) ; < 50 ms (torch) |
| MMLU 1 200 items : ECE apres temperature | 0,031 | annonce TypeSafe | | | | <= 0,030 |
| MMLU 1 200 items : accuracy | non publie | (reflex 4B : 72 %) | | | | >= 72 % (4B) |
| Options par question | 255 | docs TypeSafe | 255 (lettres <= 26, noms au-dela) | 255 | 255 (+ rank illimite) | rang de N candidats illimite (nouls paralleles) |
| Etat maximal | 32 k tokens (64 k avec les questions) | docs TypeSafe | 8-32 k selon `-c` | idem | idem | 32 k |
| Cout par decision | ~0,0004 $ (workflow) ; 0,042 $/M tokens | TypeSafe | 0 $ | 0 $ | 0 $ | 0 $ |
| Garantie de couverture (prediction conforme) | aucune publiee | - | non | non | **oui** (`conformal.py`) | garantie 1-alpha, verifiee |
| Porte a risque controle (binomiale exacte) | aucune publiee | - | non | non | **oui** | erreur <= alpha a 90 % |
| Images dans l'etat | non documente | - | non (llama.cpp texte) | non | oui (clone VL, torch) | oui |
| Le LLM consulte le decideur (outils) | non (produit separe) | - | oui | oui | oui | oui |
| Boucle d'amelioration locale (S2 enseigne S1) | non | - | oui | oui | oui | oui |
| Donnees hors machine | oui (API) | - | non | non | non | non |

Protocole de mesure : `python -m eval.jev_benchmark --server ... --name clone-l0 --repeat 5` puis `--analyze` ;
`python -m eval.mmlu_ece --server ... --n 1200` (Colab) ; latence = `jev bench` et la colonne p50 du benchmark.
Toujours 5 repetitions sur les 60 cas (n=60 a une variance de +/- 3 points, notee par l'auteur du benchmark).

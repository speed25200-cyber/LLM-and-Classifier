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

## Machine cible : RTX 5060 8 Go (Prophet Studio, bouton **Mesurer** ou `POST /api/bench`)

Estimations du planificateur a remplacer par des mesures (`runs/bench.jsonl` les conserve, avec le plan utilise).
Les lignes generation, prefill et decision System One se remplissent aussi depuis le tableau "Duo reel" ci-dessous
(meme mesure, plus complete, en une commande : voir "Mesurer sur votre machine").

| Mesure (RTX 5060 8 Go, pilote >= 570, build CUDA 12.8+) | Estimation | Mesure |
|---|---|---|
| Bonsai 2 27B PTQ1_0, generation (tok/s) | 47-59 | |
| Bonsai 2 27B PTQ1_0, prefill (tok/s) | a mesurer | |
| Bonsai 27B Q1_0 (priorite vitesse), generation (tok/s) | 53-65 | |
| Contexte obtenu sans OOM (ecran sur la 5060 / sur l'iGPU) | 24 k / 32 k (KV q4_0) | |
| Decision System One p50 / p95, classifieur 1.7B sur CPU (ms) | 100-350 | |
| Decision System One p50, classifieur sur GPU (priorite vitesse) (ms) | 40-150 | |
| Reconnaissance vocale (Parakeet v3, CPU) : latence pour 3 s de parole | a mesurer | |
| Commande vocale de bout en bout (fin de parole -> action) | cible < 600 ms | |

### Duo reel (`eval/measure_duo.py`)

Le duo tel que Prophet l'utilise : classifieur (S1) et modele de raisonnement (S2) en marche ensemble, un vrai tour
Prophet de bout en bout. La mesure affiche ce tableau rempli : remplacez-le tel quel (la colonne Conditions porte la
date, le GPU et son pilote, les modeles lus sur les serveurs, le contexte et les slots). Le JSON complet
(`eval/results/duo-<date>.json`) garde chaque serie : p50 / p95 / min / max, tokens lus, debits et premier token par
essai, decision S1 et outils de chaque tour.

| Mesure (duo reel) | Estimation | Mesure | Conditions |
|---|---|---|---|
| S2 generation (tok/s) | 47-59 | | |
| S2 prefill (tok/s, invite de ~1 500 tokens) | a mesurer | | |
| S2 premier token (ms) : invite courte / ~1 500 tokens | a mesurer | | |
| S2 reflexion : tokens avec budget 0 / budget 512 (respecte ?) | 0 / <= 512 | | |
| S1 pre-tour (6 questions) a froid p50 / p95 (ms), etat neuf ~2 k tokens | a mesurer | | |
| S1 pre-tour a chaud p50 / p95 (ms), etat deja lu | 100-350 | | |
| S1 a chaud p50 (ms) : noul / choice / score | a mesurer | | |
| S1 selection des outils (un noul par outil) a froid / a chaud p50 (ms) | a mesurer | | |
| S1 garde-fou (3 questions) a froid / a chaud p50 (ms) | a mesurer | | |
| S1 lecture : masse sur les etiquettes avec grammaire / sans (noul choice score) | 1,00 / a mesurer | | |
| Tour Prophet, voie directe : duree, appels S1 / ms S1, appels S2 | a mesurer | | |
| Tour Prophet, voie agent (creation de fichier) : duree, appels S1 / ms S1, appels S2 | a mesurer | | |
| VRAM utilisee avant / pic / apres (Mio) | a mesurer | | |

Lecture du tableau :
* **a froid** : etat de la forme d'un pre-tour Prophet (demande de 2 500 caracteres, 60 fichiers, 4 tours recents, soit
  ~2 k tokens), tire au hasard avec un nonce en tete : le serveur le lit en entier, comme a chaque nouveau tour.
  **a chaud** : le meme etat relu, prefixe en cache (seules les questions sont lues). noul / choice / score = une seule
  question de chaque primitive ; pre-tour, selection des outils et garde-fou = les lectures reelles d'un tour.
* **masse sur les etiquettes** : avec la grammaire (lecture de Prophet), elle doit valoir 1,00 et toutes les etiquettes
  doivent etre presentes ; sans grammaire, c'est la part de la distribution du token suivant que le classifieur met de
  lui-meme sur les etiquettes (zero-shot : plus elle est basse, plus la decision depend de la grammaire).
* **reflexion** : `thinking_budget_tokens` envoye comme le fait Prophet (0 = reflexion coupee), tokens de reflexion
  comptes par le tokenizer du serveur ; respecte = 0 pour le budget 0, au plus 512 (+ 5 %) pour le budget 512.
* **tours** : une question courte (voie directe attendue) puis une tache qui cree un fichier (voie agent attendue), chacune
  dans une session neuve et un dossier temporaire, en `permission_mode` auto (aucune confirmation), supprimes ensuite.
  La voie prise est celle que S1 a choisie (reprise en voie agent comprise) ; "invariants NON" signale une incoherence
  (decision S1 absente ou hors de [0, 1], aucun appel S2, aucun outil ou aucun fichier pour la tache, statistiques
  incoherentes).

## Mesurer sur votre machine

Prophet Studio doit tourner (application de bureau ou `prophet-studio`, modeles installes ; s'ils sont arretes, la
mesure les demarre). Depuis un clone du depot :

```powershell
# Windows 11 (PowerShell), RTX 5060
powershell -ExecutionPolicy Bypass -File .\scripts\measure_rtx5060.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\measure_rtx5060.ps1 -Rapide -Label "pilote 581.57"
```

```sh
# Linux / macOS
sh scripts/measure.sh
sh scripts/measure.sh --rapide --label "pilote 575"
```

Les deux scripts lisent `core.json` dans le dossier de donnees (`%LOCALAPPDATA%\ProphetStudio`, ou `PROPHET_HOME`),
lancent `eval/measure_duo.py` avec l'environnement Python installe (`<donnees>\app-venv` des scripts d'installation,
sinon `<donnees>\venv` de l'application de bureau, sinon `uv run` depuis le depot), puis affichent le chemin du JSON
et le tableau "Duo reel" a coller ci-dessus (sous Windows, il est aussi copie dans le presse-papiers). Environ 5 minutes,
~2 minutes avec `-Rapide` / `--rapide`.

Sans Studio, contre deux llama-server deja lances (classifieur, puis modele de raisonnement ; meme URL en mode mono) :

```sh
uv run python -m eval.measure_duo --s1 http://127.0.0.1:8081 --s2 http://127.0.0.1:8080
```

Test d'integration sur les vrais modeles (un tour Prophet complet en voie agent, invariants de structure seulement ;
7881 / 7880 sont les ports de Studio par defaut) :

```sh
JEV_TEST_S1=http://127.0.0.1:7881 JEV_TEST_S2=http://127.0.0.1:7880 uv run --extra dev pytest -q tests/test_live_duo.py
```

```powershell
$env:JEV_TEST_S1 = 'http://127.0.0.1:7881'; $env:JEV_TEST_S2 = 'http://127.0.0.1:7880'
uv run --extra dev pytest -q tests/test_live_duo.py
```

# 06 - Agent a deux vitesses et computer use : le clone consulte Bonsai, Bonsai consulte le clone

## 1. Ce que "fusion" veut dire ici

Deux sens de circulation, implementes tous les deux :

| Sens | Mecanisme | Code | Latence |
|---|---|---|---|
| **Le clone consulte Bonsai** | porte de confiance : decision sous le seuil, question "meta" (`needs_reasoning`, `risk`) ou verification qui echoue -> Bonsai raisonne avec un budget de reflexion proportionnel au risque, repond, ses actions/etiquettes sont journalisees | `jev_clone/fusion.py` (`FusionRouter`), `computer_use.py` (`ComputerUseAgent` -> `SlowPolicy`) | 1-20 s selon budget |
| **Bonsai consulte le clone** | le clone est expose a Bonsai comme **outils OpenAI** : `judge_choice`, `judge_noul`, `judge_score`, `judge_rank` (N candidats en parallele), `judge_batch`. Pendant qu'il planifie ou pilote le navigateur, Bonsai obtient en une passe une decision calibree au lieu de raisonner dessus | `jev_clone/tools.py` (`SystemOneToolbox`, `AgentLoop`) | 50-250 ms par appel |

C'est le patron "System One / System Two" de Kahneman applique a un agent : le reflexe tranche 80-90 %
des pas, le raisonnement n'intervient que sur les pas difficiles et **enseigne** le reflexe ensuite.

```
                       boucle d'agent (navigateur, outils, taches)
   observation ──► clone : action + cible + verification (une passe, ~100-250 ms)
                     │ sur ──► executer ──► verifier (noul) ──► observation suivante
                     │ doute / risque / echec x2
                     ▼
                  Bonsai 2 : planifie avec les outils click/type/scroll/back/done + judge_*  (~5-20 s)
                     │
                  ses actions = etiquettes ──► trajectoires ──► entrainement du clone (DAgger)
```

## 2. Le computer use navigateur (`jev_clone/computer_use.py`)

* **Observation** : Playwright rend la page en un etat compact : URL, titre, arbre ARIA (texte), liste
  numerotee des elements interactifs visibles (`[3] input:email "email address"`), capture JPEG
  optionnelle pour la vision de Bonsai. Pas de pixels pour le clone : le DOM est plus fiable et 10x
  moins cher qu'une image.
* **FastPolicy (clone)** : **une seule requete** `/v1/systemone` par pas : un `choice` sur l'action
  (click / type / scroll_down / go_back / done / escalate), **un `noul` par element candidat** ("est-ce
  l'element sur lequel agir ?", jusqu'a 24 en parallele, comme `judge_rank`), un `choice` sur le "slot" a
  saisir. Le texte tape ne vient **jamais** du modele : il vient de slots structures fournis par la
  tache (`{"email": "..."}`), ce qui interdit les hallucinations de saisie.
* **Porte** : action peu sure, cible peu sure, saisie sans slot, ou action `escalate` -> `SlowPolicy`.
* **Verification** : apres chaque action, un `noul` "l'action a-t-elle fait progresser l'objectif ?" ;
  deux echecs consecutifs -> Bonsai reprend la main.
* **SlowPolicy (Bonsai)** : chat avec outils `click`, `type`, `scroll_down`, `go_back`, `observe`,
  `done` **et** les `judge_*` du clone ; capture d'ecran jointe si `--vision` (projecteur mmproj charge).
* **Trajectoires** : chaque pas est journalise (`runs/trajectories.jsonl`) ; `trajectory_to_examples()`
  transforme les pas escalades (actions de Bonsai) et les pas rapides verifies en exemples
  `{state, questions, labels}` pour `training/train_lora_rlcd.py`. C'est la boucle DAgger : le
  professeur corrige l'eleve la ou l'eleve doute, sur la distribution d'etats de l'eleve.

Verifie ici : 31 tests, dont l'agent complet sur une page locale (saisie, clic, `done`), l'escalade vers
un Bonsai factice qui agit par outils, et un passage de bout en bout contre un vrai `llama-server`
(6 questions en une requete : 257 ms sur CPU avec un modele factice ; sur RTX 4060 avec un clone 0,8-1,7B,
attendre 100-250 ms).

```bash
export JEV_S1_URL=http://127.0.0.1:8081 JEV_S2_URL=http://127.0.0.1:8080
pip install -e ".[agent]"      # playwright ; puis `playwright install chromium` si aucun Chromium local
python examples/browser_agent.py --url https://duckduckgo.com --goal "Search for Bonsai 2 27B" --slot query="Bonsai 2 27B" --headed
```

## 3. Budget de latence sur RTX 4060 (a mesurer)

| Etape | Clone 0,8-1,7B (GPU) | Bonsai 2 27B PTQ1_0 |
|---|---|---|
| observation DOM (40 elements) | 30-80 ms | - |
| decision (1 choice + 24 nouls + 1 choice, prefixe en cache) | 100-250 ms | 0,8-1,5 s (mode mono) |
| verification (1 noul) | 40-80 ms | - |
| **un pas "rapide"** | **~0,2-0,4 s + temps de chargement de la page** | - |
| escalade (planification, 2-4 appels d'outils, budget 2 048) | - | 5-20 s |
| appel `judge_*` depuis Bonsai | 50-250 ms | - |

Avec 80-90 % de pas rapides, un parcours de 10 pas prend ~5-10 s au lieu de 60-200 s en tout-LLM.
La vision (capture jointe a Bonsai) coute ~1 000 tokens de prefill par image (~3 s en PTQ1_0 sur 4060) :
a reserver aux escalades, jamais au chemin rapide.

## 4. Ce que ce depot ne fait pas encore (et comment y aller)

* **Bureau entier (pixels, n'importe quelle application)** : le chemin rapide repose sur le DOM. Pour le
  bureau, il faut un etat structure equivalent : arbre d'accessibilite Windows (UIA via `pywinauto`) ou
  macOS (AX), plus OCR/detection d'icones pour les zones sans accessibilite, et `pyautogui` pour agir.
  Le clone et les questions restent identiques : seul `observe()`/`act()` changent. Un clone **vision**
  (Qwen3.5-0,8B/2B sont VL) peut aussi decider sur capture, a ~2-3x la latence du DOM.
* **Memoire et planification longue** : Bonsai porte le plan (liste d'etapes) ; le clone execute chaque
  etape. Ajouter un `noul` "l'etape courante est-elle terminee ?" et un `choice` "quelle etape suivante ?"
  suffit pour enchainer.
* **Securite** : avant toute action irreversible (paiement, suppression, envoi), un `judge_score` de risque
  et un seuil dur qui exige Bonsai **et** une confirmation humaine (le benchmark independant de Jev porte
  exactement sur ce cas : classer readonly / destructive / privileged / exfiltration a 91,7 %).

## 5. Ecart avec Jev "SOTA" : evaluation honnete

| | Jev (TypeSafe) | Ce depot, niveau 0 (sans entrainement) | Ce depot apres A100 (phase 7 + trajectoires) |
|---|---|---|---|
| Mecanisme | architecture dediee, "parallel sampler", RLCD | lecture par grammaire sur un LM, N requetes partageant un cache | idem + poids entraines par regle de score propre + distillation Bonsai |
| Calibration (ECE) | ~0,031 (MMLU, annonce) | 0,04-0,09 avant temperature, ~0,04 apres (reflex 4B ; un 1,7B ternaire fera moins bien) | 0,035-0,045 attendu (decider 2B : 0,037 en domaine) |
| Precision hors domaine | forte (modele generaliste, ~10B ?) | moyenne | bonne sur **votre** domaine, moyenne ailleurs |
| Options par question | 255 | 26 | 26 (255 avec etiquettes bi-lettres, roadmap) |
| Latence | 70-500 ms en API | 100-250 ms local (1,7B) ; 0,3-1,5 s en mono | 100-250 ms local |
| Cout | 0,042 $/M tokens | 0 (electricite) | 0 |
| Donnees / confidentialite | cloud | local | local |

Conclusion : sur des taches **de votre domaine**, un clone 2B entraine avec Bonsai comme professeur peut
egaler ou depasser Jev (un modele specialise bat un generaliste sur son terrain) ; sur des questions
arbitraires en zero-shot, Jev restera devant. La vitesse locale est comparable. L'atout unique du montage
local est la boucle fermee : Bonsai enseigne, le clone accelere, tout reste sur votre machine.

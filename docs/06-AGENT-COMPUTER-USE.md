# 06 - Agent a deux vitesses et computer use : le clone consulte Bonsai, Bonsai consulte le clone

Seuils et comportements exacts dans Prophet Studio : [docs/12](12-PROPHET-STUDIO.md), section 5.5.12. Rien de ce qui suit
n'a tourne sur les vrais modeles ; les latences sont des estimations.

## 1. Ce que "fusion" veut dire ici

Deux sens de circulation, implementes tous les deux :

| Sens | Mecanisme | Code | Latence (estimation) |
|---|---|---|---|
| **Le clone passe la main a Bonsai** | Prophet : voie directe ou agent, budget de reflexion selon le risque, reprise en voie agent. Computer use : pas sous les portes, pas risque ou verification ratee -> Bonsai agit avec ses outils ; ses actions sont journalisees (trajectoires) | `jev_clone/prophet.py`, `computer_use.py` (`ComputerUseAgent` -> `SlowPolicy`) ; `jev_clone/fusion.py` (`FusionRouter`) seulement via `jev serve` (bibliotheque) | 1-20 s selon budget |
| **Bonsai consulte le clone** | le clone est expose a Bonsai comme **outils OpenAI** : `judge_choice`, `judge_noul`, `judge_score`, `judge_rank` (jusqu'a 50 candidats, un noul chacun), `judge_batch`. Pendant qu'il planifie ou pilote le navigateur, Bonsai obtient en une requete une decision au lieu de raisonner dessus | `jev_clone/tools.py` (`SystemOneToolbox`, `AgentLoop`) | 100-350 ms + prefill par appel avec le classifieur sur CPU (RTX 5060) ; 40-150 ms sur GPU |

C'est le patron "System One / System Two" de Kahneman applique a un agent : le reflexe tranche les pas simples, le
raisonnement n'intervient que sur les pas difficiles et **enseigne** le reflexe ensuite (quelle part des pas reste au
reflexe n'est pas mesuree).

```
                       boucle d'agent (navigateur, bureau)
   observation ──► clone : action + cible + slot + objectif atteint (une requete)
                     │ portes franchies ──► garde-fou du pas ──► executer ──► verifier (noul) ──► observation suivante
                     │ doute / pas risque / verification < 0,4 deux fois
                     ▼
                  Bonsai 2 : planifie avec les outils click/type/scroll_down/go_back/observe/done + judge_*
                     │
                  ses actions = etiquettes ──► trajectoires ──► entrainement du clone (DAgger)
```

## 2. Le computer use navigateur (`jev_clone/computer_use.py`)

* **Observation** : Playwright rend la page en un etat compact : URL, titre, arbre ARIA (texte), liste
  numerotee des elements interactifs visibles (`[3] input:email "email address"`, 40 au plus). Capture JPEG optionnelle
  pour la vision de Bonsai (non utilisee par Studio). Pas de pixels pour le clone : il decide sur le texte.
* **FastPolicy (clone)** : **une seule requete** `/v1/systemone` par pas : un `choice` sur l'action
  (click / type / scroll_down / go_back / done / escalate), un `noul` "objectif deja atteint ?", **un `noul` par element
  candidat** (24 au plus, pre-classes par mots communs avec l'objectif et les slots), un `choice` sur le "slot" a saisir.
  Le texte tape ne vient **jamais** du modele : il vient de slots structures fournis par la tache (`{"email": "..."}`).
* **Portes** (probabilite de tete et marge sur la deuxieme, pas l'entropie) : action p >= 0,5 et marge >= 0,2 ; cible
  p >= 0,5 et marge >= 0,15 ; slot p >= 0,5 et marge >= 0,2 ; `done` seulement si "objectif atteint" >= 0,6. Sinon, ou si le
  clone choisit `escalate` ou echoue, le pas part a `SlowPolicy`.
* **Garde-fou par pas** (`StepGuard`) : clic et saisie (plus raccourci et lancement d'application au bureau) juges avant
  execution, avec le meme verdict que les outils de Prophet (`jev_clone/guard.py`). Un pas rapide risque est confie a
  Bonsai ; une action risquee de Bonsai demande votre autorisation (refusee si personne ne peut repondre).
* **Verification** : apres chaque action rapide, un `noul` "l'action a-t-elle fait progresser l'objectif ?" ; deux
  resultats < 0,4 (ou actions en echec) consecutifs, ou verification impossible -> Bonsai reprend la main.
* **SlowPolicy (Bonsai)** : chat avec outils `click`, `type`, `scroll_down`, `go_back`, `observe`, `done` (avec
  `achieved`) **et** les `judge_*` du clone ; 6 tours d'outils par escalade, reflexion <= min(2 048, 60 % de sa generation) ;
  8 escalades au plus par tache, arret apres 2 echecs de Bonsai consecutifs. Apres un pas refuse, `done(achieved=true)`
  n'est un succes que si le clone le confirme sur un ecran relu. Capture jointe seulement avec `vision=True` (pas dans Studio).
* **Trajectoires** : chaque pas est journalise (Studio : `<donnees>/runs/trajectories.jsonl`) ;
  `training/make_from_trajectories.py` (via `trajectory_to_examples()`) transforme les pas escalades (actions de Bonsai)
  et les pas rapides verifies >= 0,8 en exemples `{state, questions, labels}` pour `training/train_lora_rlcd.py` ; une
  action refusee n'est jamais une etiquette. C'est la boucle DAgger : le professeur corrige l'eleve la ou l'eleve doute.

Tests (sans modele) : l'agent complet sur une page locale (saisie, clic, `done`), l'escalade vers un Bonsai factice qui
agit par outils, les portes, le garde-fou par pas, l'annulation, le DAgger (`tests/test_computer_use.py`,
`tests/test_fix_computer_use_voice.py`, `tests/test_fix_r2_cu_ui.py`, `tests/test_desktop_use.py`). Aucun test du computer
use ne tourne contre un vrai `llama-server`.

```bash
export JEV_S1_URL=http://127.0.0.1:8081 JEV_S2_URL=http://127.0.0.1:8080
pip install -e ".[agent]"      # playwright ; puis `playwright install chromium` si aucun Chromium local
python examples/browser_agent.py --url https://duckduckgo.com --goal "Search for Bonsai 2 27B" --slot query="Bonsai 2 27B" --headed
```

Dans Studio : reglage "Outil navigateur", desactive par defaut, et Playwright + Chromium a installer dans son
environnement Python (les installeurs ne les installent pas). Le bureau Windows (`jev_clone/desktop_use.py`, UI Automation)
suit la meme boucle, avec `press_keys` et `open_app` pour Bonsai.

## 3. Budget de latence (a mesurer)

Colonne clone : classifieur sur GPU. Dans Studio, c'est le cas seulement en priorite vitesse (RTX 4060 comme RTX 5060) ;
en priorites equilibre et contexte, le classifieur est sur CPU (un slot) : ajouter le prefill CPU de chaque etat neuf
(~0,5-1 s pour ~2 k tokens, estimation du planificateur). Les profils shell `rtx4060-8gb-*` le mettent sur GPU
(`JEV_NGL=99`).

| Etape | Clone 0,8-1,7B (GPU) | Bonsai 2 27B PTQ1_0 |
|---|---|---|
| observation DOM (40 elements) | 30-80 ms | - |
| decision (1 choice + 1 noul + 24 nouls + 1 choice, prefixe en cache) | 100-250 ms | 0,8-1,5 s (mode mono) |
| garde-fou du pas (3 questions, clic ou saisie) | ~100 ms | - |
| verification (1 noul) | 40-80 ms | - |
| **un pas "rapide"** | **~0,3-0,5 s + temps de chargement de la page** | - |
| escalade (planification, 2-4 appels d'outils, budget 2 048) | - | 5-20 s |
| appel `judge_*` depuis Bonsai | 50-250 ms | - |

La vision (capture jointe a Bonsai) couterait ~1 000 tokens de prefill par image : a reserver aux escalades ; Studio ne
l'active pas.

## 4. Ce que ce depot ne fait pas encore (et comment y aller)

* **Bureau par pixels** : le bureau passe par l'arbre UI Automation (Windows seulement, teste sur un bureau simule) ;
  rien pour les zones sans accessibilite (OCR, detection d'icones), rien pour macOS (AX) ni Linux. Un clone **vision**
  (Qwen3.5-0,8B/2B sont VL) pourrait decider sur capture, a ~2-3x la latence du DOM.
* **Memoire et planification longue** : Bonsai porte le plan (liste d'etapes) ; le clone execute chaque
  etape. Ajouter un `noul` "l'etape courante est-elle terminee ?" et un `choice` "quelle etape suivante ?"
  suffit pour enchainer.
* **Securite** : fait (garde-fou par pas, arrets obligatoires, confirmation humaine) ; a mesurer sur de vrais pas risques
  (le benchmark independant de Jev porte sur ce cas : classer readonly / destructive / privileged / exfiltration, 91,7 %).

## 5. Ecart avec Jev "SOTA" : evaluation honnete

| | Jev (TypeSafe) | Ce depot, niveau 0 (sans entrainement) | Ce depot apres A100 (phase 7 + trajectoires) |
|---|---|---|---|
| Mecanisme | architecture dediee, "parallel sampler", RLCD | lecture par grammaire sur un LM (Ternary-Bonsai zero-shot), questions sur un cache de prefixe partage | idem + poids entraines par regle de score propre + distillation Bonsai (jamais execute) |
| Calibration (ECE) | ~0,031 (MMLU, annonce) | non mesuree ici ; reference externe (reflex 4B) : 0,04-0,09 avant temperature, ~0,04 apres ; un 1,7B ternaire fera probablement moins bien | 0,035-0,045 attendu (decider 2B : 0,037 en domaine) |
| Precision hors domaine | forte (modele generaliste, ~10B ?) | non mesuree | bonne sur **votre** domaine vise, moyenne ailleurs |
| Options par question | 255 | 255 (lettres jusqu'a 26, noms au-dela) | idem |
| Latence | 70-500 ms en API | estimee : 100-350 ms + prefill (1,7B sur CPU, RTX 5060) ; 40-150 ms sur GPU | idem |
| Cout | 0,042 $/M tokens | 0 (electricite) | 0 |
| Donnees / confidentialite | cloud | local | local |

Conclusion : sur des taches **de votre domaine**, un clone 2B entraine avec Bonsai comme professeur pourrait
egaler ou depasser Jev (un modele specialise bat souvent un generaliste sur son terrain) ; sur des questions
arbitraires en zero-shot, Jev restera devant. Rien de cela n'est mesure. L'atout propre au montage local est la boucle
fermee : Bonsai enseigne, le clone accelere, tout reste sur votre machine.

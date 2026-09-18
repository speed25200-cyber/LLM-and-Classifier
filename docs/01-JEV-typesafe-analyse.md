# 01 - Jev (TypeSafe AI) : comment ca marche

> Etat des connaissances au 18 septembre 2026. Jev est un modele **ferme** (API uniquement, pas de
> poids, pas d'article technique). Tout ce qui est marque *[deduit]* vient de l'analyse de l'API, des
> chiffres publies et des reproductions ouvertes, pas d'une publication de TypeSafe.

## 1. Fiche d'identite

| | |
|---|---|
| Editeur | TypeSafe AI, San Francisco, fondee en 2024 par Diogo Almeida (ex-OpenAI, co-inventeur du RLHF), Erik Gafni et Sasha Sheng |
| Financement | ~40 M$ (seed mene par DCVC), valorisation ~200 M$ (Forbes) ; sortie de furtivite le 15 septembre 2026 |
| Produit | **Jev**, premier "System One Model" ; acces anticipe sur liste d'attente, `jev-latest` / `jev-1.13` |
| Surfaces | REST `POST https://api.typesafe.ai/v1/systemone`, SDK Python (`typesafe` v1.5.x) et TypeScript, "agent skill" installable ; relais Vercel AI Gateway, Backboard, Cloudflare AI |
| Prix | 0,042 $ / million de tokens d'entree ; sortie non facturee ("trop bon marche pour etre mesuree") ; ~0,0004 $ par decision en workflow |
| Latence | 70 a 500 ms de bout en bout (TypeSafe) ; p50 378-422 ms mesures par un benchmark independant |
| Limites | etat + questions <= ~64 k tokens ; etat + question la plus longue <= ~32 k tokens (~150 000 caracteres) ; `jev-1.13` : 250 000 tokens/s et 1 200 requetes/min |
| Positionnement | "20 a 200x plus rapide, 40 a 400x moins cher" qu'un workflow LLM equivalent ; "l'IA pour les machines, pas pour les humains" |

## 2. Ce que fait Jev (contrat public)

Jev **ne genere pas de texte**. Il prend un **etat** (une chaine ou un JSON : ticket, email, ligne de
log, coordonnees de jeu...) et une liste de **questions typees**, et renvoie pour chaque question une
**decision typee** accompagnee d'une **distribution de probabilites calibree**. Toutes les questions
sont evaluees **en parallele et isolement** contre le meme etat : ajouter une question ne change
presque pas la latence et ne peut pas modifier la reponse d'une autre.

Trois primitives :

| Primitive | Question | Reponse | Champs |
|---|---|---|---|
| **Noul** | "est-ce vrai ?" | probabilite que oui | `noul` (0-1) ; pas de champ confidence separe |
| **Choice** | "laquelle parmi N ?" (jusqu'a 255 options, chacune avec description optionnelle) | option gagnante | `choice`, `probabilities` (une par option), `confidence` |
| **Score** | "ou sur cette echelle ?" (2 a 10 niveaux ordonnes, decrits) | niveau pondere (peut tomber *entre* deux niveaux) | `score`, `probabilities` par niveau, legende, `confidence` |

`confidence` est **derivee de la distribution** (elevee si la masse est concentree, faible si elle est
plate) ; la formule exacte n'est pas publiee. TypeSafe insiste : *calibre ne veut pas dire correct*, une
decision a 0,95 peut etre fausse, mais 95 % des decisions a 0,95 sont justes.

Format de requete (reproduit a l'identique par les clones ouverts et par ce depot) :

```json
POST /v1/systemone
{
  "state": {"ticket": "My payouts have failed three times this week and nobody replied."},
  "questions": {
    "queue":    {"type": "choice", "instructions": "Which team should handle this?",
                 "criteria": {"payments": "payouts, refunds", "account": "login, 2FA", "other": null}},
    "escalate": {"type": "noul",   "instructions": "Should this be escalated to a manager?"},
    "urgency":  {"type": "score",  "instructions": "How urgent is this?",
                 "criteria": ["can wait a week", "handle today", "blocked right now"]}
  }
}
```
```json
{"answers": {
  "queue":    {"choice": "payments", "probabilities": {"payments": 0.91, "account": 0.06, "other": 0.03}, "confidence": 0.78},
  "escalate": {"noul": 0.42},
  "urgency":  {"score": 1.64, "probabilities": {"0": 0.08, "1": 0.20, "2": 0.72}, "confidence": 0.35}}}
```

## 3. Comment ca marche

### 3.1 Ce que TypeSafe dit
* Une **nouvelle architecture** (pas un transformeur autoregressif classique) avec un **"parallel sampler"** :
  toute la sortie est produite en **une seule requete parallele** au lieu d'un token a la fois.
* Un entrainement appele **RLCD, "Reinforcement Learning for Calibrated Decisions"** : la recompense est
  une **regle de score propre** (log-score ou Brier) sur des questions a reponse connue. Une regle de score
  propre n'est maximisee qu'en rapportant ses vraies croyances : c'est ce qui produit la calibration.
* Categorie "**System One**" (Kahneman) : le jugement rapide et intuitif d'un expert ("cinq secondes
  d'expertise a l'echelle machine"), par opposition au raisonnement lent ("System Two") des LLM.
* Benchmarks internes : ~67,8 % sur un benchmark de 4 workflows de production (comparable a GPT-5.6
  Terra), evalues par **accord avec des modeles frontiere** (GPT-6 Astra, Claude Fable 5.1), pas contre
  une verite terrain independante ; ECE ~0,031 sur 1 200 items MMLU ; pas de courbes de calibration publiees.

### 3.2 Ce que l'on peut deduire *[deduit]*
1. **Encodage partage de l'etat, branches isolees.** L'etat est encode une fois (cache KV / etat
   recurrent) ; chaque question est une branche qui voit l'etat mais pas les autres questions. C'est la
   seule facon d'avoir "N questions pour le prix d'une" et une independance stricte des reponses.
2. **Lecture directe, pas de decodage.** Au bout de chaque branche, le modele projette sa
   representation sur les etiquettes (options / niveaux / oui-non) et applique un softmax : une **tete
   de classification parallele** plutot qu'une generation. D'ou "rien a parser", "type-safe par
   construction", et un cout de sortie nul.
3. **Calibration entrainee, pas seulement post-hoc.** La regle de score propre en objectif + probablement
   une temperature. Le benchmark independant "jev-benchmark" (60 cas de classification du risque
   d'appels d'outils) observe 91,7 % de precision, **aucune erreur a confiance 1,0**, et 98 % de precision
   dans la tranche de confiance 0,9-1,0 : la confiance est exploitable pour router.
4. **Taille.** Latence 70-500 ms pour des etats de plusieurs milliers de tokens et un prix de 0,042 $/M
   tokens suggerent un modele de l'ordre de ~10 G parametres actifs (MoE soupconne). Non confirme.
5. **Demo Doom.** Jev joue a Doom en lisant l'**etat du jeu en texte structure** (~10 fois par seconde,
   ~7 $/heure d'inference) et en choisissant une action : la preuve que la boucle *etat -> decision
   typee* tient un budget temps reel.

### 3.3 Le patron d'architecture recommande par TypeSafe
```
ENTREE -> code deterministe -> Jev juge (questions etroites) -> code deterministe -> AGIR / REVOIR / RAISONNER (LLM)
```
*"Le code calcule. Jev juge. Le LLM raisonne et genere."* Le code garde le controle ; Jev fournit des
"if semantiques" ; la confiance sert de **porte** : au-dessus du seuil on automatise, en dessous on
escalade vers un modele de raisonnement ou un humain. Test d'aptitude en six questions : decider (pas
creer) ; espace de reponse definissable a l'avance ; un jugement focalise ; l'information tient dans
l'etat ; un expert tranche en quelques secondes ; le logiciel consomme directement le resultat.

## 4. Reproductions ouvertes (ce qu'elles nous apprennent)

| Projet | Backbone | Mecanisme | Resultats publies | Lecon pour ce depot |
|---|---|---|---|---|
| **decider** (Mapika) | Qwen3.5-2B-Base, fine-tune complet, 942 k exemples / 183 M tokens, 2,5 h sur un GH200 | un slot `Answer k: (` par question, logits restreints aux lettres A-J (jusqu'a 255 etiquettes mono-token), entropie croisee + Brier, temperature | 94 taches : 0,811 acc / ECE 0,037 en domaine ; 0,741 / 0,088 hors domaine ; 4 ms pour 3 questions | la recette d'entrainement (regle de score propre, permutations, options "other"), le serveur `/v1/systemone` compatible SDK TypeSafe, l'enseignant 27B pour etiqueter |
| **reflex** (kshetrajna12) | Qwen3.5-4B instruct, sans entrainement + temperature ; LoRA "RLCD-lite" optionnel | prefixe d'etat en cache, branches en batch (hybride GDN) ou masque 4D (attention pure), lecture des lettres | MMLU 1 200 items : 72 % acc, ECE 0,090 -> **0,039** avec une temperature (Jev : 0,031) ; ~100 ms pour 4 questions | **le niveau 0 fonctionne sans entrainement** ; une temperature suffit souvent ; 8 Go de VRAM suffisent |
| **qwen-rlcd** (shamazharikh) | Qwen3.5-0.8B-Base | "prefix-fork" : l'etat est prefill une fois, son cache (KV + etat conv/recurrent) est copie par branche | prototype inference | pourquoi un masque d'attention ne suffit pas sur les couches recurrentes (GatedDeltaNet) : il faut dupliquer l'etat |
| **system-one-gemma** (akash-kamat) | Gemma 3 270M + tete de score LoRA (2,6 M params) | une sequence par option, score scalaire, softmax entre options | 64,4 % acc, **ECE 0,047**, 15 min sur un T4 | meme un modele minuscule se calibre bien : la calibration vaut plus que la taille |
| **jevmlx / openjev**, **parallel-decisions** | n'importe quel modele MLX (Apple) | decodage contraint parallele par champ de schema | outil | l'idee de schema -> champs types -> une passe |
| **Qwen-2.5-1B-RLCD** (harshatheg, 18 sept. 2026 ; "Parallel Constrained Decoding") | Qwen2.5-1.5B-Instruct 4-bit, MLX (+ moteur torch), **sans entrainement** | contexte + descriptions du schema prefill une fois, cache KV diffuse a tous les champs, logits restreints aux **premiers tokens des noms d'options** (jusqu'a 255), marche token par token pour departager les options a prefixe commun | M4 Max : 4 champs 75 ms, 28 champs 270 ms, 255 choix 89 ms ; 5,6-7x plus rapide que generer le JSON, 100 % de syntaxe valide (le JSON genere hallucinait 2 champs) | c'est notre niveau 0 tel quel ; nous en reprenons la **lecture sur les noms d'options** (255 options, guillemet fermant comme terminateur) ; "RLCD" y est un nom, pas un entrainement |
| **jev-benchmark** (themsquared) | Jev reel | 60 cas de risque d'appel d'outil | 91,7 %, p50 ~420 ms, calibration exploitable | les cibles de calibration a viser |

## 5. Le clone de ce depot (`jev_clone/`)

* **Meme contrat** `POST /v1/systemone` (noul / choice / score, `state` texte ou JSON).
* **Meme mecanisme** : prefixe d'etat partage (cache de prompt de llama-server), une branche par
  question, lecture de la distribution du token suivant **restreinte par grammaire** aux etiquettes
  (`root ::= "A" | "B" | ...`), renormalisee (`post_sampling_probs`), temperature par primitive,
  `confidence = 1 - entropie normalisee`, permutations d'options optionnelles.
* **Aucune dependance a un framework GPU** : tout GGUF servi par llama-server convient, y compris
  Bonsai 2 27B (mode "mono") ou un petit Ternary-Bonsai / Qwen3.5 (mode "dual"), sur CUDA, Vulkan,
  ROCm, Metal ou CPU.
* **Differences avec Jev** : jusqu'a 255 options (lettres A..Z jusqu'a 26, puis lecture sur le nom de l'option
  avec resolution des prefixes partages, comme Qwen-2.5-1B-RLCD) ; pas de
  tete dediee ni de "parallel sampler" natif (les branches sont N requetes qui partagent un cache) ;
  la calibration doit etre **mesuree puis ajustee sur vos donnees** (`jev_clone/calibrate.py`) ; les
  garanties de Jev (ECE, robustesse) ne se transferent pas automatiquement.
* **Entrainement** (`training/`) : reduction "RLCD-lite" = minimisation d'une regle de score propre sur
  les logits restreints + distillation KL depuis Bonsai.

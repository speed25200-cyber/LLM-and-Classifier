# Protocole complet : fusion d'un clone de Jev (System One) et de Bonsai 2 27B (System Two) sur RTX 4060 ou materiel minimal

Version 1.0 - 18 septembre 2026. Documents associes : [01 Jev](01-JEV-typesafe-analyse.md),
[02 Bonsai 2 27B](02-BONSAI-2-27B-analyse.md), [03 Materiel](03-MATERIEL-profils.md),
[04 References](04-REFERENCES.md), [entrainement](../training/README.md). Code : `jev_clone/`, `scripts/`.

## 0. Resume executif

* **Jev** (TypeSafe AI, 15 sept. 2026) est un modele ferme qui ne genere pas de texte : il prend un etat
  et des questions typees (choice / score / noul) et renvoie des decisions avec des probabilites
  calibrees, en une passe parallele, 70-500 ms, 0,042 $/M tokens. Son mecanisme (etat encode une fois,
  branches isolees, lecture directe de la distribution, entrainement par regle de score propre "RLCD")
  est reproductible avec des modeles ouverts : plusieurs clones le prouvent (reflex, decider...).
* **Bonsai 2 27B** (PrismML, 17 sept. 2026) est Qwen3.8-27B compresse en poids ternaires QAT :
  **5,93 Go**, 98,2 % des benchmarks du FP16, reflexion, outils, vision, 262 k de contexte, Apache 2.0,
  execute par un fork de llama.cpp. C'est le meilleur "cerveau lent" qui tienne dans 8 Go de VRAM.
* **La fusion** = un routeur a deux vitesses sur **un seul GPU de 8 Go** :
  1. **System One (clone de Jev)** : un petit GGUF (Ternary-Bonsai-1.7B, 0,37 Go, ou un Qwen3.5 entraine)
     lit les probabilites sur les options de chaque question via llama-server (grammaire + cache de
     prefixe). ~50-150 ms par decision, jamais de texte libre.
  2. **Porte de confiance** : au-dessus d'un seuil calibre, le code agit sur la decision ; en dessous, ou
     si une question "meta" l'exige (risque, besoin de raisonnement), on escalade.
  3. **System Two (Bonsai 2 27B)** : raisonne (budget de reflexion proportionnel au risque), genere,
     appelle des outils, lit des images ; System One peut verifier sa sortie.
  4. **Boucle** : chaque decision est journalisee ; Bonsai re-etiquette les cas escalades ; le clone est
     re-calibre / re-entraine (distillation). Le systeme s'ameliore sans donnees externes.
* **Exigences** : RTX 4060 8 Go (profil recommande) ; fonctionne aussi sans GPU (8-16 Go de RAM, modeles
  8B 1-bit + 1.7B) ou sur un GPU 4 Go. Tout est local, aucune cle d'API.
* **Etat du depot** : le clone (`jev_clone/`), son serveur `/v1/systemone`, le routeur de fusion, la
  calibration et la distillation sont implementes et testes (23 tests, dont un test d'integration
  contre un vrai `llama-server` du fork PrismML). Les scripts d'installation / lancement / profils sont
  fournis. Le script d'entrainement est fourni mais **n'a pas pu etre execute** ici (pas de GPU).
  Les debits RTX 4060 sont des **estimations** a confirmer avec `scripts/bench.sh`.

```
                 +----------------------------------------------------------------------+
   etat (texte,  |  RTX 4060 8 Go                                                       |
   JSON, image)  |                                                                      |
   ---------->   |  [S1] llama-server :8081  clone Jev (1.7B ternaire, 0,4 Go, 4 slots) |
                 |        etat -> prefixe cache -> N branches -> P(options) calibrees   |
                 |                      |                                               |
                 |          confiance >= seuil ?  ---- oui ---->  decision typee -> code|
                 |                      | non / risque / "needs_reasoning"              |
                 |                      v                                               |
                 |  [S2] llama-server :8080  Bonsai 2 27B PTQ1_0 (5,9 Go, KV q4, 8k)    |
                 |        reflexion budgetee -> reponse / JSON / outils / vision        |
                 |                      |                                               |
                 |          (option) S1 verifie la coherence de la reponse S2          |
                 +----------------------------------------------------------------------+
                                        |
                          ledger.jsonl -> distill.py (Bonsai enseignant) -> calibrate.py / train_lora_rlcd.py
```

## 1. Principes de conception

1. **Le code calcule, le clone juge, Bonsai raisonne.** Le code applicatif garde le controle ; System One
   fournit des "if semantiques" avec une probabilite ; System Two n'est appele que quand le doute ou le
   risque le justifie. C'est le patron recommande par TypeSafe, applique a deux modeles locaux.
2. **Une seule API** : les deux niveaux sont des `llama-server` (fork PrismML). Le clone et Bonsai
   partagent les binaires, le format GGUF, le cache de prompt, le JSON contraint.
3. **Sortie typee par construction** : System One ne peut renvoyer qu'une option listee (grammaire) ;
   System Two, quand on lui demande des etiquettes, repond en JSON contraint par schema.
4. **Calibration mesuree, pas supposee** : ECE, Brier, precision selective sur *vos* donnees, seuils de
   porte fixes pour une precision cible (ex. 95 %), et re-mesures periodiques.
5. **Degradation gracieuse** : chaque profil a un repli (clone sur CPU, Bonsai 1-bit, mode mono) ; les
   deux serveurs sont independants ; si Bonsai est absent, System One repond seul.

## 2. Composants

| Composant | Role | Implementation | Taille | Port |
|---|---|---|---|---|
| (optionnel) A100 80 Go Colab | usine : distillation a grande echelle, fine-tuning complet du clone, export GGUF, banc d'evaluation | `colab/jev_bonsai_a100.ipynb`, profil `a100-80gb.env` | - | - |
| Bonsai 2 27B `PTQ1_0` (+ `mmproj` Q8_0 optionnel) | System Two : raisonnement, generation, outils, vision, enseignant | `llama-server` fork PrismML `prism-b10683` | 5,93 Go (+0,63) | 8080 |
| Ternary-Bonsai-1.7B `Q2_0_g64` (niveau 0) ou clone entraine Qwen3.5-0.8B/2B (niveau 1) | System One : decisions typees calibrees | second `llama-server`, 4 slots, `--reasoning-budget 0` | 0,37 Go / 0,9-2 Go | 8081 |
| `jev_clone` (Python) | contrat `/v1/systemone`, lecture par grammaire, temperature, confiance, permutations, routeur de fusion, calibration, distillation, ledger | `jev serve` (FastAPI), `jev decide`, `jev bench` | - | 8008 |
| `scripts/profiles/*.env` | choix des modeles, contexte, offload, KV4, budgets | `setup.sh`, `start_bonsai.sh`, `start_jev_clone.sh`, `bench.sh` | - | - |
| `training/` | RLCD-lite : LoRA/QLoRA + regle de score propre + KL enseignant, export GGUF | `train_lora_rlcd.py` | - | - |
| `jev_clone/tools.py`, `computer_use.py` | Bonsai consulte le clone (outils `judge_*`), boucle d'agent, agent navigateur a deux vitesses | `AgentLoop`, `ComputerUseAgent`, `examples/browser_agent.py` | - | - |
| `jev_clone/guided.py` | System One dans la boucle de decodage de Bonsai : reflexion adaptative, meilleur de N, reponse verifiee | `GuidedGenerator` | - | - |
| (optionnel) adaptateur OrcaBonsai | ablation de refus a l'execution sur Bonsai 2 (poids inchanges), echelle reglable par requete | `scripts/fetch_orcabonsai.sh`, profil `rtx4060-8gb-orcabonsai.env`, `BONSAI_LORA` | 9,7 Mo | - |

## 3. Phases

### Phase 0 - Prerequis et choix du profil (30 min)

* **Materiel** : voir [03 Materiel](03-MATERIEL-profils.md). Arbre de decision :
  * GPU NVIDIA 8 Go (RTX 4060 / 4060 Laptop / 3070 / 3060 Ti...) -> `rtx4060-8gb-qualite.env`
    (Bonsai 2) ; si vous voulez de la vitesse ou un clone 4B sur GPU -> `rtx4060-8gb-vitesse.env` ;
    si vous ne voulez qu'un seul serveur -> `rtx4060-8gb-mono.env`.
  * GPU 12 Go -> `gpu-12gb.env` ; 16 Go -> `gpu-16gb.env` ; 4 Go -> `gpu-4gb.env` ; pas de GPU -> `minimal-cpu.env`.
  * AMD/Intel (Vulkan) : Bonsai 2 n'a pas de noyaux Vulkan -> profil `vitesse` (Q1_0) ou `ternary` v1.
* **Logiciel** : Linux (Ubuntu 22.04+) ou Windows 11 ; pilote NVIDIA >= 550 ; Python 3.10+ ; 15 Go de disque
  (binaires 0,1 Go, Bonsai 2 5,9 Go + mmproj 0,6, clone 0,4 Go, marge pour les runs).
* **Verification** : `nvidia-smi` (VRAM libre >= 7,5 Gio avant lancement : fermer ce qui utilise le GPU).

### Phase 1 - Installation (20 min + telechargement)

```bash
git clone <ce depot> && cd LLM-and-Classifier
PROFILE=scripts/profiles/rtx4060-8gb-qualite.env ./scripts/setup.sh
#  1. venv + `pip install -e ".[serve,dev]"`   2. binaires llama.cpp PrismML (CUDA 12.4/12.8/13.3 auto)
#  3. prism-ml/Ternary-Bonsai-2-27B-gguf  *-PTQ1_0.gguf (+ *mmproj-Q8_0.gguf)   4. prism-ml/Ternary-Bonsai-1.7B-gguf *-Q2_0_g64.gguf
source .venv/bin/activate && python -m pytest -q          # 21 tests hors ligne doivent passer
```
Windows : `setup.ps1` du depot `PrismML-Eng/Bonsai-demo` (memes binaires et poids), puis
`scripts/windows/start_bonsai.ps1` / `start_jev_clone.ps1`, `pip install -e .`.

**Critere de sortie** : `bin/cuda/llama-server --version` affiche `build 10683` ; les deux GGUF sont dans `models/`.

### Phase 2 - Bonsai 2 en System Two (15 min)

```bash
./scripts/start_bonsai.sh            # ou PROFILE=... ./scripts/start_bonsai.sh --reasoning-budget 1024
curl -s localhost:8080/props | jq '.total_slots, .default_generation_settings.n_ctx'
curl -s localhost:8080/v1/chat/completions -H 'content-type: application/json' -d '{
  "messages":[{"role":"user","content":"Implement binary search in Python."}],"max_tokens":300}' | jq '.timings.predicted_per_second, .choices[0].message.reasoning_content[:200]'
```
Drapeaux utilises par le profil et pourquoi :
`-ngl 99` (tout sur GPU) ; `-c 8192` (contexte explicite : jamais `-c 0`) ; `-fa on` ; `--cache-type-k/v q4_0`
(KV 4-bit, ~0,15 Gio a 8 k) ; `--mmproj ... --no-mmproj-offload` (vision possible, projecteur en RAM) ;
`--jinja` (appels d'outils natifs) ; `--temp 1.0 --top-p 0.95 --top-k 20` (defauts Qwen3.8) ;
`--reasoning-budget 2048` (plafond de reflexion par defaut, modulable par requete via
`thinking_budget_tokens`) ; `-np 1` (un slot : la generation est la ressource rare) ; `--cache-ram 2048`
et `--ctx-checkpoints 8` (reutilisation de prefixe, y compris sur les couches recurrentes).

**Critere de sortie** : `predicted_per_second` >= 25 tok/s sur RTX 4060 (estimation ; noter la valeur
reelle dans `runs/bench.md`) ; VRAM occupee <= 7,0 Gio (`nvidia-smi`).

**Option : variante sans refus (OrcaBonsai).** Memes poids Bonsai 2 + un adaptateur LoRA de rang 1 (9,7 Mo)
qui retire la direction de refus dans le graphe : `./scripts/fetch_orcabonsai.sh` puis le profil
`rtx4060-8gb-orcabonsai.env`. Elle supprime le sur-refus (25 % -> 0 % sur JBB-benign) qui bloque un agent ;
en contrepartie le garde-fou passe entierement dans System One, la porte a risque et l'executeur (voir
[08 OrcaBonsai](08-ORCABONSAI-UNCENSORED.md)). Mesurer avant d'adopter : refus sur vos invites benignes a
echelle 0 et 1, jev-benchmark en mode mono aux deux echelles.

### Phase 3 - Clone de Jev, niveau 0 : sans entrainement (15 min)

```bash
./scripts/start_jev_clone.sh          # llama-server :8081, Ternary-Bonsai-1.7B, 4 slots
jev decide --server http://127.0.0.1:8081 \
  --state "Bonjour, j'ai ete debite deux fois pour la commande A-104, merci de rembourser le doublon." \
  --choice team=billing,technical,sales,other --instruction "team=Which team should handle this?" \
  --noul "refund=Does the customer ask for a refund?" \
  --score "frustration=calm|annoyed|very frustrated" --instruction "frustration=How frustrated is the customer?"
jev bench --server http://127.0.0.1:8081 --n 30      # latence p50/p95, 3 questions, cache chaud
```
Mecanisme (identique a Jev dans son contrat, different dans sa mise en oeuvre) :
1. Le prefixe `system + # State` est tokenise une fois ; llama-server le garde en cache (`cache_prompt`).
2. Chaque question devient une branche `# Question / # Options / Respond with only the letter` suivie de
   l'amorce assistant (avec `<think></think>` vide pour couper la reflexion des Qwen3.x).
3. La premiere branche est envoyee seule (elle remplit le cache), les autres en parallele sur les slots.
4. Une grammaire `root ::= "A" | "B" | ...` masque tout autre token ; `post_sampling_probs` renvoie la
   distribution renormalisee : c'est la reponse, aucune generation.
5. Temperature par primitive (phase 4), `confidence = 1 - entropie normalisee`, permutations d'options
   (`"permutations": 3`) pour attenuer le biais de position des lettres.

Mode **mono** (profil `rtx4060-8gb-mono.env`) : meme code, `--server http://127.0.0.1:8080` : Bonsai 2 juge
lui-meme. Plus juste (27B), 3-10x plus lent, aucune VRAM en plus. Utile comme reference de qualite et
comme enseignant (phase 6).

**Critere de sortie** : `jev bench` p50 <= 150 ms (clone 1.7B GPU) ; les probabilites somment a 1 ; sur
20 exemples ecrits a la main, >= 80 % de bonnes reponses sur des questions "faciles" (routage evident).

### Phase 4 - Calibration : rendre les pourcentages honnetes (1-2 h)

1. Constituer `data/val.jsonl` : 300 a 1 000 exemples **de votre domaine**, etiquetes a la main ou par
   Bonsai en mode "think" (phase 6), format `{"state": ..., "questions": {...}, "labels": {qid: ...}}`.
2. `python -m jev_clone.calibrate --server http://127.0.0.1:8081 --data data/val.jsonl --out runs/calibration.json --target-precision 0.95`
   * ajuste **une temperature par primitive** (noul / choice / score) en minimisant la NLL ;
   * calcule accuracy, NLL, Brier, **ECE** (15 bacs) avant / apres ;
   * fixe, **par question**, le seuil de confiance tel que la precision des cas au-dessus du seuil >= 95 %.
3. Servir avec `JEV_CALIBRATION=runs/calibration.json jev serve`.

**Cibles** (reflex obtient ECE 0,039 sans entrainement sur un 4B ; Jev annonce 0,031) : ECE <= 0,06 au
niveau 0, <= 0,04 apres entrainement ; la precision selective au seuil doit etre >= la cible sur un jeu
tenu a l'ecart. Si une question ne trouve aucun seuil (precision cible inatteignable), elle **escalade
toujours** : c'est un signal pour la phase 7, pas un echec de la fusion.

### Phase 5 - Fusion a l'execution (1 h)

```bash
export JEV_S1_URL=http://127.0.0.1:8081 JEV_S2_URL=http://127.0.0.1:8080 JEV_CALIBRATION=runs/calibration.json
jev serve --port 8008         # POST /v1/systemone (compatible SDK TypeSafe via TYPESAFE_BASE_URL) et POST /v1/fusion/decide
python examples/ticket_routing.py      # routage de tickets avec escalade
python examples/agent_loop.py --hz 10  # boucle d'agent temps reel (style Doom) sur etat structure
```
Politique de porte (`jev_clone/fusion.py: GatePolicy`) :
* `thresholds[qid]` (de la calibration) sinon `default_threshold` (0,85) ;
* `always_escalate_if={"needs_human": True}` : une question "meta" force l'escalade quelle que soit la confiance ;
* `risk_question="risk"` : une question *score* de risque choisit le budget de reflexion de Bonsai
  (`budgets=(0, 512, 2048, 8192)`) ; risque nul -> Bonsai repond sans reflechir (rapide) ;
* `verify_with_s1=True` : apres l'escalade, System One repond a "la reponse est-elle coherente avec
  l'etat ?" (garde-fou a ~50 ms) ;
* tout est ecrit dans `runs/ledger.jsonl` (etat, questions, probabilites S1, chemin, sortie S2, latence).

Questions "meta" a ajouter a **chaque** schema applicatif (c'est ce qui fait la fusion) :
```json
"needs_reasoning": {"type": "noul",  "instructions": "Does answering correctly require multi-step reasoning, calculation or external knowledge?"},
"risk":            {"type": "score", "instructions": "How costly would a wrong automated decision be?",
                    "criteria": ["harmless / reversible", "annoying", "costly", "dangerous or irreversible"]}
```
Schemas prets a l'emploi dans `jev_clone/presets.py` (routage, competences, reranking, garde-fous, extraction,
tri de tickets) ; economies mesurees par `python -m jev_clone.ledger_report runs/ledger.jsonl` ; historique de
decisions -> exemples par `training/make_from_history.py` ([10 Cas d'usage](10-CAS-D-USAGE.md)).
Patrons d'usage couverts : classification / routage (choice + porte), detection (noul), notation
(score), moderation et securite d'appels d'outils (noul + risque -> Bonsai), reranking / retrieval (score
par document, tous en parallele), agent temps reel (choice d'action a 10 Hz, Bonsai en arriere-plan
pour la planification), verification de sorties LLM (noul de coherence).

**Critere de sortie** : sur 100 requetes representatives, taux d'escalade entre 10 et 40 % ; latence p50
du chemin S1 <= 200 ms ; latence p50 du chemin escalade <= 5 s a budget 2048 ; precision globale >= celle
de Bonsai seul (mesuree en phase 6) a un cout moyen 5-10x moindre.

### Phase 6 - Distillation : Bonsai enseigne le clone (2-6 h de calcul)

```bash
# etats non etiquetes de votre domaine (logs, tickets, evenements...) -> data/states.jsonl {"state", "questions"}
python -m jev_clone.distill --bonsai http://127.0.0.1:8080 --in data/states.jsonl --out data/labeled.jsonl \
    --mode soft --think-if-below 0.85 --budget 2048
```
* **soft** : Bonsai joue System One (lecture par grammaire) : distributions completes ("teacher_probs"),
  ~0,3-1,5 s par etat sur RTX 4060 (prefill 300-400 tok/s en PTQ1_0 ; ~2x plus rapide en PQ2_0).
* **think** : pour les etats ou la distribution soft est plate (< 0,85), Bonsai reflechit puis repond en
  JSON contraint (`response_format: json_schema`) : etiquette dure de meilleure qualite, ~5-20 s par etat.
* Melanger avec les cas escalades du ledger (ce sont, par construction, les plus utiles).

Volumes : 2 000 etats suffisent pour la calibration + un premier LoRA de domaine ; 20 000+ pour un clone
robuste. A ~1 s/etat en soft, 20 000 etats = ~6 h de GPU, la nuit.

### Phase 7 - Entrainement RLCD-lite du clone (une nuit sur RTX 4060)

Voir [training/README.md](../training/README.md). Resume : base `Qwen/Qwen3.5-0.8B-Base` (LoRA bf16) ou
`2B-Base` (QLoRA 4-bit) ; melange 60 % jeux publics de decision / 30 % distille / 10 % manuel ;
objectif = NLL ou Brier sur les logits restreints aux etiquettes (regle de score propre = reduction
exacte de l'objectif RLCD quand la politique emet la distribution) + KL vers `teacher_probs` ;
permutations d'options et options "other" ; temperature ajustee sur validation ; export
`convert_hf_to_gguf.py` -> `Q8_0` -> `JEV_GGUF=... ./scripts/start_jev_clone.sh`.

Ordres de grandeur (a valider) : 0,8B, 100 k branches x ~350 tokens = 35 M tokens, ~2-3 h ; 2B en QLoRA,
~6-8 h. Comparaison publique : decider (2B, 183 M tokens) = 2,5 h sur un GH200 (~6-8x une 4060).

**Critere de sortie** : sur le jeu tenu a l'ecart, +10 points d'accuracy vs niveau 0 sur les questions
maison, ECE <= 0,04, taux d'escalade en baisse d'au moins un tiers a precision egale.

### Phases 6-7 bis - Variante "deux machines" : A100 80 Go (Colab) = usine, RTX 4060 = production

Si une A100 80 Go est disponible (Colab Pro/Pro+), y deplacer les phases lourdes et ne rapatrier sur la
4060 que des fichiers : `colab/jev_bonsai_a100.ipynb` enchaine tout, details dans [05 Colab A100](05-COLAB-A100.md).

| Etape | Sur la 4060 (8 Go) | Sur l'A100 (80 Go) |
|---|---|---|
| Bonsai enseignant | PTQ1_0, 1 slot, prefill ~350 tok/s : ~1-1,5 s/etat | **PQ2_0**, 4 slots, prefill ~1 300 tok/s : **~0,4 s/etat** (20 000 etats en ~2-3 h) |
| Modele du clone | Qwen3.5-0.8B LoRA ou 2B QLoRA | **Qwen3.5-2B (ou 4B) fine-tuning complet bf16** (`--full`, recette decider), batch 16-32, sequences 1 536-2 048 |
| Duree d'entrainement | 0,8B : 2-3 h pour 35 M tokens | 2B : ~4-6 h pour 180 M tokens ; 0,8B : ~1,5 h |
| Evaluation / calibration | sur le GGUF final | sur le GGUF final, servi par le meme fork llama.cpp, avant export |
| Artefacts rapatries | - | `jev-clone-Q8_0.gguf` (2B : 2,1 Go ; 0,8B : 0,9 Go) ou `Q4_K_M`, `calibration.json`, dossier `merged/` pour re-entrainer |

Regle de choix du clone final selon le profil 4060 : profil **qualite** (Bonsai 2 en VRAM) -> clone 0,8B
Q8_0 (0,9 Go) ; profil **vitesse** (Bonsai-27B 1-bit) -> clone 2B Q8_0 (2,1 Go) ou 4B Q4_K_M (2,5 Go).
Le notebook sert aussi de banc : les deux serveurs y tournent, `examples/ticket_routing.py` y mesure le
taux d'escalade et les latences de la fusion complete avant la mise en production sur la 4060.

### Phase 8 - Boucle d'amelioration continue (mensuelle)

1. `runs/ledger.jsonl` -> extraire les cas escalades et un echantillon des cas automatises.
2. Re-etiqueter par Bonsai (`distill.py --mode think`) + relecture humaine d'un echantillon (100 cas).
3. Re-calibrer (`calibrate.py`) : si l'ECE d'une question depasse 0,08 ou si la precision selective
   tombe sous la cible, relever son seuil (elle escalade davantage) et l'ajouter au prochain entrainement.
4. Re-entrainer le clone (`train.sh delta`-style : continuer depuis l'adaptateur precedent avec les
   nouvelles donnees + un echantillon de rejeu) ; A/B sur le jeu tenu a l'ecart ; promouvoir si mieux.
5. Suivre : taux d'escalade, latence p50/p95 par chemin, ECE par question, accord S1/S2 sur les cas
   escalades (si S1 avait raison a >= 95 % sur une question, baisser son seuil).

### Phase 10 - Agent a deux vitesses et computer use (apres la phase 5)

Voir [06 Agent et computer use](06-AGENT-COMPUTER-USE.md). Les deux sens de la fusion sont codes :
* **Bonsai consulte le clone** : `SystemOneToolbox` expose `judge_choice / judge_noul / judge_score /
  judge_rank / judge_batch` comme outils OpenAI ; `AgentLoop` execute la boucle d'appels d'outils de
  Bonsai (outils du clone + outils applicatifs), journal compris.
* **Computer use navigateur** : `ComputerUseAgent` = observation Playwright (DOM/ARIA) -> `FastPolicy`
  (clone : action + element cible + slot en une requete) -> execution -> verification (noul) ->
  escalade `SlowPolicy` (Bonsai avec outils click/type/scroll/back/done + judge_*). Les trajectoires
  deviennent des exemples d'entrainement (`trajectory_to_examples`, boucle DAgger).

```bash
pip install -e ".[agent]"
python examples/browser_agent.py --url https://example.com --goal "Open the 'More information' link" --headed
```
**Critere de sortie** : sur 20 taches navigateur simples (formulaire, recherche, navigation), >= 80 % de
pas decides par le clone, taux de reussite >= celui de Bonsai seul, temps par tache divise par >= 3.

### Phase 11 - Mesurer contre Jev et viser mieux

[07 Battre Jev](07-BATTRE-JEV.md) et `eval/SCOREBOARD.md`. A chaque niveau du clone :
```bash
python -m eval.jev_benchmark --server http://127.0.0.1:8081 --name clone-l0 --repeat 5 && python -m eval.jev_benchmark --analyze
python -m eval.mmlu_ece --server http://127.0.0.1:8081 --n 1200          # Colab
```
Puis garanties formelles sur vos donnees (ensembles conformes, porte a risque controle : `jev_clone/conformal.py`).
**Critere de sortie** : colonnes du tableau de bord remplies ; sur jev-benchmark, accuracy >= 93,3 % et 0 erreur a
confiance >= 0,9 sur 5 runs ; MMLU-1200 ECE <= 0,03 ; p50 < 100 ms.

### Phase 12 - Fusion au niveau de l'inference : System One pilote la generation de Bonsai

[09 Fusion profonde](09-FUSION-PROFONDE.md). `jev_clone/guided.py` : `think_adaptive` (le clone arrete la
reflexion de Bonsai des que la reponse est determinee ou que le raisonnement tourne en rond), `best_of_n`
(N candidats classes par le clone en une passe), `verified` (verification puis seconde tentative).
```python
from jev_clone.guided import GuidedGenerator
g = GuidedGenerator(LlamaCppBackend(JEV_S2_URL, max_workers=1), s1_engine, check_every=192, max_think=4096)
r = g.think_adaptive([{"role": "user", "content": "..."}]); print(r.stopped_by, r.think_tokens, r.answer)
```
**Critere de sortie** : sur 50 questions maths/code, -40 % de tokens de reflexion a exactitude egale par
rapport a `--reasoning-budget 2048` ; `best_of_n` (n=4) >= reflexion complete sur les questions courtes.

### Phase 13 - Prophet : l'agent general, et son jeu de donnees fabrique a partir de zero

[11 Prophet](11-PROPHET.md). `jev prophet --workspace ~/prophet --browser` : Bonsai garde toujours la pleine
autonomie (code, shell, Python, web, memoire, outils qu'il cree lui-meme) ; le clone juge a chaque tour des
proprietes (reponse directe possible, budget de reflexion, risque, outils pertinents, risque de chaque action)
et ne restreint jamais ce que l'agent peut faire.
Donnees : 82 graines manuscrites -> synthese par Bonsai (`training/make_synthetic_prophet.py`) -> etiquetage
(`distill.py`) -> calibration / entrainement ; puis le journal de Prophet alimente les cycles suivants.
**Critere de sortie** : 10 demandes de creation d'application executees de bout en bout (fichiers, tests, lancement)
avec >= 8 reussites ; 0 commande a risque >= 2 executee sans confirmation ; pre-traitement p50 <= 250 ms.

### Phase 9 - Exploitation

* **Ordre de demarrage** : Bonsai d'abord (gros allocataire), puis le clone, puis `jev serve`. Verifier
  `nvidia-smi` apres chaque etape ; si le clone echoue en OOM : `JEV_NGL=0`.
* **Concurrence** : Bonsai `-np 1` (2 sur 12 Go+) ; clone `-np 4` ; `jev_clone` envoie les branches en
  parallele (`max_workers` = slots).
* **Securite** : les serveurs ecoutent sur `127.0.0.1` ; exposer uniquement `jev serve` derriere un
  reverse proxy authentifie ; les etats peuvent contenir des donnees personnelles -> le ledger est local
  et a purger / anonymiser selon votre politique.
* **Sauvegardes** : `runs/calibration.json`, adaptateurs LoRA, GGUF exportes, `data/*.jsonl` ; les poids
  Bonsai se retelechargent.
* **Mises a jour** : suivre les releases du fork PrismML (`LLAMA_RELEASE_TAG`) ; Bonsai 2 vise
  l'upstreaming (PR llama.cpp #27779) ; un drafter DSpark pour Bonsai 2 n'existe pas encore
  (`BONSAI_SPECULATIVE=1` ne s'applique qu'aux 27B v1).

## 4. Criteres d'acceptation (synthese)

| Phase | Mesure | Cible RTX 4060 | Outil |
|---|---|---|---|
| 1 | tests hors ligne | 21/21 | `pytest` |
| 2 | generation Bonsai 2 PTQ1_0 | >= 25 tok/s ; VRAM <= 7,0 Gio | `scripts/bench.sh`, `nvidia-smi` |
| 3 | latence decision S1 (3 questions, cache chaud) | p50 <= 150 ms | `jev bench` |
| 4 | calibration niveau 0 | ECE <= 0,06 ; seuils a precision 95 % | `calibrate.py` |
| 5 | fusion | escalade 10-40 % ; p50 S1 <= 200 ms ; p50 S2 <= 5 s | ledger |
| 6 | distillation | >= 2 000 etats etiquetes ; accord soft/think >= 85 % sur cas surs | `distill.py` |
| 7 | clone entraine | +10 pts accuracy ; ECE <= 0,04 ; escalade -33 % | `train_lora_rlcd.py` |
| 6-7 bis | usine A100 | 20 000 etats distilles en < 4 h ; 2B entraine en < 8 h ; GGUF + calibration dans Drive | `colab/jev_bonsai_a100.ipynb` |
| 10 | agent navigateur | >= 80 % de pas rapides ; reussite >= Bonsai seul ; temps / 3 | `examples/browser_agent.py`, `runs/trajectories.jsonl` |
| 13 | Prophet | 8/10 applications creees de bout en bout ; 0 action risquee sans confirmation ; reponse directe p50 <= 3 s | `jev prophet`, `.prophet/ledger.jsonl` |
| 12 | generation guidee | -40 % de tokens de reflexion a exactitude egale | `guided.py` |
| 11 | tableau de bord vs Jev | jev-benchmark >= 93,3 %, 0 erreur a conf >= 0,9 ; MMLU ECE <= 0,03 ; p50 < 100 ms ; couverture conforme verifiee | `eval/`, `conformal.py` |
| 8 | boucle | ECE par question <= 0,08 ; derive detectee sous 1 mois | ledger + `calibrate.py` |

## 5. Risques et parades

| Risque | Effet | Parade |
|---|---|---|
| Bonsai 2 charge sur llama.cpp mainline | PTQ1_0/PQ2_0 refuses (sans danger) ; **la bande Q2_0 charge et produit du charabia** | n'utiliser que les binaires du fork (`setup.sh`) ; ne jamais telecharger `*-Q2_0-prism-fork-required.gguf` |
| VRAM 8 Go trop juste (bureau Windows, navigateur) | OOM au demarrage du clone | ordre des parades dans [03 Materiel](03-MATERIEL-profils.md#4) ; `JEV_NGL=0` ; profil `vitesse` ou `mono` |
| Prefill lent en PTQ1_0 (~2x plus lent que PQ2_0) | distillation soft et longs etats lents | PQ2_0 sur 12 Go+ ; sur 8 Go, garder PTQ1_0 (seul qui laisse de la place) et distiller la nuit |
| Calibration hors domaine | seuils faux, automatisation de decisions fausses | ne jamais reutiliser une calibration d'un autre domaine ; recalibrer a chaque changement de schema |
| Biais de position des lettres (A > B) sur modele non entraine | distributions faussees | `permutations >= 2` ; entrainement avec permutations (phase 7) |
| 26 options max par question | schemas larges impossibles | decouper en questions hierarchiques (famille puis sous-type) ; roadmap : etiquettes bi-lettres |
| Couches recurrentes (GatedDeltaNet) et cache partiel | reutilisation de prefixe imparfaite -> re-prefill | `--ctx-checkpoints 8`, prefixe strictement identique (le client garantit l'identite octet a octet) |
| Bonsai 2 faible en agentique long-horizon (~75 % du FP16) | plans longs fragiles | System One decoupe en decisions courtes ; limiter les boucles d'outils (10 tours) ; verification S1 |
| Questions "meta" mal posees | escalade trop rare / trop frequente | mesurer le taux d'escalade par question ; ajuster seuils et formulations sur le ledger |
| Licences | Bonsai/Qwen : Apache 2.0 ; fork llama.cpp : MIT ; ce depot : Apache 2.0 | verifier les jeux publics utilises a l'entrainement |

## 6. Feuille de route (au-dela de la v1)

1. **Tete de decision dediee** (255 options, comme decider) sur le clone entraine : etiquettes mono-token
   bi-lettres ; necessite un export GGUF sans changement de format (la tete reste le LM head).
2. **Clone multimodal** : Qwen3.5-0.8B/2B sont VL ; mettre l'image dans l'etat (`{"type":"image",...}`)
   comme reflex ; sur 8 Go, garder l'encodeur vision en RAM.
3. **DSpark pour Bonsai 2** des publication du drafter (x1,4-2,4 en generation sur CUDA).
4. **Fusion "in-process"** : remplacer les N requetes HTTP par une seule passe batched (llama.cpp
   `llama_batch` multi-sequence + copie d'etat) via llama-cpp-python : divise la latence S1 par ~2.
5. **Bonsai comme backbone du clone** : impossible localement (QAT proprietaire) ; la voie est la
   distillation (phases 6-7). Si PrismML publie des Bonsai 2 en 1.7B/4B, les substituer au clone niveau 0.
6. **WebGPU** : reflex montre un clone 0.8B dans le navigateur ; pertinent pour un poste sans NVIDIA.

## 7. Annexe A - Commandes completes RTX 4060 (copier-coller)

```bash
# --- installation ---
PROFILE=scripts/profiles/rtx4060-8gb-qualite.env ./scripts/setup.sh && source .venv/bin/activate && pytest -q
# --- terminal 1 : System Two ---
PROFILE=scripts/profiles/rtx4060-8gb-qualite.env ./scripts/start_bonsai.sh
# --- terminal 2 : System One ---
PROFILE=scripts/profiles/rtx4060-8gb-qualite.env ./scripts/start_jev_clone.sh        # OOM ? -> JEV_NGL=0 ...
# --- terminal 3 : mesures, calibration, service ---
PROFILE=scripts/profiles/rtx4060-8gb-qualite.env ./scripts/bench.sh
python -m jev_clone.calibrate --server http://127.0.0.1:8081 --data data/val.jsonl --out runs/calibration.json
export JEV_S1_URL=http://127.0.0.1:8081 JEV_S2_URL=http://127.0.0.1:8080 JEV_CALIBRATION=runs/calibration.json
jev serve --port 8008
curl -s localhost:8008/v1/fusion/decide -H 'content-type: application/json' -d '{
  "state": {"ticket": "Third time I write, nobody answers, cancel everything."},
  "questions": {
    "team": {"type":"choice","instructions":"Which team?","criteria":{"billing":"charges, refunds","technical":"bugs","retention":"cancellations","other":null}},
    "needs_human": {"type":"noul","instructions":"Does this require a human agent?"},
    "risk": {"type":"score","instructions":"How costly would a wrong automated decision be?","criteria":["harmless","annoying","costly","irreversible"]}}}'
# --- nuit : distillation puis entrainement ---
python -m jev_clone.distill --bonsai http://127.0.0.1:8080 --in data/states.jsonl --out data/labeled.jsonl --mode soft --think-if-below 0.85
python training/train_lora_rlcd.py --model Qwen/Qwen3.5-0.8B-Base --data data/train.jsonl --val data/val.jsonl --out runs/jev-0.8b --kl 0.5
```

## 8. Annexe B - Format d'echange (identique a TypeSafe, `POST /v1/systemone`)

Requete : `state` (texte ou JSON), `questions` : `{id: {"type": "noul"|"choice"|"score", "instructions": texte, "criteria": ...}}`
(`criteria` : choice = `{option: description|null}` ou liste ; score = liste de niveaux du plus bas au plus haut, 2-10 ;
noul = `{"true": ..., "false": ...}` optionnel), `permutations` (1-8, extension).
Reponse : `answers[id]` = `{"type":"noul","noul":p}` | `{"type":"choice","choice","probabilities","confidence"}` |
`{"type":"score","score","probabilities","legend","confidence"}` ; `usage` (tokens, branches, `state_cache_hit`) ;
`model` ; `latency_ms`. Un client ecrit pour le SDK TypeSafe fonctionne en pointant `TYPESAFE_BASE_URL`
sur `jev serve`.

## 9. Glossaire

**System One / System Two** : jugement rapide vs raisonnement lent (Kahneman), repris par TypeSafe pour
opposer Jev aux LLM. **Noul** : primitive booleenne de Jev (probabilite de oui). **RLCD** : Reinforcement
Learning for Calibrated Decisions, entrainement de Jev par regle de score propre. **ECE** : Expected
Calibration Error, ecart moyen entre confiance et precision par bac. **Brier** : erreur quadratique entre
la distribution et la verite. **QAT** : quantization-aware training, entrainement avec les poids deja
contraints (ici ternaires). **PTQ1_0 / PQ2_0** : deux empaquetages GGUF des poids ternaires de Bonsai 2
(1,76 et 2,16 bits/poids). **Q1_0** : format 1-bit de Bonsai v1, supporte par llama.cpp mainline.
**GatedDeltaNet** : attention lineaire (recurrente) qui compose 75 % des couches de Qwen3.5-3.8, d'ou un
cache KV tres petit et la necessite de "forker" l'etat plutot que de masquer l'attention.

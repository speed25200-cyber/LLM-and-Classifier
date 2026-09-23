# Entrainer le classifieur (S1) de Prophet Studio : donnees -> RLCD-lite -> GGUF -> Studio

Objectif : partir d'un petit modele ouvert (Qwen3.5-0.8B-Base ou 2B-Base, hybrides GatedDeltaNet comme Bonsai 2) et lui
apprendre a **decider** ce que Prophet lui demande : lire l'etat, sortir une distribution honnete sur les options, sans
jamais generer de texte. Le S1 livre (Ternary-Bonsai en lecture zero-shot) fonctionne deja ; l'entrainement apporte la
justesse et la calibration sur les questions "maison". Chaine complete sur A100 : `colab/jev_bonsai_a100.ipynb`
(guide : `docs/05-COLAB-A100.md`).

## 1. Donnees

| Source | Script | Remarque |
|---|---|---|
| **Prophet, par regles** (pre-tour, outils, verification, voix, garde-fou benin + exemples defensifs risques) | `make_synthetic_prophet.py` (`prophet_data.py`, `prophet_templates.py`) | forme d'etat et questions exactes de Prophet / Studio ; validation par gabarits tenus a l'ecart ; `--calib` = pre-tour pour `jev_clone.calibrate` |
| Garde-fou : **vos** exemples en plus | `make_synthetic_prophet.py --guard-extra fichier.jsonl` | optionnel, s'ajoute aux exemples integres (format dans l'en-tete du script) |
| Bonsai 2 27B enseignant | `--teacher URL`, ou `python -m jev_clone.distill --in ... --workers 4 --resume` | `teacher_probs` (KL), estimations remplacees, `teacher_disagrees` a relire |
| Computer use de Studio (DAgger) | `make_from_trajectories.py` | pas escalades vers Bonsai = exemples etiquetes |
| Graines etendues par Bonsai | `make_synthetic_prophet.py --expand URL --per-seed 20` | demandes nouvelles de meme nature |
| Historique de vos decisions | `make_from_history.py` | CSV / JSONL + schema |
| Jeux publics (banking77, ag_news...) | `make_public_mix.py` | hors domaine : generalisation seulement, a petite dose (<= 20 %) |

Format commun (une ligne JSON) : `{"state", "questions", "labels", "teacher_probs"?}` ; les champs en plus (`family`,
`group`, `weak`...) sont ignores par l'entrainement. Les graines livrees (`jev_clone/seeds/`) ne sont jamais des lignes
d'entrainement : le bouton Calibrer de Studio les lit. Avec `--expand`, chaque graine fait toutefois ecrire a Bonsai des
demandes qui heritent de ses etiquettes `intent`, `language`, `risk` : la calibration sur les graines n'est alors plus
tenue a l'ecart.

## 2. Entrainement

```bash
pip install -e ".[train]"
python training/make_synthetic_prophet.py --out data/train.jsonl --val data/val.jsonl --calib data/calib.jsonl --n 20000 \
    --guard-extra data/guard_risky.jsonl
# A100 : QLoRA (defaut du notebook), ou --full pour tous les poids (lr plafonne a 2e-5)
python training/train_lora_rlcd.py --model Qwen/Qwen3.5-0.8B-Base --qlora --data data/train.jsonl --val data/val.jsonl \
    --out runs/jev-clone --bs 16 --accum 2 --max-len 1536 --loss nll --kl 0.5 --permutations 2 --save-every 500
# RTX 5060 / 4060 8 Go (Studio arrete) : 0.8B en QLoRA ou LoRA bf16, batch 2 x 8
python training/train_lora_rlcd.py --model Qwen/Qwen3.5-0.8B-Base --qlora --bs 2 --accum 8 --data data/train.jsonl --out runs/jev-clone
```

Objectif = regle de score propre (`--loss nll` ou `brier`) sur les logits restreints aux etiquettes, plus une KL vers les
distributions de Bonsai (`--kl`) quand `teacher_probs` existe : la reduction "RL pour decisions calibrees" quand la
politique emet la distribution elle-meme. `--sft-data` (`{"prompt", "response"}`) ajoute la generation de reponses
courtes au meme modele (fusion au niveau modele, voir `docs/09-FUSION-PROFONDE.md`).

Cibles (validation tenue a l'ecart, `eval_clone.py`) : exactitude >= S1 livre + 10 points par famille, **ECE <= 0,05**,
garde-fou : `risky_miss_rate` = 0 et `benign_confirm_rate` en baisse.

## 3. Export GGUF : `merge_lora.py`

```bash
# QLoRA / LoRA : base rechargee en bf16 (jamais 4-bit), adaptateur fusionne, GGUF f16 puis quantifies
python training/merge_lora.py --adapter runs/jev-clone --llama-cpp ../llama.cpp --quant Q8_0,Q4_K_M
# --full (deja complet) : conversion seulement
python training/merge_lora.py --merged runs/jev-clone/merged --llama-cpp ../llama.cpp --quant Q8_0
python training/merge_lora.py --adapter runs/jev-clone --llama-cpp ../llama.cpp --check   # outils seulement
```
`--llama-cpp` : clone du fork PrismML au tag du runtime de Studio
(`git clone --depth 1 -b prism-b10683-d8f26ee https://github.com/PrismML-Eng/llama.cpp`), qui fournit
`convert_hf_to_gguf.py` ; `llama-quantize` est cherche dans `--quantize-bin`, `bin/*/`, le runtime de Studio,
`<llama.cpp>/build/bin`, le PATH. Tout outil manquant arrete le script **avant** la fusion, avec la commande pour
l'installer. Sorties : `runs/jev-clone-f16.gguf`, `-Q8_0.gguf`, `-Q4_K_M.gguf`, `jev-clone.manifest.json`.

## 4. Dans Prophet Studio

1. **Modeles > Importer un GGUF** : `jev-clone-Q8_0.gguf`, role **Classifieur** (id `custom-s1-jev-clone-q8-0`), puis
   redemarrer les modeles.
2. **Calibrer** quand il tourne (ou `python -m jev_clone.calibrate --server http://127.0.0.1:7881 --studio-model
   custom-s1-jev-clone-q8-0`) ; variante : **Importer une calibration** avec le `calibration.json` produit par
   `jev_clone.calibrate --data data/calib.jsonl` sur le GGUF (notebook ; sans `--teacher`, `data/calib.jsonl` n'etiquette
   pas `needs_reasoning`, que seules les graines calibrent). Le `calibration.json` ecrit par
   `train_lora_rlcd.py` (temperature seule, generique) n'est pas applique par Studio.
3. `python training/eval_clone.py --server http://127.0.0.1:7881 --data data/val.jsonl` avant / apres.

Le serveur lit toujours les memes etiquettes (A..Z / Yes/No) par grammaire : aucune tete speciale a porter dans GGUF.

## 5. Pourquoi pas Bonsai lui-meme comme backbone entraine ?

Bonsai 2 est une representation ternaire QAT produite avec des GPU datacenter et une propriete intellectuelle PrismML
fermee : on ne peut pas le "re-entrainer" sur une carte grand public. On l'utilise donc **tel quel** (a) comme enseignant
(distillation), (b) comme System One "mono" sans entrainement, (c) comme System Two. Pour un clone **entraine**, on part
d'un Qwen3.5 en pleine precision puis on le quantifie classiquement (Q8_0 / Q4_K_M) : a 0.8B-2B, 0,5 a 2 Go.

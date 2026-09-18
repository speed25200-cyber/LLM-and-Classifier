# Entrainer son propre clone Jev (RLCD-lite) sur une RTX 4060

Objectif : partir d'un petit modele ouvert (Qwen3.5-0.8B-Base ou 2B-Base, hybrides GatedDeltaNet comme
Bonsai 2) et lui apprendre a **decider** : lire l'etat, sortir une distribution honnete sur les options,
sans jamais generer de texte. C'est la partie "clone de Jev" proprement dite ; le niveau 0 (lecture sur un
modele non entraine) fonctionne deja sans cette etape, mais la calibration et la robustesse aux questions
"maison" viennent de l'entrainement.

## 1. Donnees

| Source | Comment | Volume conseille |
|---|---|---|
| Vos propres exemples etiquetes | JSONL `{"state", "questions", "labels"}` | 500 a 5 000 |
| Distillation Bonsai 2 27B | `python -m jev_clone.distill --mode soft --think-if-below 0.85` sur vos etats non etiquetes | 5 000 a 50 000 |
| Jeux publics de decision (banking77, go_emotions, ag_news, MMLU, HelpSteer2, tickets support...) | convertir en questions typees (voir `decider/data/` et `system-one-gemma` comme modeles de conversion) | 50 000 a 500 000 |
| Ledger de production (`runs/ledger.jsonl`) | les cas escalades, re-etiquetes par Bonsai | continu |

Melange recommande : 60 % public (generalisation) / 30 % distille (domaine) / 10 % etiquete a la main (verite).
Ajouter des permutations d'options (`--permutations 2`) et des options "other / none of the above".

## 2. Entrainement (RTX 4060 8 Go)

```bash
pip install -e ".[train]"
# 0.8B : LoRA bf16 direct (poids 1.6 Go), batch 4 x accum 4, seq 1024  -> ~2-3 h pour 100k branches
python training/train_lora_rlcd.py --model Qwen/Qwen3.5-0.8B-Base --data data/train.jsonl --val data/val.jsonl \
    --out runs/jev-0.8b --loss nll --kl 0.5 --permutations 2
# 2B : QLoRA (base 4-bit), batch 2 x accum 8                              -> ~6-8 h pour 100k branches
python training/train_lora_rlcd.py --model Qwen/Qwen3.5-2B-Base --qlora --bs 2 --accum 8 ...
```

Objectif = regle de score propre (`--loss nll` ou `brier`) sur les logits restreints aux etiquettes,
plus une KL vers les distributions de Bonsai (`--kl`) quand `teacher_probs` existe. C'est exactement la
reduction "RL pour decisions calibrees" quand la politique emet la distribution elle-meme.

Cibles de validation (jeu tenu a l'ecart) : accuracy >= niveau 0 + 10 points sur vos questions,
**ECE <= 0.05**, Brier en baisse, precision selective >= 95 % a >= 70 % de couverture.

## 3. Export vers llama.cpp (pour servir avec `scripts/start_jev_clone.sh`)

```bash
# 1) fusionner l'adaptateur (fait automatiquement sans --qlora : runs/jev-0.8b/merged ; avec QLoRA :
#    recharger la base en bf16, appliquer l'adaptateur avec peft puis merge_and_unload())
# 2) convertir (le fork PrismML ou llama.cpp mainline incluent convert_hf_to_gguf.py)
python llama.cpp/convert_hf_to_gguf.py runs/jev-0.8b/merged --outfile runs/jev-0.8b-f16.gguf --outtype f16
llama-quantize runs/jev-0.8b-f16.gguf runs/jev-0.8b-Q8_0.gguf Q8_0     # 0.8B Q8_0 ~ 0.9 Go
# 3) servir
JEV_GGUF=runs/jev-0.8b-Q8_0.gguf ./scripts/start_jev_clone.sh
JEV_CALIBRATION=runs/jev-0.8b/calibration.json jev serve
```

Le serveur lit toujours les memes etiquettes (A..Z / Yes/No) par grammaire : aucune tete speciale a
porter dans GGUF, ce qui est la raison de ce choix de conception.

## 4. Pourquoi pas Bonsai lui-meme comme backbone entraine ?

Bonsai 2 est une representation ternaire QAT (quantization-aware training) produite avec des GPU
datacenter et une propriete intellectuelle PrismML fermee : on ne peut pas le "re-entrainer" sur une 4060.
On l'utilise donc **tel quel** (a) comme enseignant (distillation), (b) comme System One "mono" sans
entrainement, (c) comme System Two. Les petits Ternary-Bonsai (1.7B/4B) servent de clone niveau 0 ;
pour un clone **entraine**, on part d'un Qwen3.5 en pleine precision puis on le quantifie classiquement
(Q8_0/Q4_K_M) : a 0.8B-2B, cela reste minuscule (0.5-2 Go).

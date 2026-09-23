# 05 - Usine A100 (Colab) : entrainer le classifieur (S1) de Prophet Studio

Une A100 ne sert pas a *faire tourner* Prophet Studio (il tourne sur la machine cible : Windows 11 + RTX 5060 8 Go, Bonsai 2
27B sur le GPU, le classifieur sur le CPU) mais a **fabriquer** le classifieur : generer les donnees, les faire etiqueter par
Bonsai 2 27B (optionnel), entrainer, fusionner, exporter en GGUF, calibrer. Le notebook `colab/jev_bonsai_a100.ipynb` fait
tout, cellule par cellule ; le resultat s'importe dans Studio en deux clics.

```
make_synthetic_prophet.py --(option)--> Bonsai 2 27B enseignant (jev_clone.distill) --> train_lora_rlcd.py --qlora
    --> merge_lora.py (fusion bf16 + convert_hf_to_gguf + llama-quantize) --> jev_clone.calibrate + eval_clone.py
    --> Drive / telechargement --> Studio : Modeles > Importer un GGUF (Classifieur) > Calibrer
```

## 1. Repartition

| | A100 (Colab) | RTX 5060 8 Go (Prophet Studio) |
|---|---|---|
| Bonsai 2 27B | enseignant optionnel, PQ2_0 (plus rapide sur Ampere), 4 slots | System Two sur le GPU (PTQ1_0) |
| Classifieur | entraine (QLoRA par defaut), fusionne, converti, calibre sur le GGUF servi comme dans Studio | S1 sur le CPU, un slot, 8 k de contexte |
| Sortie | `jev-clone-Q8_0.gguf`, `jev-clone-Q4_K_M.gguf`, `calibration.json`, `eval_clone.json`, `jev-clone.manifest.json` | GGUF importe, calibre sur place |

## 2. Deroule du notebook

1. **Parametres** : `BRANCH` (branche clonee, par defaut `claude/local-llm-high-performance-q4wk6f` : celle qui contient
   `prophet_studio/`), `MODEL` (base HF), `TRAIN_MODE` (`qlora` | `lora` | `full`), `N_EXAMPLES`, `USE_TEACHER`,
   `GUARD_EXTRA`, `QUANTS`, `LLAMA_TAG` (runtime de Studio : `prism-b10683-d8f26ee`), `NAME`, `DRIVE`.
2. GPU, Drive.
3. Clone de la branche, `pip install -e ".[train,serve]"` ; la cellule s'arrete si `prophet_studio/` ou
   `training/merge_lora.py` manquent (mauvaise branche).
4. llama.cpp du fork PrismML **au tag du runtime de Studio** : binaires CUDA (`llama-server`, `llama-quantize`) et source
   (`convert_hf_to_gguf.py`). Pas de `requirements-convert_hf_to_gguf.txt` : il installerait un torch CPU a la place du
   torch CUDA.
5. Enseignant (si `USE_TEACHER`) : profil A100 sans clone niveau 0 ni vision, Bonsai 2 27B PQ2_0 sur le port 8080.
6. Donnees (section 3), copiees dans Drive ; Bonsai est arrete ensuite (tout le GPU pour l'entrainement).
7. Entrainement `train_lora_rlcd.py` (NLL + KL si enseignant, permutations, reprise tous les 500 pas copiee dans Drive).
8. `merge_lora.py` : base rechargee en bf16 (jamais en 4-bit), adaptateur fusionne, GGUF f16, puis Q8_0 et Q4_K_M.
9. Le GGUF est servi comme dans Studio (8 k par slot, KV q8_0) : `jev_clone.calibrate` sur `prophet_calib.jsonl`,
   `training/eval_clone.py` sur `val.jsonl`.
10. Artefacts dans Drive et telechargement direct ; la derniere cellule rappelle l'import dans Studio (section 4).

## 3. Donnees : ce que Prophet demande vraiment au classifieur

`training/make_synthetic_prophet.py` (generateurs dans `training/prophet_data.py`, gabarits dans
`training/prophet_templates.py`) ecrit des exemples au format de `train_lora_rlcd.py`, avec **les questions et la forme
d'etat exactes** des appels de Prophet et de Studio (un test compare les deux, `tests/test_train_chain.py`) :

| Famille | Questions | Etat | Qui la pose |
|---|---|---|---|
| `turn` | `PROPHET_TURN` : direct, clarify, intent, language, needs_reasoning, risk | `{request, workspace_files, recent_turns}` | pre-tour de chaque demande |
| `tools` | un noul `t_<outil>` par outil du catalogue reel (shell PowerShell ou bash) | `{request, recent_turns}` | choix des outils exposes a Bonsai |
| `verify` | `ok` | `{request, response, files}` | verification d'une reponse (voie directe reprise si trop basse) |
| `voice` | `intent` : commandes de `prophet_studio/voice.py` + `prompt` | `{utterance, context}` | commande vocale hors grammaire exacte |
| `guard` | `JUDGE_QUESTIONS` : tool_risk (5 classes), risk, policy_violation | `{user_request, proposed_action}` | garde-fou de chaque action |

* **Garde-fou** : le generateur ecrit des actions benignes (lecture ; ecriture, build, tests, installation dans le projet,
  sous PowerShell et bash) et ~40 % d'exemples defensifs des classes risquees (`training/prophet_templates.py`,
  `GUARD_RISKY` : destructive, privileged, exfiltration, violations de politique, souvent derriere une demande anodine :
  le juge apprend a evaluer l'action, pas l'intention affichee). Vos propres exemples s'ajoutent avec
  `--guard-extra guard_risky.jsonl` (dans Drive pour le notebook), une ligne par action jugee :
  `{"state": {"user_request": "...", "proposed_action": "shell: ..."}, "labels": {"tool_risk": "destructive", "risk": 3, "policy_violation": true}}`.
  Sources : vos refus dans Studio (journal `runs/ledger.jsonl`), vos regles internes, une relecture humaine.
* **Etiquettes** : les regles font foi ; `weak` liste les estimations (`needs_reasoning` du pre-tour). Avec un enseignant
  (`--teacher`, ou `python -m jev_clone.distill --in ... --workers 4 --resume`), Bonsai ajoute `teacher_probs` (terme KL),
  remplace les estimations, et `teacher_disagrees` liste ses desaccords avec les regles (a relire). Un outil "limite" pour
  une demande n'est pas etiquete.
* **Validation** : des gabarits entiers tenus a l'ecart (jamais vus a l'entrainement). `--calib` = la part pre-tour de la
  validation : une calibration ajustee aussi sur le garde ou la voix s'y appliquerait (meme nom de question `risk`,
  `intent`), d'ou ce fichier a part.
* Les graines livrees (`jev_clone/seeds/prophet_seeds.jsonl`) ne vont jamais dans l'entrainement : le bouton Calibrer de
  Studio les lit, elles doivent rester tenues a l'ecart.
* En plus : trajectoires du computer use de Studio (`make_from_trajectories.py`, deposez `trajectories*.jsonl` dans
  Drive), graines etendues par Bonsai (`--expand URL`), jeux publics (`make_public_mix.py`, hors domaine : a petite dose).

## 4. Import dans Prophet Studio

1. Recuperer `jev-clone-Q8_0.gguf` (Drive ou telechargement du notebook).
2. **Modeles > Importer un GGUF** : chemin, role **Classifieur**, Importer. Studio le choisit comme S1
   (`custom-s1-jev-clone-q8-0`) au prochain demarrage des modeles.
3. Quand il tourne : **Calibrer** (graines livrees lues sur ces poids : la calibration de reference, liee au fichier ; un GGUF
   remplace la rend perimee). Variante : **Importer une calibration** -> `calibration.json` du notebook. En ligne de
   commande : `python -m jev_clone.calibrate --server http://127.0.0.1:7881 --studio-model custom-s1-jev-clone-q8-0`.
4. Mesurer avant / apres sur la machine (S1 de Studio : port 7881) :
   `python training/eval_clone.py --server http://127.0.0.1:7881 --data prophet_val.jsonl`.
   Par famille : exactitude, ECE ; garde-fou : `benign_confirm_rate` (confirmations inutiles) et `risky_miss_rate`
   (actions risquees non arretees : doit rester a 0).

## 5. Choisir la taille

Le S1 de Studio tourne sur le CPU a cote de Bonsai 2 27B (la VRAM de la 5060 va au 27B) : chaque demande coute plusieurs
lectures S1 (pre-tour, outils, garde a chaque action, verification). **0.8B en Q8_0** (~0,9 Go) est le choix par defaut ;
un 2B en Q4_K_M (~1,3 Go) est plus juste mais plus lent sur CPU : mesurer la latence dans Studio (bandeau S1) avant de
l'adopter.

## 6. Sans Colab (sur la machine, PowerShell)

```powershell
uv sync --extra train                     # torch CUDA : voir pytorch.org pour la roue adaptee a la 5060
python training/make_synthetic_prophet.py --out data/prophet_train.jsonl --val data/prophet_val.jsonl --calib data/prophet_calib.jsonl --n 20000 --guard-extra data/guard_risky.jsonl
# Studio arrete (le GPU est libre) : QLoRA 0.8B, batch 2 x 8
python training/train_lora_rlcd.py --model Qwen/Qwen3.5-0.8B-Base --qlora --data data/prophet_train.jsonl --val data/prophet_val.jsonl --out runs/jev-clone --bs 2 --accum 8
git clone --depth 1 -b prism-b10683-d8f26ee https://github.com/PrismML-Eng/llama.cpp ..\llama.cpp
python training/merge_lora.py --adapter runs/jev-clone --llama-cpp ..\llama.cpp --quant Q8_0
```
`merge_lora.py` trouve `llama-quantize.exe` dans le runtime installe par Studio ; `--check` verifie les outils sans rien
calculer.

## 7. Contraintes Colab

* Session ephemere (12 h en Pro, 24 h en Pro+) : donnees, points de reprise et artefacts sont copies dans Drive a chaque
  etape ; `jev_clone.distill --resume` reprend un etiquetage interrompu.
* Disque local : Bonsai 2 PQ2_0 (7,3 Go, seulement avec enseignant) + base HF + runs : < 30 Go.
* Reseau : Hugging Face et GitHub accessibles ; aucun jeton pour les poids publics.
* Ordres de grandeur (a confirmer) : generation 20 000 exemples < 1 min ; enseignant soft ~0,4 s par exemple et par slot ;
  QLoRA 0.8B sur ~150 000 branches ~1-2 h ; fusion + conversion + quantification < 10 min.

## 8. Ce qui reste a verifier a la premiere execution

Rien de cette chaine n'a tourne sur de vrais modeles dans l'environnement de redaction (pas de GPU, Hugging Face bloque) :
les generateurs, l'etiquetage enseignant (serveur llama factice), la calibration, l'import de la calibration dans Studio,
les erreurs de `merge_lora.py` et la structure du notebook sont testes (`tests/test_train_chain.py`). A surveiller : noms
des modules LoRA de la base (`in_proj_qkvz`, `in_proj_ba`, `out_proj` pour les couches GatedDeltaNet), version de
`transformers` compatible avec la base, prise en charge de son architecture par `convert_hf_to_gguf.py` du tag
`prism-b10683-d8f26ee`, nom exact de l'archive CUDA de la release (`llama-<tag>-bin-linux-cuda-<12.4|12.8|13.3>-x64.tar.gz`).

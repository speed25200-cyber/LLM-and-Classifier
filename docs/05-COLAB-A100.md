# 05 - Usine A100 80 Go (Colab) et production RTX 4060

Une A100 80 Go ne sert pas a *faire tourner* la fusion (elle doit tenir sur la 4060) mais a la **fabriquer** :
distiller Bonsai 2 a grande echelle, entrainer le clone en pleine precision, l'evaluer, l'exporter.
Le notebook `colab/jev_bonsai_a100.ipynb` fait tout ; ce document explique quoi faire ou et pourquoi.

## 1. Repartition

| | A100 80 Go (Colab) | RTX 4060 8 Go (production) |
|---|---|---|
| Bonsai 2 27B | **PQ2_0** (7,25 Go), contexte 64 k en KV f16, vision sur GPU, 4 slots ; ~74 tok/s et ~1 300 tok/s de prefill (livre blanc, A100 SXM ; PTQ1_0 est plus lent sur Ampere) | PTQ1_0 (5,93 Go), 8 k, KV q4, 1 slot ; ~30 tok/s estimes |
| Clone de Jev | Ternary-Bonsai-8B (niveau 0, enseignant rapide) puis **le clone entraine** teste sur llama-server | Ternary-Bonsai-1.7B (niveau 0) puis le **GGUF entraine** rapatrie |
| Distillation | 20 000-50 000 etats, mode soft ~0,4 s/etat, think ~3-8 s/etat | quelques centaines d'etats (calibration) |
| Entrainement | Qwen3.5-2B (ou 4B) **fine-tuning complet** bf16, batch 16-32, 1 536-2 048 tokens ; 0,8B en ~1,5 h, 2B en ~4-6 h | LoRA 0,8B ou QLoRA 2B, la nuit |
| Evaluation | jeu tenu a l'ecart + fusion complete (`examples/ticket_routing.py`) | verification finale sur cible |
| Sortie | `jev-clone-Q8_0.gguf`, `jev-clone-Q4_K_M.gguf`, `calibration.json`, `merged/` dans Google Drive | `models/jev-clone-*.gguf`, `runs/calibration.json` |

## 2. Deroule du notebook (14 cellules)

1. GPU, montage de Drive (`/content/drive/MyDrive/jev-bonsai`).
2. Clone du depot, `pip install -e ".[serve,dev,train]"`, `SKIP_VENV=1 PROFILE=scripts/profiles/a100-80gb.env sh scripts/setup.sh`
   (binaires CUDA du fork PrismML, Bonsai 2 PQ2_0 + mmproj, Ternary-Bonsai-8B). Repli : compilation depuis
   les sources avec `-DCMAKE_CUDA_ARCHITECTURES=80` (~10 min).
3. Bonsai en arriere-plan (`subprocess.Popen` de `start_bonsai.sh`), attente de `/health`, mesure tok/s.
4. Donnees : `training/make_public_mix.py` (banking77, ag_news, go_emotions, MMLU, yelp) + vos `states.jsonl`
   depuis Drive etiquetes par `jev_clone.distill` (soft, escalade think sous 0,85). Le domaine est duplique
   (x2) dans le melange ; 300 exemples de domaine sont reserves a la validation.
5. `train_lora_rlcd.py --full` sur `Qwen/Qwen3.5-2B-Base` (ou 0.8B), NLL + KL enseignant, permutations,
   points de reprise tous les 500 pas copies dans Drive (une session Colab peut s'arreter).
6. Export : `convert_hf_to_gguf.py` (fork PrismML) -> f16 -> `llama-quantize` Q8_0 et Q4_K_M.
7. Le GGUF est servi par `start_jev_clone.sh` (port 8081) : `calibrate.py` ajuste temperature et seuils
   **sur le modele exactement tel qu'il sera deploye** (quantifie), `jev bench` mesure la latence.
8. Copie des artefacts dans Drive ; commandes a lancer sur la 4060.

## 3. Choisir la taille du clone final

| Profil 4060 | VRAM restante a cote de Bonsai | Clone conseille | Taille GGUF |
|---|---|---|---|
| qualite (Bonsai 2 PTQ1_0) | ~0,6-1 Gio | Qwen3.5-0.8B entraine, Q8_0 | 0,9 Go |
| vitesse (Bonsai-27B 1-bit) | ~3 Gio | Qwen3.5-2B entraine, Q8_0 (ou 4B en Q4_K_M) | 2,1 Go (2,5 Go) |
| mono | 0 | aucun (Bonsai juge) ; le clone entraine peut tourner sur CPU (`JEV_NGL=0`) | - |

Un clone 2B entraine sur 180 M tokens depasse nettement un 1.7B ternaire non entraine sur les questions
"maison" (decider : +19 points d'accuracy en domaine et ECE divise par 3 par rapport a la lecture brute).

## 4. Contraintes Colab a connaitre

* Session ephemere (12 h en Pro, 24 h en Pro+) : tout ce qui compte va dans Drive au fil de l'eau
  (`--save-every`, copies apres chaque etape) ; relancer le notebook reprend ou il en etait.
* Disque local ~200 Go : Bonsai 2 PQ2_0 (7,3 Go) + 8B (1,8 Go) + base HF 2B (4 Go) + runs : < 30 Go.
* Reseau : Hugging Face et GitHub sont accessibles ; les poids se telechargent sans jeton.
* Pas de port expose : tout passe par `127.0.0.1` dans la VM ; pour une demo interactive, tunneler
  `jev serve` (par exemple avec `cloudflared`) ou utiliser les exemples en local.
* CUDA : Colab fournit un pilote recent ; `setup.sh` choisit la build 12.4 / 12.8 / 13.3 selon `nvcc`.
  L'A100 (sm_80) est couverte par les binaires precompiles ; sinon compiler (cellule de repli).
* Facturation : ~2-4 unites de calcul par heure d'A100 ; un cycle complet (distillation 20 k etats +
  entrainement 2B + export) tient dans une session de 8-10 h.

## 5. Ce qui reste a verifier sur la machine cible

Ce notebook n'a pas pu etre execute dans l'environnement de redaction (pas de GPU, Hugging Face bloque).
Points a surveiller a la premiere execution : noms exacts des fichiers GGUF dans les depots `prism-ml/*`
(les motifs `*-PQ2_0.gguf`, `*mmproj-Q8_0.gguf`, `*-Q2_0_g64.gguf` viennent du depot officiel
`Bonsai-demo`), version de `transformers` compatible Qwen3.5 (>= 4.57 ; 5.x pour les couches
GatedDeltaNet avec `peft`), noms des modules LoRA (`in_proj_qkvz`, `in_proj_ba`, `out_proj`) si vous
n'utilisez pas `--full`, et le script `convert_hf_to_gguf.py` du fork pour l'architecture `qwen35`.

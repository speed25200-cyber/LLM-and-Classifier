# 03 - Materiel : RTX 4060 et exigences minimales

## 1. La RTX 4060 en chiffres utiles

| | RTX 4060 (desktop) | RTX 4060 Laptop | Ce que ca implique |
|---|---|---|---|
| VRAM | 8 Go GDDR6 (~7,6-7,8 Gio utilisables, moins ~0,5-1 Gio si un bureau Windows tourne dessus) | 8 Go | le **budget** de la fusion : Bonsai 2 (~6,8 Gio) + clone (~0,6 Gio) = ~7,4 Gio -> juste ; d'ou les profils "qualite" (KV4, projecteur en RAM) et "vitesse" (1-bit) |
| Bande passante | 272 Go/s | 256 Go/s | le decodage est limite par la relecture des poids : **tok/s max ~= 272 / taille des poids en Go** |
| Calcul | Ada (sm_89), 3 072 coeurs, tensor cores 4e gen, 115 W | 96 W | prefill correct ; PTQ1_0 est plus rapide que PQ2_0 en generation sur Ada (comme sur la 4090) mais 2x plus lent en prefill |
| CUDA | binaires PrismML pour CUDA 12.4 / 12.8 / 13.3 | idem | `scripts/setup.sh` detecte la version du pilote |

### Estimation de debit (a verifier avec `scripts/bench.sh`)

Efficacite observee sur Ada/Blackwell grand public : ~0,7-0,8 de la limite bande passante en ternaire
(RTX 5060 Ti : 44,4 tok/s pour 7,15 Go a 448 Go/s ; RTX 4080 Laptop : ~45 tok/s pour 7,6 Go a 432 Go/s),
~0,5 en 1-bit (noyaux moins amortis : L40S 107 tok/s pour 3,8 Go a 864 Go/s).

| Modele sur RTX 4060 | Poids | Limite bande passante | **Estimation realiste** | Prefill (pp512) estime |
|---|---|---|---|---|
| Bonsai 2 27B `PTQ1_0` | 5,93 Go | 46 tok/s | **28-34 tok/s** | ~300-400 tok/s |
| Bonsai 2 27B `PQ2_0` | 7,25 Go | 37 tok/s | 24-28 tok/s (et ne laisse pas de place au clone sur 8 Go) | ~600-700 tok/s |
| Bonsai-27B 1-bit `Q1_0` | 3,79 Go | 72 tok/s | **35-45 tok/s** | ~500-600 tok/s |
| Bonsai-8B 1-bit `Q1_0` | 1,15 Go | 237 tok/s | ~100-130 tok/s | > 2 000 tok/s |
| Ternary-Bonsai-1.7B (clone) | 0,37 Go | > 700 tok/s | n/a (pas de generation) | > 4 000 tok/s |

Latence d'une **decision System One** (clone 1.7B sur GPU, etat de 300 tokens deja en cache, 4 questions
de ~60 tokens chacune) : ~40-80 ms ; premiere requete sur un nouvel etat : + ~100 ms (prefill de l'etat).
En mode "mono" sur Bonsai 2 PTQ1_0 : ~0,8-1,5 s pour un nouvel etat (prefill lent), ~150-300 ms si l'etat
est en cache. Ces ordres de grandeur sont **a mesurer** : `jev bench --server ...`.

## 2. Matrice des profils

| Profil (`scripts/profiles/`) | Materiel | System Two (Bonsai) | System One (clone) | Contexte | VRAM / RAM estimee | Generation attendue |
|---|---|---|---|---|---|---|
| `minimal-cpu.env` | **aucun GPU**, 8-16 Go RAM, >= 4 coeurs | Bonsai-8B Q1_0 (1,07 Gio) sur CPU | Ternary-Bonsai-1.7B sur CPU | 4 k | ~4 Go RAM | ~15-20 tok/s (8 coeurs AVX2/AVX-512) ; 27B 1-bit possible avec 16 Go : ~5-10 tok/s |
| `gpu-4gb.env` | GPU 4 Go (RTX 3050 4 Go, A2000, portable) | Bonsai-8B Q1_0 sur GPU | Ternary-Bonsai-1.7B sur GPU | 8 k | ~3,5 Gio | ~60 tok/s (mesure A2000 : 63) |
| `rtx4060-8gb-vitesse.env` | 8 Go | Bonsai-27B 1-bit Q1_0 (3,53 Gio) | Ternary-Bonsai-4B (0,86 Go) sur GPU | 16 k (KV4) | ~6,5 Gio | ~35-45 tok/s, qualite 89,5 % de Qwen3.6 |
| **`rtx4060-8gb-qualite.env`** (recommande) | 8 Go | **Bonsai 2 27B PTQ1_0** (5,52 Gio) | Ternary-Bonsai-1.7B sur GPU (ou CPU si OOM) | 8 k (KV4) | ~7,2-7,4 Gio | ~28-34 tok/s, qualite 98,2 % de Qwen3.8 |
| `rtx4060-8gb-mono.env` | 8 Go | Bonsai 2 27B PTQ1_0 | **Bonsai lui-meme** (lecture par grammaire) | 8 k (KV4), 3 slots | ~6,8 Gio | idem ; decisions plus lentes mais meilleures |
| `gpu-12gb.env` | 12 Go (RTX 3060 12 Go, 4070) | Bonsai 2 27B PQ2_0 + vision sur GPU | Ternary-Bonsai-4B | 32 k (KV4) | ~10,5 Gio | ~25-45 tok/s selon carte |
| `gpu-16gb.env` | 16 Go (4060 Ti 16 Go, 5060 Ti) | Bonsai 2 27B PQ2_0 + vision | Ternary-Bonsai-8B (1,75 Go) | 64 k (KV f16) | ~14,5 Gio | ~45 tok/s (mesure 5060 Ti sur ternaire v1) |
| `a100-80gb.env` | **A100 80 Go (Colab Pro)** : usine de distillation / entrainement, banc d'evaluation | Bonsai 2 27B PQ2_0 + vision, 4 slots | Ternary-Bonsai-8B puis le clone entraine (Qwen3.5-2B complet) | 64 k (KV f16) | ~20 Gio + entrainement | 74 tok/s / 1 328 pp512 (livre blanc, A100 SXM ; PTQ1_0 plus lent sur Ampere : 55 / 703) |

**Exigence minimale absolue** pour faire tourner *la fusion* : un CPU x86-64 (AVX2) ou ARM avec 8 Go de
RAM, ~6 Go de disque : profil `minimal-cpu` (8B 1-bit + 1.7B ternaire). Pour un *27B* : 16 Go de RAM
(CPU) ou 8 Go de VRAM (GPU). Pour *Bonsai 2 27B* : 8 Go de VRAM ou 16 Go de RAM unifiee (Mac M-series).

**A100 80 Go disponible (Colab)** : elle ne remplace pas la 4060, elle la prepare. Voir
[05 Colab A100](05-COLAB-A100.md) : distillation a ~0,4 s/etat, fine-tuning complet d'un Qwen3.5-2B en
quelques heures, export GGUF Q8_0 (2,1 Go) ou Q4_K_M (1,3 Go) a deposer dans `models/` sur la 4060.

## 3. Feuille de calcul VRAM (a faire avant de choisir)

```
VRAM necessaire ~= poids_S2 + surcout_runtime (0,9-1,3 Gio) + KV_S2 + [mmproj 0,6-0,9 Gio si sur GPU]
                 + poids_S1 + surcout_S1 (~0,2-0,3 Gio) + KV_S1
KV (27B hybride) : 64 Kio/token en f16, ~18 Kio/token en q4_0   -> 8 k tokens : 0,5 Gio f16, 0,14 Gio q4
KV (8B / 4B / 1.7B Ternary-Bonsai, attention complete) : ~140 / ~70 / ~35 Kio/token en f16 (8 k : 1,1 / 0,55 / 0,27 Gio)
```
Exemple profil qualite : 5,52 + 1,2 + 0,14 + 0 + 0,35 + 0,25 + 0,14 = **~7,6 Gio** en cache f16 pour le
clone, ~7,4 Gio avec `-ctk q4_0` ; c'est pourquoi le profil met le projecteur vision en RAM et pourquoi
`JEV_NGL=0` (clone sur CPU) est la premiere parade en cas de `out of memory`.

## 4. Que faire en cas de "CUDA out of memory" (dans l'ordre)

1. `BONSAI_MMPROJ=off` (ou `cpu`) : -0,6 a -0,9 Gio.
2. `BONSAI_KV4=1` : KV divise par ~3,5.
3. Reduire `BONSAI_CTX` (8192 -> 4096) et `BONSAI_NP=1`.
4. `JEV_NGL=0` : le clone 1.7B tourne sur CPU (prefill ~3 000 tok/s sur un i7, ca reste rapide).
5. Fermer le navigateur / passer l'affichage sur l'iGPU (Windows : ~0,5-1 Gio de VRAM recuperes).
6. Passer au profil `vitesse` (Bonsai-27B 1-bit) ou `mono`.
7. Dernier recours : `BONSAI_NGL=50` (dechargement partiel, environ -30 a -50 % de vitesse par tranche
   de couches sur CPU) ; llama.cpp ne fait **pas** tomber silencieusement le KV en RAM si `-c` et `-ngl`
   sont explicites, mais un `-c 0` ou un `-ngl` absent le fait (piege documente : "tous les coeurs a
   100 %, 10x plus lent").

## 5. Notes de plateforme

* **Linux** : pilote NVIDIA >= 550 (CUDA 12.4+) ; `scripts/setup.sh` telecharge les binaires du fork
  (`prism-b10683`). Compilation depuis les sources : `git clone -b prism https://github.com/PrismML-Eng/llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build -j2`
  (le `-j2` est volontaire sous 16 Go de VRAM : la compilation CUDA sature la memoire).
* **Windows** : binaires `win-cuda-12.4-x64` de la meme release ; `scripts/windows/*.ps1` reprennent
  les memes drapeaux ; le depot `PrismML-Eng/Bonsai-demo` fournit `setup.ps1` pour telecharger
  binaires + poids. Python 3.11+ et `pip install -e .` pour `jev_clone`.
* **AMD** : ROCm 7.2 (Linux) ou Vulkan ; **Bonsai 2 n'a pas encore de noyaux Vulkan** (PQ2_0 : CUDA,
  Metal, ROCm, CPU) -> sur Vulkan prendre Bonsai-27B Q1_0 ou Ternary Q2_0_g64 (mainline).
* **Mac Apple Silicon** : 16 Go de memoire unifiee suffisent pour Bonsai 2 (MLX 8,5 Go ou GGUF Metal) ;
  le decodage speculatif n'y apporte rien ; M4 Pro ~18 tok/s, M5 Pro ~28, M5 Max ~47.
* **CPU seul** : AVX-512 VNNI (Intel 11e gen+, Zen 4+) accelere nettement le ternaire ; regler `-t` au
  nombre de coeurs physiques (pas de gain au-dela de 7-8 threads mesure sur un i7-11800H).

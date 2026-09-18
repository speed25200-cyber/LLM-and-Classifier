# 02 - Bonsai 2 27B (PrismML) : le modele 27B qui tient dans 6 Go

> Sources : livre blanc "Ternary Bonsai 2 27B" (PrismML, septembre 2026), livre blanc "Bonsai 27B"
> (juillet 2026), depot `PrismML-Eng/Bonsai-demo` (README, MODEL-FORMATS, KV-CACHE, SPECULATIVE,
> benchmarks communautaires), annonces presse. Chiffres exacts en annexe de `docs/04-REFERENCES.md`.

## 1. La famille Bonsai en une page

| Date | Modele | Base | Format | Taille | Quoi |
|---|---|---|---|---|---|
| debut 2026 | Bonsai 1.7B / 4B / 8B (1-bit) | Qwen3 | Q1_0 (1,125 bpw) | 0,23 / 0,54 / 1,07 Gio | premiers modeles 1-bit derives d'un pretrained |
| 16 avr. 2026 | Ternary-Bonsai 1.7B / 4B / 8B | Qwen3 | ternaire {-1,0,+1}, 1,58 bpw | 0,37 / 0,86 / 1,75 Go | +5 points vs 1-bit pour ~600 Mo de plus (8B) |
| 14 juil. 2026 | Bonsai 27B (1-bit) et Ternary-Bonsai 27B | **Qwen3.6-27B** | Q1_0 3,53 Gio ; Q2_0 6,66 Gio | 89,5 % / 95 % du FP16 | premier 27B sur telephone (~11 tok/s iPhone 17 Pro Max) ; vision, outils, reflexion, 262 k de contexte |
| **17 sept. 2026** | **Ternary Bonsai 2 27B** | **Qwen3.8-27B** (sorti le 14 aout 2026) | ternaire g128 rotation Hadamard | **5,93 Go (PTQ1_0)** / 7,25 Go (PQ2_0) | **98,2 % du FP16** (83,9 vs 85,4 sur 20 benchmarks), depasse Qwen3.6-27B FP16 (83,6) |

Licence Apache 2.0 pour tous ; poids publics sur Hugging Face (`prism-ml/*`), sans token.
Collection officielle Bonsai 2 : **https://huggingface.co/collections/prism-ml/bonsai-2** (annonce PrismML
du 17 septembre 2026, 23:03 : "Based on Qwen3.8 27B, Bonsai 2 27B is 9x smaller than its full-precision
counterpart while retaining 98.2% of its aggregate benchmark performance. [...] The footprint remains 5.9 GB").
Depots connus de la collection (d'apres les scripts officiels `Bonsai-demo`) : `Ternary-Bonsai-2-27B-gguf`
(PTQ1_0, PQ2_0, mmproj Q8_0 / BF16), `Ternary-Bonsai-2-27B-mlx-2bit`, et a part `Ternary-Bonsai-2-27B-gguf-dev`
(bande Q2_0 de test, fork obligatoire). **Bonsai 2 n'existe qu'en 27B** pour l'instant : les scripts
officiels refusent les autres tailles pour cette famille.

## 2. Bonsai 2 27B : specification

| Element | Valeur |
|---|---|
| Architecture | celle de **Qwen3.8-27B** : 64 blocs hybrides, ~75 % attention lineaire **Gated DeltaNet** (48 couches) / ~25 % attention complete gated (16 couches), SwiGLU, RoPE, RMSNorm |
| Parametres | 24,35 G (LM) + 0,47 G (tour de vision, 27 blocs) + 2,54 G (embeddings / LM head) = **27,36 G** |
| Contexte | **262 144 tokens** (rendu praticable par l'attention lineaire : cache KV ~64 Kio/token en FP16, ~18 Kio en Q4) |
| Poids | ternaires {-1, 0, +1}, une echelle FP16 par groupe de 128 -> 1,71 bit/poids ; 0,0976 % des parametres (chemin d'etat recurrent `in_proj_a/b`, `conv1d`, normes) gardes en pleine precision -> 1,72 bpw |
| Base tournee | rotation **Walsh-Hadamard par blocs de 1024** avec signes fixes ; a l'inference chaque projection calcule `W (R x)`, d'ou une transformee d'activation a executer (couteuse a batch 1, fusionnee dans les noyaux CUDA/Metal) |
| Formats GGUF | **PTQ1_0** 1,76 bpw **5,93 Go** (trits densement empaquetes, plus petit) ; **PQ2_0** 2,16 bpw 7,25 Go (2 bits/poids, deballage plus simple, **prefill plus rapide**) ; projecteur vision `mmproj` HQQ 4-bit 0,63 Go (ou BF16 0,93 Go) ; MLX 2-bit 8,49 Go |
| Reflexion | modele "thinking" ; efforts `xhigh` ou `medium` (`low` ne reduit pas la reflexion) ; budget ajustable (`--reasoning-budget N`) |
| Outils / vision | appels d'outils OpenAI natifs (`--jinja`), images via `mmproj` (~4 096 tokens vision max) |
| Runtime | **fork llama.cpp PrismML** (branche `prism`, release `prism-b10683`) : PTQ1_0 / PQ2_0 sont refuses par llama.cpp mainline (ids de type inconnus) ; **ne jamais** charger la bande `Q2_0` de Bonsai 2 sur mainline (elle charge sans erreur et produit du charabia) ; MLX (Python/Swift) via mlx-vlm |
| Entrainement | QAT (quantization-aware training) + distillation avec traces de reflexion du professeur FP16 (communication PrismML), "propriete intellectuelle Caltech" ; part du modele pretrained, contrairement a BitNet qui pre-entraine de zero |

## 3. Qualite : ce qui est preserve, ce qui ne l'est pas

Moyenne des 20 benchmarks (mode reflexion, effort xhigh, EvalScope + vLLM sur H100) :

| Variante | bpw | Taille | Moyenne | vs Qwen3.8 FP16 |
|---|---|---|---|---|
| Qwen3.8-27B FP16 | 16 | 53,8 Go | 85,4 | 100 % |
| Qwen3.6-27B FP16 | 16 | 53,8 Go | 83,6 | 97,9 % |
| Qwen3.8-27B IQ2_XXS (quantif. classique) | 2,2 | 7,3 Go | 75,2 | 88,4 % |
| **Ternary Bonsai 2 27B** | **1,76** | **5,93 Go** | **83,9** | **98,2 %** |

Par categorie (table de l'annonce PrismML, identique au livre blanc) :

| Capacite | Qwen3.6 27B | Qwen3.8 27B | **Ternary Bonsai 2 27B** | Retention vs Qwen3.8 |
|---|---|---|---|---|
| Connaissances & raisonnement | 84,71 | 86,66 | **83,95** | 96,9 % |
| Maths | 94,64 | 97,06 | **96,57** | 99,5 % |
| Code | 82,57 | 82,17 | **81,58** | 99,3 % |
| Agentique & appels d'outils | 80,05 | 79,74 | **77,57** | 97,3 % |
| Suivi d'instructions | 74,53 | 81,25 | **82,66** | 102 % |
| Vision | 79,82 | 81,64 | **78,59** | 96,3 % |
| **Global (20 benchmarks)** | 83,6 | 85,4 | **83,9** | **98,2 %** |
 Points saillants : AIME26 95,8, LiveCodeBench 90,1 (IQ2_XXS : 78,6 et 70,1) ;
τ²-Bench 80,2, BFCL v3 74,9 ; **Terminal-Bench 2.1 52,8 (Qwen3.8 : 69,7) et SWE-bench Verified 60,8 (80,6)** :
l'agentique long-horizon retient ~75 %, c'est la faiblesse relative a connaitre. Le livre blanc note
aussi que la generation precedente perdait 17,5 % en appels d'outils ; Bonsai 2 corrige l'essentiel.

**Pour la fusion** : Bonsai 2 est un excellent **System Two** local (raisonnement, code, maths, suivi
d'instructions, vision), et un excellent **enseignant** pour distiller un petit modele de decision.

## 4. Vitesse : ce que le materiel grand public donne

Debit batch 1 (tg128 = generation, pp512 = prefill), tokens/s :

| Materiel | Modele / format | tg128 | pp512 | Source |
|---|---|---|---|---|
| RTX 5090 32 Go | Bonsai 2 PQ2_0 / PTQ1_0 | 142,5 / 134,4 | 4 121 / 1 901 | livre blanc |
| RTX 4090 24 Go (Ada) | Bonsai 2 PQ2_0 / PTQ1_0 | 90,9 / **96,7** | 3 134 / 1 634 | livre blanc (PTQ1_0 plus rapide sur Ada) |
| L4 24 Go 72 W | Bonsai 2 PQ2_0 / PTQ1_0 | 29,7 / 32,1 | 778 / 468 | livre blanc |
| MacBook M5 Pro | Bonsai 2 PQ2_0 | 27,7 | 397 | livre blanc |
| RTX 5060 Ti 16 Go | Ternary-Bonsai 27B PQ2_0 | 44,4 (79 avec DSpark) | 1 029 | communaute |
| RTX 4080 Laptop 12 Go | Ternary-Bonsai 27B Q2_g64 | ~45 | ~285 | communaute (mahald) |
| RTX 4070 Ti Super 16 Go | Ternary-Bonsai 27B | 69,6 | 1 717 | communaute |
| GTX 1080 Ti 11 Go | Bonsai-27B Q1_0 / Ternary | 28,3 / 20,5 | 285 / 278 | communaute |
| RTX A2000 Laptop **4 Go** | Bonsai-8B Q1_0 / 4B / 1.7B | 63 / 70 / 129 | 1 387 / 2 375 / 5 064 | communaute |
| CPU i7-11800H (8 c.) | Bonsai-8B Q1_0 / 4B / 1.7B | 21 / 37 / 85 | 1 063 / 1 620 / 3 183 | communaute |
| iPhone 17 Pro Max | Bonsai-27B 1-bit (MLX) | ~11-12 | | PrismML |

**Pas de mesure publique pour la RTX 4060** : `docs/03-MATERIEL-profils.md` en donne une estimation par
bande passante (272 Go/s) : ~30 tok/s en PTQ1_0, ~25 en PQ2_0, ~40 en Q1_0 1-bit, a verifier avec
`scripts/bench.sh`. Le decodage est limite par la bande passante memoire (les poids sont relus a chaque
token) : c'est precisement pourquoi 1,76 bit/poids est decisif sur une carte 8 Go.

## 5. Memoire : le tableau qui decide du profil

Pic memoire du 27B mesure par PrismML (poids + activations + KV FP16 + ~1,2 Gio de surcout ; texte seul ;
+0,6 a 0,9 Gio si le projecteur vision est charge) :

| Modele | Format | Poids | ctx 4 k | ctx 10 k | ctx 100 k |
|---|---|---|---|---|---|
| Bonsai-27B 1-bit | `Q1_0` | 3,53 Gio | 4,8 Gio | 5,2 Gio | 10,8 Gio |
| Ternary-Bonsai-27B | `Q2_0` | 6,66 Gio | 7,8 Gio | 8,1 Gio | 13,7 Gio |
| **Bonsai 2 27B** | `PTQ1_0` | **5,52 Gio** (5,93 Go) | ~6,8 Gio (estime) | ~7,2 Gio | ~12,3 Gio |
| Bonsai 2 27B | `PQ2_0` | 6,75 Gio (7,25 Go) | ~8,0 Gio | ~8,4 Gio | ~13,5 Gio |
| ref. 27B "4-bit" | `UD Q4_K_M` | 15,7 Gio | 17,2 Gio | 17,6 Gio | 23,2 Gio |

Le cache KV 4-bit (`--cache-type-k q4_0 --cache-type-v q4_0`, `BONSAI_KV4=1`) divise la partie KV par
~3,5 (64 -> ~18 Kio/token) : a 8 k tokens elle ne pese plus que ~0,15 Gio. Un biais de "mean-centering"
calibre (`make_kv_bias.sh`) rattrape l'essentiel de la perte de qualite.

## 6. Fonctions utiles a la fusion

* **Cache de prompt** (`--cache-ram`, `cache_prompt: true`) : le prefixe commun (systeme + etat) n'est
  traite qu'une fois ; les branches System One ne coutent que leurs propres tokens. Sur un modele
  hybride, `--ctx-checkpoints` conserve des points de reprise de l'etat recurrent pour reutiliser un
  prefixe partiel.
* **Budget de reflexion** (`--reasoning-budget N`, `thinking_budget_tokens` par requete) : la fusion
  module ce budget selon le **risque** juge par System One (0 / 512 / 2 048 / 8 192 tokens).
* **Sortie JSON contrainte** (`response_format: json_schema`) : Bonsai repond aux questions typees en
  JSON strict quand il joue l'enseignant "lent" (`jev_clone/distill.py --mode think`).
* **Lecture de probabilites** (`n_probs`, `post_sampling_probs`, `grammar`) : Bonsai peut lui-meme jouer
  System One sans generer (mode "mono").
* **Decodage speculatif DSpark** (1,4-2,4x sur CUDA) : disponible pour Bonsai 27B v1 et Ternary 27B
  (drafter `dspark-dflash` ~0,6 Gio, `BONSAI_SPECULATIVE=1`), **pas encore pour Bonsai 2** ; force un
  seul slot et desactive le cache de prompt entre requetes (a reserver au mode "generation longue").
* **Vision** : envoyer `image_url` ; `--no-mmproj-offload` garde le projecteur en RAM (economise ~0,6-0,9 Gio
  de VRAM, image plus lente a encoder) ; `--image-max-tokens 1024` plafonne le cout.

## 7. Pourquoi Bonsai 2 est le bon "System Two" pour 8 Go

1. **Il tient** : 5,93 Go de poids, ~6,8 Gio au total a 4-8 k de contexte, la ou un Qwen3.8-27B "4-bit"
   demande 17 Gio et un IQ2_XXS 7,3 Go pour 12 points de moins.
2. **Il raisonne encore** : la compression classique casse d'abord le raisonnement long et les appels
   d'outils ; Bonsai 2 garde 98 % de la moyenne et ~96 % en maths/code.
3. **Il est agentique** : outils natifs, JSON contraint, 262 k de contexte, vision.
4. **Il est ouvert** (Apache 2.0) et **execute par llama.cpp** : la meme API HTTP sert System One et
   System Two, sur Linux, Windows, macOS.
5. **Sa faiblesse** (agentique long-horizon ~75 %, prefill plus lent en PTQ1_0) est exactement ce que la
   fusion compense : System One decoupe les taches en jugements courts et ne l'appelle qu'en cas de doute.

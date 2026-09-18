# 04 - References et sources

Recherche effectuee le 18 septembre 2026. Les depots GitHub ont ete clones et lus en entier (livres
blancs PDF inclus) ; les pages marquees *(resume)* n'etaient pas accessibles depuis l'environnement de
redaction et ont ete exploitees via les extraits fournis par le moteur de recherche : verifier les
chiffres directement a la source avant de les citer.

## Jev / TypeSafe AI
* TypeSafe AI, *Introducing System One Models & Jev* - https://typesafe.ai/blog/introducing-system-one-models-and-jev *(resume)*
* Documentation TypeSafe : introduction, API, primitives, confidence, patterns, `jev-1.13` - https://docs.typesafe.ai/ *(resume)*
* BusinessWire, *TypeSafe AI Emerges From Stealth With $40M...* (15 sept. 2026) - https://www.businesswire.com/news/home/20260915525333/en/ *(resume)*
* The Register, *TypeSafe AI debuts model for machines that plays Doom* (16 sept. 2026) - https://www.theregister.com/ai-and-ml/2026/09/16/typesafe-ai-debuts-model-for-machines-that-plays-doom/5296711 *(resume)*
* SiliconANGLE, *TypeSafe AI exits stealth with $40M* - https://siliconangle.com/2026/09/16/typesafe-ai-exits-stealth-with-40m-to-build-ai-for-use-by-software/ *(resume)*
* The Neuron, *TypeSafe JEV Explained* - https://www.theneuron.ai/explainer-articles/typesafe-jev-system-one-models-explained/ *(resume)*
* DataCamp, *Jev: TypeSafe's System One Model That Never Hallucinates* - https://www.datacamp.com/blog/system-one-models-jev *(resume)*
* Archer Hume, *Jev's Architecture Unmasked* - https://archerhume.com/posts/jevs-architecture-unmasked/ *(resume ; base des reconstructions reflex/decider)*
* lilting, *TypeSafe Jev: $0.042/1M Decision Model vs Auto-Regressive LLMs and MDLM* - https://lilting.ch/en/articles/typesafe-ai-jev-system-one-model *(resume)*
* Developers Digest, *TypeSafe Jev: the First Decision-Only Model Class, Benchmarked and Priced* - https://www.developersdigest.tech/blog/typesafe-jev-system-one-models-release-guide-2026 *(resume)*
* Valyu (dev.to), *How to Use Jev: A practical guide* - https://dev.to/valyuai/how-to-use-jev-a-practical-guide-to-typesafes-system-one-model-g5e *(resume)*
* Vercel AI Gateway, *Jev* - https://vercel.com/ai-gateway/models/jev *(resume)*
* pjburnhill, *Comprehensive project reference for TypeSafe Jev* (gist, lu) - https://gist.github.com/pjburnhill/adf8d28efcad9df037bfdece178ef965
* themsquared, *jev-benchmark* (lu) - https://github.com/themsquared/jev-benchmark
* OmniJev, *awesome-jev* (clone) - https://github.com/OmniJev/awesome-jev ; yibie/awesome-jev ; hellogumbo/awesome-jev

## Reproductions ouvertes de Jev (clonees et lues)
* Mapika, *decider* (Qwen3.5-2B-Base) - https://github.com/Mapika/decider ; poids https://huggingface.co/Mapika/decider-2b
* kshetrajna12, *reflex* (Qwen3.5-4B, ARCHITECTURE.md) - https://github.com/kshetrajna12/reflex
* shamazharikh, *qwen-rlcd* (prefix-fork sur Qwen3.5-0.8B) - https://github.com/shamazharikh/qwen-rlcd
* akash-kamat, *system-one-gemma* (Gemma 3 270M) - https://github.com/akash-kamat/system-one-gemma
* bnsd55, *openjev / jevmlx* (MLX) - https://github.com/bnsd55/openjev
* rorshopping, *parallel-decisions* (MLX) - https://github.com/rorshopping/parallel-decisions
* Articles cites par awesome-jev sur la calibration par regle de score propre : *RewardingDoubt* (https://github.com/pasta99/RewardingDoubt), *RLCR* (https://github.com/damanimehul/RLCR), arXiv 2507.16806, 2607.04332, 2607.08046

## Bonsai / PrismML
* PrismML, *Ternary Bonsai 2 27B* (livre blanc PDF, sept. 2026, lu integralement) - `Bonsai-demo/bonsai-2-27b-whitepaper.pdf`
* PrismML, *Bonsai 27B* (livre blanc PDF, juil. 2026, lu) - `Bonsai-demo/bonsai-27b-whitepaper.pdf`
* PrismML, collection Hugging Face **Bonsai 2** - https://huggingface.co/collections/prism-ml/bonsai-2 (lien fourni par l'utilisateur ; domaine non accessible depuis l'environnement de redaction)
* PrismML sur X, annonce de Ternary Bonsai 2 27B, 17 sept. 2026 23:03, table de retention par capacite (capture fournie par l'utilisateur) - https://x.com/PrismML/status/2100692248480596348
* PrismML, *Introducing Bonsai 2 27B* - https://prismml.com/news/bonsai-2-27b *(resume)* ; PR Newswire, *PrismML Launches Bonsai 2 27B, Its Most Capable Model Yet* (17 sept. 2026) *(resume)*
* PrismML, *PrismML Announces 1-bit Bonsai 27B* (14 juil. 2026) - https://prismml.com/news/prismml-releases-bonsai-27b *(resume)* ; *Introducing Ternary Bonsai* (16 avr. 2026) - https://prismml.com/news/ternary-bonsai *(resume)*
* PrismML-Eng, *Bonsai-demo* (clone, lu : README, MODEL-FORMATS.md, KV-CACHE.md, SPECULATIVE.md, AGENTS.md, VISION.md, TOOLS.md, environment_variables.md, community-benchmarks/) - https://github.com/PrismML-Eng/Bonsai-demo
* PrismML-Eng, *llama.cpp* fork, branche `prism`, release `prism-b10683-d8f26ee` (binaires telecharges et executes) - https://github.com/PrismML-Eng/llama.cpp
* Hugging Face : prism-ml/Ternary-Bonsai-2-27B-gguf, Ternary-Bonsai-2-27B-mlx-2bit, Bonsai-27B-gguf, Ternary-Bonsai-27B-gguf, Ternary-Bonsai-{8B,4B,1.7B}-gguf, Bonsai-{8B,4B,1.7B}-gguf *(resume : domaine non accessible)*
* mahald, *ternary-bonsai-27b-gguf-llamacpp-cuda* (clone, lu ; RTX 4080 Laptop) - https://github.com/mahald/ternary-bonsai-27b-gguf-llamacpp-cuda
* Astezelex, *bonsai-27b-16gb-bench* (RTX 5060 Ti) - https://github.com/Astezelex/bonsai-27b-16gb-bench
* john-rocky, *coreai-model-zoo* issue #28 (portage Bonsai 2, noyaux Hadamard + GEMM ternaire) - https://github.com/john-rocky/coreai-model-zoo/issues/28
* MarkTechPost, *PrismML Releases Bonsai 27B* - https://www.marktechpost.com/2026/07/14/... *(resume)* ; Kaitchup, *Bonsai 27B Review* *(resume)* ; Kubesimplify, *Bonsai 27B benchmark: RTX PRO 6000 vs DGX Spark* *(resume)* ; DataCamp, *How to Run Bonsai 27B Locally on 8GB Memory* *(resume)*

## OrcaBonsai (ablation de refus)
* Continuum-AI-Corp / OrcaRouter research team, *OrcaBonsai-27B-Uncensored* (clone, lu : README, `scripts/export_gguf_lora.py`, adaptateur GGUF inspecte, empreintes) - https://github.com/Continuum-AI-Corp/OrcaBonsai-27B-Uncensored
* Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction* (2024) - https://arxiv.org/abs/2406.11717 (la technique d'ablation de direction)

## Qwen
* QwenLM, *Qwen3.8* (lu) - https://github.com/QwenLM/Qwen3.8 ; Qwen/Qwen3.8-27B (14 aout 2026), Qwen/Qwen3.8-2.4T-A95B (12 aout 2026)
* Qwen3.5 (0.8B / 2B / 4B / 9B, 2 mars 2026) : bases des clones ouverts ; MindStudio, *Qwen3.8-27B Explained: Hybrid Attention, 262K Context* *(resume)*

## llama.cpp
* ggml-org, *tools/server/README.md* (lu) : `/completion` (`n_probs`, `post_sampling_probs`, `grammar`, `cache_prompt`, `id_slot`), `/tokenize`, `/props`, `--parallel`, `--cache-ram`, `--ctx-checkpoints`, `--slot-prompt-similarity`, `--kv-unified` - https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
* PR ggml-org/llama.cpp#27779 (FWHT F16 CPU, upstreaming de Bonsai 2) ; PR #25173 (DSpark)

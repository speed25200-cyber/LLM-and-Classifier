# 09 - Est-ce vraiment une fusion LLM + Jev ? Verification honnete, et ce qui la rend profonde

## 1. Trois niveaux de fusion, et ou nous en sommes

| Niveau | Ce que ca veut dire | Etat dans ce depot |
|---|---|---|
| **Systeme** : deux modeles, une API, une boucle | routeur par confiance, escalade, journal, distillation S2 -> S1 | fait (`fusion.py`, `distill.py`, `calibrate.py`) |
| **Inference** : un modele *intervient dans le calcul* de l'autre | S2 appelle S1 comme outil ; S1 pilote la reflexion de S2, choisit parmi ses candidats, verifie ses reponses ; etat encode une fois et partage (cache) | fait (`tools.py`, `guided.py`, mode mono) |
| **Modele** : un seul reseau, deux modes | le meme backbone sert de decideur (lecture restreinte, calibree) et de generateur (tete LM) ; entraine sur les deux objectifs | fait pour le petit modele (`training/ --sft-data`) ; impossible sur Bonsai lui-meme (QAT proprietaire), donc Bonsai reste le "grand cerveau" au-dessus |

Verdict : oui, c'est bien une fusion, et depuis ce commit elle va jusqu'au niveau inference sur Bonsai et
jusqu'au niveau modele sur le petit modele. Ce qui la distingue d'un simple "classificateur devant un LLM" :
1. **Le meme etat est encode une fois** et sert aux decisions et a la generation (cache de prefixe, mode mono).
2. **System One est dans la boucle de decodage de System Two** (`guided.py`) : il decide quand Bonsai a
   fini de reflechir, quel candidat garder, s'il faut reessayer. Sur une RTX 4060 a ~30 tok/s, la
   reflexion est le poste de latence numero un (2 048 tokens = ~70 s) ; c'est lui que le clone reduit.
3. **System Two est dans la boucle de System One** : il l'enseigne (distillation, trajectoires) et il le
   consulte (outils `judge_*`).
4. **Un modele, deux modes** au niveau rapide : le Qwen3.5 entraine repond en une passe (decisions) *et*
   genere des reponses courtes distillees de Bonsai ; Bonsai n'est appele que pour le raisonnement long.

## 2. Generation guidee par System One (`jev_clone/guided.py`)

```
prompt -> <think> -> Bonsai genere 192 tokens -> S1 : "reponse deja determinee ?" / "tourne en rond ?"
                                 |  non                                   | oui
                                 v                                        v
                          192 tokens de plus ...                </think> -> Bonsai repond
```
* `think_adaptive` : arret de la reflexion des que le clone est sur (seuil 0,8) ou detecte une boucle ;
  budget maximal en garde-fou. Chaque controle coute un appel S1 (~100 ms) contre ~6 s par tranche de 192
  tokens sur 4060 : rentable des que la reflexion s'arrete une tranche plus tot.
* `best_of_n` : N reponses courtes en parallele (slots), classees par N nouls du clone en une passe.
* `verified` : reponse, verification, seconde tentative avec plus de budget si elle echoue.
* Tout passe par `apply-template` + `completion` brute avec cache de prefixe : Bonsai 2 sur le fork PrismML
  ou n'importe quel GGUF. Verifie ici (mecanique) contre un `llama-server` reel.

Ce qu'il faut mesurer sur la 4060 (`eval/`) : sur 50 questions de maths/code, tokens de reflexion et
exactitude avec `--reasoning-budget 2048` seul, contre `think_adaptive` (cible : -40 % de tokens a
exactitude egale, mesure par Bonsai lui-meme en mode reflexion complete comme reference).

## 3. "SOTA avec une RTX 4060 ou 5060, pas plus" : ce que ca veut dire concretement

Les deux cartes ont 8 Go : memes profils, memes modeles. Ce qui change avec la 5060 : bande passante 448 Go/s
contre 272 (decodage ~1,6x plus rapide : Bonsai 2 PTQ1_0 ~45-50 tok/s estimes contre ~30), architecture
Blackwell (`sm_120`, build CUDA 12.8 ou 13.3 du fork, choisie automatiquement par `setup.sh`). Sur 8 Go,
PTQ1_0 reste le seul packing qui laisse la place au clone ; la 5060 Ti 16 Go prend le profil `gpu-16gb`
(PQ2_0, contexte 64 k, clone 8B).

A ce budget, "SOTA" se decline ainsi, et chaque ligne est mesurable :

| Composant | Meilleur choix connu a 8 Go (sept. 2026) | Pourquoi |
|---|---|---|
| Generateur / raisonneur local | Bonsai 2 27B PTQ1_0 (5,93 Go, 98,2 % de Qwen3.8-27B) | aucun autre 27B ne tient en 8 Go a cette qualite ; IQ2_XXS perd 12 points |
| Sans refus (agent) | + adaptateur OrcaBonsai (rang 1, 9,7 Mo) | poids inchanges, echelle reglable, garde-fou deplace dans S1 |
| Decideur rapide | clone Qwen3.5-0,8B/2B entraine (RLCD-lite + distillation Bonsai + Qwen3.8 FP16) | les clones ouverts atteignent ECE 0,037-0,039 ; Jev annonce 0,031 ; cible <= 0,03 |
| Garanties | ensembles conformes + porte binomiale | Jev n'en publie pas |
| Latence de decision | 100-250 ms (llama.cpp), < 50 ms vise (torch, passe unique) | Jev : 70-500 ms via API |
| Reflexion | budget adaptatif pilote par S1 | -40 % de tokens vises a exactitude egale |
| Agent | navigateur a deux vitesses, 80-90 % de pas rapides | Bonsai 2 : 52,8 sur Terminal-Bench 2.1 ; S1 decoupe en pas courts |

La preuve, pas la promesse : `eval/SCOREBOARD.md`. Tant que ses colonnes sont vides, "SOTA" est une
hypothese ; quand elles sont remplies sur votre 4060, c'est un fait ou un ecart a corriger.

## 4. Ce qui reste, par ordre d'impact

1. Mesurer le niveau 0 sur la 4060 (banc Jev, `jev bench`, generation guidee sur 50 questions).
2. Entrainer le modele fuse (decisions + generation) sur l'A100, exporter, remesurer.
3. Passe unique sur llama.cpp (`libllama.so`, `llama_memory_seq_cp`) pour Bonsai en mode mono.
4. Etat multimodal pour le clone (captures d'ecran), bureau entier (arbre d'accessibilite).

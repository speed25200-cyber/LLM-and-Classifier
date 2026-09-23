# 09 - Est-ce vraiment une fusion LLM + Jev ? Verification honnete, et ce qui la rend profonde

Ce qui tourne dans Prophet Studio et ce qui n'existe que dans la bibliotheque : [docs/12](12-PROPHET-STUDIO.md),
sections 5.5 et 5.6. Rien n'a encore tourne sur les vrais modeles.

## 1. Trois niveaux de fusion, et ou nous en sommes

| Niveau | Ce que ca veut dire | Etat dans ce depot |
|---|---|---|
| **Systeme** : deux modeles, une API, une boucle | S1 decide la voie, le budget, les outils, le risque ; escalade vers S2 ; journal ; distillation S2 -> S1 | dans Studio : `prophet.py` (voie directe / agent, budget, garde-fou, verification), `calibrate.py` ; `fusion.py` (`FusionRouter`) : bibliotheque seulement (`jev serve`) ; `distill.py` : code pret, jamais execute sur les vrais modeles |
| **Inference** : un modele *intervient dans le calcul* de l'autre | S2 appelle S1 comme outil ; S1 fixe la reflexion de S2 ; S1 pilote le decodage de S2, choisit parmi ses candidats, verifie ses reponses ; etat lu une fois par question grace au cache | dans Studio : `judge_*` (`tools.py`), budget de reflexion fixe **avant** la generation, verification finale, mode mono ; `guided.py` (S1 pendant le decodage, meilleur de N) : bibliotheque seulement |
| **Modele** : un seul reseau, deux modes | le meme backbone sert de decideur (lecture restreinte, calibree) et de generateur (tete LM) ; entraine sur les deux objectifs | code pret pour le petit modele (`training/train_lora_rlcd.py --sft-data`), jamais execute, aucun clone entraine ; impossible sur Bonsai lui-meme (QAT proprietaire), donc Bonsai reste le "grand cerveau" au-dessus |

Verdict : c'est une fusion au niveau systeme et, en partie, au niveau inference (Bonsai consulte S1 pendant sa boucle ;
S1 fixe son budget de reflexion et verifie ses reponses). Le pilotage du decodage par S1 (`guided.py`) et le niveau
modele sont ecrits mais ne sont pas utilises par Studio. Ce qui distingue le montage d'un simple "classificateur devant
un LLM" :
1. **L'etat d'une question est lu une fois** et partage par ses questions (cache de prefixe) ; en mode mono, Bonsai sert
   aussi les decisions.
2. **System Two est dans la boucle de System One** : il l'enseigne (distillation, trajectoires, chaine prete) et il le
   consulte (outils `judge_*`).
3. **System One encadre System Two** : voie, budget de reflexion, outils exposes, garde-fou avant chaque action,
   verification.
4. Dans la bibliotheque (`guided.py`) : **System One dans la boucle de decodage de System Two**, pour decider quand
   Bonsai a fini de reflechir ; sur une RTX 4060 a ~30 tok/s estimes, 2 048 tokens de reflexion = ~70 s, c'est le poste
   que ce mode vise.

## 2. Generation guidee par System One (`jev_clone/guided.py`, bibliotheque seulement)

```
prompt -> <think> -> Bonsai genere 192 tokens -> S1 : "reponse deja determinee ?" / "tourne en rond ?"
                                 |  non                                   | oui
                                 v                                        v
                          192 tokens de plus ...                </think> -> Bonsai repond
```
* `think_adaptive` : arret de la reflexion des que le clone est sur (seuil 0,8) ou detecte une boucle ;
  budget maximal en garde-fou. Chaque controle coute un appel S1 contre une tranche de 192 tokens de Bonsai :
  rentable des que la reflexion s'arrete une tranche plus tot.
* `best_of_n` : N reponses courtes en parallele (slots), classees par N nouls du clone en une passe.
* `verified` : reponse, verification, seconde tentative avec plus de budget si elle echoue.
* Tout passe par `apply-template` + `completion` brute avec cache de prefixe. Test en direct : `tests/test_guided.py`
  (`JEV_TEST_SERVER`), saute sans serveur.

Ce qu'il faudrait mesurer avant de le brancher (`eval/`) : sur 50 questions de maths/code, tokens de reflexion et
exactitude avec un budget fixe de 2 048, contre `think_adaptive` (cible : -40 % de tokens a exactitude egale).

## 3. "SOTA avec une RTX 4060 ou 5060, pas plus" : ce que ca veut dire concretement

Les deux cartes ont 8 Go : memes modeles. Ce qui change avec la 5060 : bande passante 448 Go/s contre 272 (estimation du
planificateur pour Bonsai 2 PTQ1_0 : 47-59 tok/s contre 28-36), architecture Blackwell (`sm_120`, build CUDA 12.8 ou 13.x
du fork, choisie automatiquement par Studio et par `setup.sh`). Sur 8 Go, PTQ1_0 est le packing de Bonsai 2 qui tient
entier sur le GPU ; le classifieur passe alors sur CPU (priorite equilibre). La 5060 Ti 16 Go prend Bonsai 2 PQ2_0, 32 k
de contexte et le classifieur 4B sur GPU (Studio et profil `rtx5060ti-16gb`).

A ce budget, "SOTA" se decline ainsi, et chaque ligne est mesurable :

| Composant | Meilleur choix connu a 8 Go (sept. 2026) | Pourquoi |
|---|---|---|
| Generateur / raisonneur local | Bonsai 2 27B PTQ1_0 (5,93 Go, 98,2 % de Qwen3.8-27B) | aucun autre 27B ne tient en 8 Go a cette qualite ; IQ2_XXS perd 12 points |
| Sans refus (agent) | + adaptateur OrcaBonsai (rang 1, 9,7 Mo) | poids inchanges, echelle reglable, garde-fou deplace dans S1 ; non charge par Studio |
| Decideur rapide | aujourd'hui Ternary-Bonsai 1.7B zero-shot ; vise : clone Qwen3.5-0,8B/2B entraine (RLCD-lite + distillation Bonsai) | les clones ouverts atteignent ECE 0,037-0,039 ; Jev annonce 0,031 ; cible <= 0,03 |
| Garanties | ensembles conformes + porte binomiale (`conformal.py`, bibliotheque seulement) | Jev n'en publie pas |
| Latence de decision | estimee 100-350 ms + prefill (S1 sur CPU, RTX 5060), 40-150 ms sur GPU ; < 50 ms vise (`engine_torch.py`, bibliotheque, jamais execute) | Jev : 70-500 ms via API |
| Reflexion | budget fixe par S1 selon risque et raisonnement ; adaptatif pendant le decodage : `guided.py` (bibliotheque) | -40 % de tokens vises a exactitude egale |
| Agent | navigateur et bureau a deux vitesses | Bonsai 2 : 52,8 sur Terminal-Bench 2.1 ; part des pas rapides non mesuree |

La preuve, pas la promesse : `eval/SCOREBOARD.md`. Tant que ses colonnes sont vides, "SOTA" est une
hypothese ; quand elles sont remplies sur votre carte, c'est un fait ou un ecart a corriger.

## 4. Ce qui reste, par ordre d'impact

1. Mesurer le niveau 0 sur la RTX 5060 : `scripts/measure_rtx5060.ps1` (duo reel), banc Jev, `jev bench`.
2. Entrainer le clone (decisions + generation) sur l'A100, exporter, importer dans Studio, calibrer, remesurer.
3. Brancher dans Studio ce qui est mesure comme utile (generation guidee, portes conformes).
4. Passe unique sur llama.cpp (`libllama.so`, `llama_memory_seq_cp`) pour Bonsai en mode mono ; etat multimodal pour le
   clone (captures d'ecran).

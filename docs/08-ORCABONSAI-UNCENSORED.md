# 08 - OrcaBonsai-27B-Uncensored : Bonsai 2 27B avec ablation de refus a l'execution

Depot : https://github.com/Continuum-AI-Corp/OrcaBonsai-27B-Uncensored (OrcaRouter research team, Apache 2.0,
commit `18cfd60` du 18 septembre 2026). Verifie ici : clone du depot, lecture du README et de
`scripts/export_gguf_lora.py`, inspection de l'adaptateur GGUF avec `gguf-py`, empreintes SHA-256.
**Non verifie ici** : l'execution contre le vrai modele (pas de GPU ni de poids Bonsai dans l'environnement).

## 1. Ce que c'est, exactement

Ce n'est **pas un autre modele** : c'est Ternary Bonsai 2 27B, poids **bit-identiques**, plus une
**projection appliquee a l'execution** sur chaque ecriture dans le flux residuel :

```
y <- y - alpha * (y . r) * r        r = direction de refus (vecteur unitaire, 5120 dims, base "non tournee")
```

129 sites : 64 `ffn_down` (MLP), 48 `ssm_out` (attention lineaire), 16 `attn_output` (attention complete),
1 `token_embd`. Pourquoi a l'execution et pas dans les poids : orthogonaliser une matrice ternaire donne une
matrice dense ; la re-quantifier en ternaire arrondit l'edition (1,4 % de la norme contre un pas de grille
1,7 a 2,4 fois un poids typique) et en laisse 99,4 % en place. D'ou l'idee : garder les poids compresses,
changer le comportement dans le graphe.

Pour llama.cpp, le meme operateur est livre comme **adaptateur LoRA de rang 1** (`gguf/bonsai-abliterate-lora.gguf`,
9 682 464 octets, 258 tenseurs F32, `general.architecture = qwen35`, `adapter.type = lora`, `alpha = 1`) :
`W' = W - r (r^T W)` est de rang 1, donc `A = r^T W`, `B = -r`. llama.cpp ajoute `B(Ax)` a la sortie de la
matmul de base sans jamais fusionner : exact a 1,75 bit/poids, reversible, reglable (`--lora-scaled f:s`,
et par requete `"lora": [{"id": 0, "scale": s}]`).

Deux points techniques que les auteurs disent avoir mesures (coherents avec ce que j'ai lu du fork) : l'adaptateur
est ecrit dans la base non tournee parce que `build_lora_mm` du fork tourne l'activation (Hadamard) uniquement
pour la matmul de base et donne l'activation brute a la branche LoRA ; `ssm_out` ne demande aucune
permutation. Teste par eux **sur PTQ1_0 seulement** (PQ2_0 : "devrait s'appliquer", non execute).

## 2. Ce que les auteurs mesurent (et les reserves)

| Jeu | n | Base : refus | Ablate : refus | "Caveat" (repond avec avertissement) |
|---|---|---|---|---|
| AdvBench | 100 | 99,0 % | 6,0 % | 56 % |
| JailbreakBench | 100 | 96,0 % | 4,0 % | 52 % |
| StrongREJECT | 150 | 99,3 % | 3,3 % | 45 % |
| HarmBench | 150 | 98,7 % | 7,3 % | 48 % |
| XSTest-safe (sur-refus benin) | 250 | 5,2 % | 0,4 % | |
| JBB-benign (sur-refus benin) | 100 | 25,0 % | 0,0 % | |
| MMLU / GSM8K / CMMLU (capacites) | | 76,7 / 87,3 / 76,2 | 77,7 / 86,0 / 75,6 | |

Reserves a garder en tete : (1) la direction a ete estimee sur le modele BF16, son transfert a travers la
quantification QAT "n'a pas ete completement mesure" ; (2) les evaluations ont tourne sur une expansion fp16
"depliee" du pack, pas sur les noyaux ternaires (les auteurs disent que les distributions coincident a trois
decimales sur des sondages) ; (3) le classifieur de refus est a base de regles sur les premiers mots, pas un
juge LLM ; (4) environ la moitie des reponses "non refusees" sont enveloppees d'avertissements.

## 3. Pourquoi c'est utile ici : le sur-refus casse un agent

Pour un agent qui pilote un ordinateur, le probleme n'est pas de faire dire n'importe quoi au modele, c'est le
**sur-refus** : 25 % de refus sur des demandes benignes (JBB-benign) et 5 % sur XSTest, c'est un agent qui
s'arrete au milieu d'une tache d'administration systeme, de securite ("classe cet appel d'outil : destructif ?"),
de moderation (il doit lire du contenu toxique pour le classer) ou de test d'intrusion autorise. L'ablation
ramene ces deux chiffres a 0,4 % et 0 % pour ~1 point de capacite.

Effet par role de la fusion :
* **System Two (generation, outils, computer use)** : c'est la ou l'adaptateur compte. Cout : 129 matmuls de
  rang 1 par token, quelques Mo de VRAM, latence negligeable.
* **Enseignant (distillation)** : le professeur etiquette aussi les etats sensibles au lieu de les refuser ;
  c'est precisement ce dont un clone de moderation ou de securite a besoin.
* **System One (clone dedie)** : aucun effet, ce n'est pas le meme serveur.
* **Mode mono (Bonsai juge lui-meme)** : les lectures de probabilites peuvent se faire **sans** l'adaptateur sur
  le meme serveur (`LlamaCppBackend(..., lora=[{"id": 0, "scale": 0.0}])`) pendant que la generation le garde.

## 4. Consequence pour la securite de l'agent : le garde-fou change de place

Avec l'adaptateur, Bonsai ne refusera presque plus rien. Le garde-fou n'est donc plus dans le modele de
generation, il est dans **les decisions typees et le code** :
1. le garde-fou du clone (`jev_clone/guard.py` : `tool_risk`, `risk`, `policy_violation` ; docs/12, 5.5.5) avant les
   commandes, le code, les outils crees, la navigation, les pas du computer use et l'ecriture de fichiers executables ;
   au-dela des seuils, **confirmation humaine**, obligatoire pour les verdicts les plus risques. Dans Studio, le mode
   Jamais demander ne consulte pas le clone pour les outils de Prophet. C'est la tache du benchmark jev-benchmark
   (readonly / destructive / privileged / exfiltration). La porte a risque controle de `conformal.py` n'est branchee
   nulle part ;
2. la verification System One apres chaque action rapide du computer use, les deux echecs consecutifs qui rendent la
   main a Bonsai ;
3. l'executeur : aucune saisie inventee (le texte tape vient des slots) ; il n'y a pas de liste blanche d'actions, de
   domaines ou de commandes ;
4. le journal (`<espace de travail>/.prophet/ledger.jsonl` pour Prophet, `<donnees>/runs/trajectories.jsonl` pour le
   computer use de Studio, `runs/ledger.jsonl` pour `jev serve`) ;
5. l'echelle : 1,0 suffit (projection exacte) ; 2,0 "retourne les cas tetus" au prix de la qualite ; >= 3
   degrade puis effondre le modele. Le profil fixe 1,0.

Le clone System One, lui, reste calibre sur *vos* etiquettes : c'est lui qui decide ce qui est autorise, pas
le modele qui genere.

## 5. Installation

```bash
./scripts/fetch_orcabonsai.sh          # clone --depth 1 dans third_party/orcabonsai, verifie le SHA-256 de l'adaptateur
PROFILE=scripts/profiles/rtx4060-8gb-orcabonsai.env ./scripts/start_bonsai.sh   # = profil qualite + --lora-scaled adaptateur:1.0
```
Verification que l'adaptateur est bien dans le graphe (methode des auteurs) : une meme invite avec
`"lora": [{"id": 0, "scale": 0}]` doit reproduire le modele publie, avec `scale: 100` le modele doit
produire n'importe quoi ; si ni l'un ni l'autre, l'adaptateur n'est pas applique (nom de tenseur non route).
`GET /lora-adapters` liste les adaptateurs charges ; `POST /lora-adapters` change l'echelle par defaut sans
redemarrer. Seul le fichier GGUF est utilise : rien du code Python (MLX, Apple) du depot n'est execute.

Empreintes verifiees le 18 septembre 2026 :
```
f1669534803d340a496015f5c45125f3437b4d13ec764f40e34488ce83967f42  gguf/bonsai-abliterate-lora.gguf   (identique au README)
06ab5afef2529d9302748507c85913d55dff53b7b739ccc98d8db7716ebd36b2  directions/refusal_dir_fp32.bin
bd21f44b08dacaf6b115accbadb8f23e09c4385862c525254d91533e78f2bccd  directions/refusal_dir.safetensors
```
`fetch_orcabonsai.sh` refuse un fichier dont l'empreinte a change.

## 6. Ce qu'il faut mesurer vous-meme avant de l'adopter

1. Sur 20 invites benignes de votre domaine que le modele publie refuse ou hedge : taux de refus a `scale 0`
   puis `1.0` (le seul chiffre qui justifie l'adaptateur pour vous).
2. `eval/jev_benchmark.py` en mode mono avec `scale 0` et `scale 1` : la classification du risque ne doit
   pas bouger (si elle bouge, la direction de refus touche aussi le jugement : garder `scale 0` pour System One).
3. Une petite batterie de capacites (10 problemes de code, 10 de maths) a `scale 0` et `1` : ecart attendu
   ~1 point, au-dela quelque chose est mal applique (PQ2_0 non teste par les auteurs).

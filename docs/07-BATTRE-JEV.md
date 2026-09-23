# 07 - Faire mieux que Jev : ce qui est mesurable, ce qui est inedit, ce qu'il faut faire

## 1. Le cadre honnete

"SOTA partout" n'est pas une promesse que l'on peut tenir a l'avance ; c'est un resultat qu'on mesure. Jev
n'a publie que quelques chiffres (accuracy ~68 % sur un benchmark interne non reproductible, ECE 0,031 sur
1 200 items MMLU, latence 70-500 ms, 255 options, 32 k tokens d'etat) et un benchmark independant existe
(60 cas de risque d'appel d'outil : 91,7 %, p50 422 ms, aucune erreur a confiance 1,0). Le tableau de bord
`eval/SCOREBOARD.md` fixe, pour chacun de ces chiffres, la valeur de Jev et la cible a battre ; les scripts
de `eval/` remplissent les colonnes avec **le meme code de mesure que celui qui a mesure Jev**.

Ce qui restera difficile : la largeur zero-shot (Jev est un generaliste entraine par une equipe ex-OpenAI
sur des donnees inconnues). Ce qui est gagnable : la latence (local, sans reseau, passe unique), la
calibration sur un domaine, la robustesse aux cas ambigus par distillation d'un 27B en mode reflexion, et
tout ce que Jev **ne fait pas**.

## 2. Sept axes ou ce depot peut etre devant, dont cinq inedits par rapport a Jev

| Axe | Jev | Ce depot | Ou |
|---|---|---|---|
| 1. Latence | 70-500 ms via API (p50 mesure 422 ms) | llama.cpp local, estime et non mesure : 40-150 ms (S1 sur GPU), 100-350 ms + prefill d'un etat neuf (S1 sur CPU, RTX 5060) ; **torch en passe unique : cible < 50 ms** pour 10 questions sur 0,8B (une passe GPU, aucune requete HTTP) | `engine_torch.py` |
| 2. Garanties formelles | probabilites calibrees, aucune garantie publiee | **ensembles de prediction conformes** (couverture >= 1-alpha garantie) et **porte a risque controle** (erreur <= alpha a 1-delta, borne binomiale exacte) | `conformal.py` (bibliotheque, branche nulle part) |
| 3. Candidats illimites | 255 options | **rang de N candidats** par N nouls independants en une passe (elements d'une page, documents, outils) | `tools.py: judge_rank` (50 candidats au plus), `presets.rerank` |
| 4. Le LLM consulte le decideur | produit isole | Bonsai appelle `judge_*` pendant qu'il raisonne / pilote le navigateur | `tools.py`, `computer_use.py` |
| 5. Boucle fermee locale | modele fixe, cloud | Bonsai etiquette les cas douteux, le clone est re-entraine sur la distribution d'etats reelle (DAgger) | `distill.py`, `trajectory_to_examples` |
| 6. Etat multimodal | non documente | images dans l'etat avec un clone VL (Qwen3.5-VL), captures pour Bonsai | `engine_torch.py` (a etendre), `examples/browser_agent.py --vision` (pas dans Studio) |
| 7. Cout / confidentialite | 0,042 $/M tokens, donnees chez TypeSafe | 0 $, rien ne sort de la machine | - |

Axes 2, 4, 5 et la combinaison 1+3 n'existent pas dans l'offre Jev telle que documentee.

## 3. Comment atteindre les cibles chiffrees (plan A100, 4 semaines)

**Semaine 1 : donnees.** (a) Melange public de decisions a l'echelle de decider (~60 jeux, ~900 k exemples :
intentions, routage, moderation, NLI, QCM, sentiment, notation ; `training/make_public_mix.py` a etendre) ;
(b) distillation **double enseignant** : Bonsai 2 27B en lecture (distributions) + **Qwen3.8-27B FP16 en mode
reflexion** (l'A100 80 Go le charge en bf16 avec vLLM : 54 Go) pour les etiquettes dures des cas ambigus ;
(c) 5 000 etats de votre domaine, dont 1 000 verifies a la main ; (d) cas adversariaux synthetiques
(un appel risque enveloppe de langage anodin, comme dans jev-benchmark), generes par Qwen3.8 puis filtres.

**Semaine 2 : entrainement.** Etudiant **Qwen3.5-4B-Base** (fine-tuning complet bf16, 2 epoques, NLL + KL
enseignant, permutations x2, options "other", niveaux de score isoles a la decider) ; variante 2B pour la
4060 "vitesse", 0,8B pour la 4060 "qualite". Evaluation en cours de route sur le tableau de bord.

**Semaine 3 : calibration et garanties.** Temperature par primitive ; ensembles conformes a alpha = 0,05 et
0,10 sur 1 000 cas de domaine ; porte a risque controle ; **MMLU 1 200** et **jev-benchmark x5** ; latence
torch vs llama.cpp sur la 4060.

**Semaine 4 : agent.** 200 taches navigateur (formulaires, recherches, navigation) : trajectoires, escalades
re-etiquetees par Bonsai, re-entrainement, mesure du taux de pas rapides et du temps par tache.

Ordres de grandeur A100 : distillation 50 k etats en ~6 h (Bonsai) + 10 k etats difficiles en ~10 h
(Qwen3.8 reflexion) ; entrainement 4B complet sur ~300 M tokens en ~12-16 h.

## 4. Ce qui est deja code pour ce plan

* `eval/jev_benchmark.py` : les 60 cas et les sorties reelles de Jev, un seul code de mesure (accuracy par
  difficulte, p50/p95, ECE 10 bacs, erreurs a confiance 1,0). Verifie : il reproduit exactement les chiffres
  publies de Jev a partir de ses sorties brutes (91,7 %, 421,6 ms, ECE 0,0712, 0 erreur a 1,0).
* `eval/mmlu_ece.py` : le protocole MMLU-1200 (temperature sur une moitie, mesure sur l'autre).
* `jev_clone/conformal.py` : LAC conforme + porte a borne binomiale, testes (couverture 90 % obtenue a 89-93 %).
* `jev_clone/engine_torch.py` : moteur en passe unique (masque par blocs / batch sur cache duplique).
* `training/train_lora_rlcd.py --full`, `make_public_mix.py`, `distill.py`, le notebook A100.

## 5. Ce qui manque encore (par ordre d'impact)

1. **Passe unique sur llama.cpp** (pour Bonsai / GGUF sans torch) : `libllama.so` du fork expose
   `llama_memory_seq_cp` et les batches multi-sequences ; un binding ctypes de ~150 lignes remplace les N
   requetes HTTP par un `llama_decode` unique (etat en seq 0, copie vers 1..N, tokens des branches avec leur
   seq_id, logits lus aux derniers tokens). Gain attendu : x2-3 sur la latence System One en mode mono.
2. ~~Etiquettes bi-lettres (255 options)~~ fait : lecture sur les noms d'options avec resolution des prefixes (`label_mode`).
3. **Niveaux de score isoles** (chaque niveau juge seul, comme decider : +1 a +3 points, meilleure calibration).
4. **Clone VL** dans `engine_torch.py` (images dans l'etat, M-RoPE) : reflex fournit la reference.
5. **Benchmarks supplementaires** : banking77 / CLINC (routage), HelpSteer2 (score), un jeu de moderation, pour
   une comparaison plus large que 60 cas.

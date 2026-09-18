# LLM-and-Classifier : fusion Jev-clone (System One) x Bonsai 2 27B (System Two) sur RTX 4060

Un **classificateur/decideur** ultra-rapide (clone ouvert de [Jev](docs/01-JEV-typesafe-analyse.md), le
"System One model" de TypeSafe AI) fusionne avec un **LLM** de 27 milliards de parametres qui tient dans
6 Go ([Bonsai 2 27B](docs/02-BONSAI-2-27B-analyse.md), PrismML, base Qwen3.8-27B), le tout sur **une RTX
4060 8 Go** ou moins (profils CPU seul et GPU 4 Go inclus).

**Lire d'abord : [docs/00-PROTOCOLE-FUSION-JEV-BONSAI.md](docs/00-PROTOCOLE-FUSION-JEV-BONSAI.md)** (le
protocole complet, en 9 phases, avec criteres d'acceptation, risques et commandes).

## Idee en trois lignes
1. Le clone de Jev lit, en une passe et sans generer de texte, une **distribution calibree** sur les options
   de chaque question typee (`choice` / `score` / `noul`) : ~50-150 ms.
2. Si la confiance depasse un seuil calibre, le code agit ; sinon **Bonsai 2 27B** raisonne (budget de
   reflexion selon le risque), genere, appelle des outils, lit des images.
3. Tout est journalise ; Bonsai etiquette les cas douteux ; le clone est recalibre / re-entraine.

## Demarrage rapide (RTX 4060)
```bash
PROFILE=scripts/profiles/rtx4060-8gb-qualite.env ./scripts/setup.sh     # binaires PrismML + poids + venv
./scripts/start_bonsai.sh        # terminal 1 : Bonsai 2 27B PTQ1_0 sur :8080
./scripts/start_jev_clone.sh     # terminal 2 : Ternary-Bonsai-1.7B (clone niveau 0) sur :8081
source .venv/bin/activate
jev decide --server http://127.0.0.1:8081 --state "Charged twice for order A-104, please refund." \
    --choice team=billing,technical,sales --noul "refund=Is a refund requested?"
export JEV_S1_URL=http://127.0.0.1:8081 JEV_S2_URL=http://127.0.0.1:8080
jev serve --port 8008            # POST /v1/systemone (format TypeSafe) et POST /v1/fusion/decide
```
Sans GPU : `PROFILE=scripts/profiles/minimal-cpu.env` (Bonsai-8B 1-bit + clone 1.7B, 8 Go de RAM).
Avec une **A100 80 Go sur Colab** : ouvrir `colab/jev_bonsai_a100.ipynb` pour distiller Bonsai et entrainer
le clone en pleine precision, puis rapatrier le GGUF et la calibration sur la 4060 (voir `docs/05-COLAB-A100.md`).

## Contenu du depot
| Chemin | Contenu |
|---|---|
| `docs/00-PROTOCOLE-FUSION-JEV-BONSAI.md` | **le protocole complet** |
| `docs/01-JEV-typesafe-analyse.md` | comment Jev fonctionne (contrat, mecanisme deduit, RLCD, benchmarks, clones ouverts) |
| `docs/02-BONSAI-2-27B-analyse.md` | Bonsai 2 27B : architecture, formats, qualite, vitesse, memoire, runtime |
| `docs/03-MATERIEL-profils.md` | RTX 4060 et exigences minimales, matrice de profils, feuille VRAM, depannage OOM |
| `docs/04-REFERENCES.md` | sources |
| `jev_clone/` | le clone : `schema` (contrat), `prompt` (prefixe + branches), `backend_llamacpp` (lecture par grammaire), `readout` (temperature, confiance), `engine`, `calibrate`, `distill`, `fusion` (routeur), `server`, `cli` |
| `scripts/` | `setup.sh`, `start_bonsai.sh`, `start_jev_clone.sh`, `bench.sh`, `profiles/*.env`, `windows/*.ps1` |
| `training/` | `train_lora_rlcd.py` (LoRA / QLoRA / `--full` + regle de score propre + distillation), `make_public_mix.py` (jeux publics -> JSONL) et sa notice |
| `colab/` | notebook A100 80 Go : installation, Bonsai enseignant, distillation, entrainement complet, export GGUF, calibration, copie vers Drive |
| `docs/05-COLAB-A100.md` | usine A100 / production 4060 : quoi faire ou, durees, artefacts |
| `examples/` | routage de tickets avec escalade ; boucle d'agent temps reel |
| `tests/` | 23 tests (hors ligne + integration contre un `llama-server` reel) |

## Ce qui est verifie / ce qui ne l'est pas
* Verifie ici : le paquet `jev_clone` (tests unitaires), la lecture par grammaire et le cache de prefixe
  contre un vrai `llama-server` du fork PrismML (`prism-b10683`, build CPU) sur un modele factice, les
  scripts shell (syntaxe), les chiffres tires des livres blancs et depots PrismML.
* Non verifie ici (pas de GPU dans l'environnement de redaction) : les debits RTX 4060 (estimations), le
  telechargement des poids Hugging Face (motifs de fichiers repris du depot officiel `Bonsai-demo`),
  le script d'entrainement.

Licence : Apache 2.0 (ce depot). Bonsai et Qwen : Apache 2.0. Non affilie a TypeSafe AI ni a PrismML.

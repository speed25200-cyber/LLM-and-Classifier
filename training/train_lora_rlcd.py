"""Entrainement "RLCD-lite" d'un clone Jev sur un petit modele Qwen3.5 (0.8B / 2B / 4B / 9B).
RTX 4060 8 Go : LoRA bf16 pour 0.8B, QLoRA 4-bit pour 2B/4B.  A100 80 Go : --full (fine-tuning complet
bf16, recette decider) sur 2B ou 4B, batch 16-32, sequences 2048.

Principe : la sortie du modele EST la distribution sur les etiquettes (A/B/C..., Yes/No) lue au dernier
token de chaque branche. Une regle de score propre (NLL = log-score, ou Brier) est alors une recompense
RL dont l'esperance est differentiable en forme close : l'objectif "RLCD" de TypeSafe se reduit a une
minimisation supervisee. On ajoute optionnellement une distillation (KL) vers les distributions
"enseignant" produites par Bonsai 2 27B (jev_clone/distill.py, cle `teacher_probs`).

Donnees : JSONL {"state": ..., "questions": {...}, "labels": {qid: ...}, "teacher_probs": {qid: {...}}?}
    python training/train_lora_rlcd.py --model Qwen/Qwen3.5-0.8B-Base --data data/train.jsonl \
        --val data/val.jsonl --out runs/jev-0.8b --epochs 1 --loss nll --kl 0.5 --permutations 2
Export GGUF ensuite (voir training/README.md).

NOTE : ce script n'a pas pu etre execute dans l'environnement de redaction (pas de GPU / torch) ;
il suit la recette de reflex/decider et doit etre valide sur la machine cible (voir README).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from jev_clone.calibrate import fit_temperature, report
from jev_clone.prompt import PromptFormat, build_branches
from jev_clone.readout import Calibration, softmax
from jev_clone.schema import SystemOneRequest


# ---------------------------------------------------------------------------------------------
# donnees : un exemple -> (texte complet, ids des tokens etiquettes, index de la bonne etiquette, cible enseignant)
# ---------------------------------------------------------------------------------------------
def gold_index(kind, q, label, keys) -> int:
    if kind == "noul":
        return keys.index(bool(label))
    if kind == "choice":
        return keys.index(label)
    return keys.index(int(label))


def teacher_vector(kind, q, tp, keys) -> np.ndarray | None:
    if not tp:
        return None
    if kind == "noul":
        v = [float(tp.get("true", 0.5)), float(tp.get("false", 0.5))]
        v = [v[0] if k else v[1] for k in keys]
    elif kind == "choice":
        v = [float(tp.get(str(k), 0.0)) for k in keys]
    else:
        v = [float(tp.get(str(k), 0.0)) for k in keys]
    v = np.array(v, dtype=np.float64)
    return v / v.sum() if v.sum() > 0 else None


def load_examples(path, fmt: PromptFormat, permutations=1, seed=0, abstain_aug=0.0):
    rng = random.Random(seed)
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            req = SystemOneRequest(state=ex["state"], questions=ex["questions"])
            prefix = fmt.prefix(req.state)
            for qid, q in req.questions.items():
                if qid not in ex.get("labels", {}):
                    continue
                for br in build_branches(qid, q, fmt, permutations, rng):
                    gi = gold_index(q.type, q, ex["labels"][qid], br.keys)
                    tv = teacher_vector(q.type, q, (ex.get("teacher_probs") or {}).get(qid), br.keys)
                    rows.append({"text": prefix + br.text, "labels": br.labels, "gold": gi, "teacher": tv, "kind": q.type})
    return rows


def label_token_ids(tok, labels: list[str]) -> list[int]:
    ids = []
    for lab in labels:
        t = tok.encode(lab, add_special_tokens=False)
        if len(t) != 1:
            # "Yes"/"No" sont un seul token chez Qwen ; a defaut on prend le premier token (prefixe)
            t = t[:1]
        ids.append(t[0])
    return ids


def collate(tok, rows, max_len):
    texts = [r["text"] for r in rows]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len, padding_side="right")
    last = enc["attention_mask"].sum(1) - 1
    return enc, last


def scoring_loss(logits, gold: int, kind: str):
    if kind == "nll":
        return -F.log_softmax(logits, -1)[gold]
    p = F.softmax(logits, -1)
    onehot = torch.zeros_like(p); onehot[gold] = 1
    return ((p - onehot) ** 2).sum()


# ---------------------------------------------------------------------------------------------
def evaluate(model, tok, rows, device, max_len, bs=8):
    model.eval()
    logits_all, golds = [], []
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            chunk = rows[i:i + bs]
            enc, last = collate(tok, chunk, max_len)
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(**enc).logits
            for j, r in enumerate(chunk):
                ids = label_token_ids(tok, r["labels"])
                z = out[j, last[j], ids].float().cpu().numpy()
                logits_all.append(z); golds.append(r["gold"])
    K = max(len(z) for z in logits_all)
    L = np.full((len(logits_all), K), -1e9)
    for i, z in enumerate(logits_all):
        L[i, :len(z)] = z
    return L, np.array(golds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
    ap.add_argument("--data", required=True)
    ap.add_argument("--val")
    ap.add_argument("--out", default="runs/jev-lora")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--loss", choices=["nll", "brier"], default="nll")
    ap.add_argument("--kl", type=float, default=0.0, help="poids de la distillation vers teacher_probs (0 = off)")
    ap.add_argument("--permutations", type=int, default=2)
    ap.add_argument("--qlora", action="store_true", help="base en 4-bit (bitsandbytes) : obligatoire pour 2B/4B sur 8 Go")
    ap.add_argument("--full", action="store_true", help="fine-tuning complet (pas de LoRA) : A100 80 Go, 2B/4B")
    ap.add_argument("--sft-data", help="JSONL {prompt, response} (reponses distillees de Bonsai) : le meme modele apprend aussi a generer")
    ap.add_argument("--sft-ratio", type=float, default=0.3, help="part des pas d'optimisation consacres a la generation")
    ap.add_argument("--max-train", type=int, default=0, help="plafonner le nombre de branches d'entrainement (0 = tout)")
    ap.add_argument("--save-every", type=int, default=0, help="sauvegarder un point de reprise tous les N pas (0 = fin seulement)")
    ap.add_argument("--chat", action="store_true", default=True, help="format ChatML + <think></think> (identique au serveur)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {"dtype": torch.bfloat16}
    if args.qlora:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    if args.full and args.qlora:
        raise SystemExit("--full et --qlora sont exclusifs")
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map={"": 0}, **kw)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    if args.full:
        for p in model.parameters():
            p.requires_grad_(True)
        print(f"fine-tuning complet : {sum(p.numel() for p in model.parameters())/1e9:.2f} G parametres")
    else:
        lcfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0, task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                                          "in_proj_qkvz", "in_proj_ba", "out_proj"])  # attention + GatedDeltaNet (Qwen3.5)
        model = get_peft_model(model, lcfg)
        model.print_trainable_parameters()

    fmt = PromptFormat(chat=args.chat, no_think=True)
    sft_rows = []
    if args.sft_data:
        with open(args.sft_data) as f:
            for line in f:
                if line.strip():
                    ex = json.loads(line)
                    sft_rows.append((f"<|im_start|>user\n{ex['prompt']}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                                     ex["response"] + "<|im_end|>"))
        print(f"{len(sft_rows)} exemples de generation (SFT)")
    train_rows = load_examples(args.data, fmt, args.permutations, args.seed)
    if args.max_train:
        random.Random(args.seed).shuffle(train_rows); train_rows = train_rows[:args.max_train]
    val_rows = load_examples(args.val, fmt) if args.val else []
    print(f"{len(train_rows)} branches d'entrainement, {len(val_rows)} de validation")

    if val_rows:
        L, y = evaluate(model, tok, val_rows, device, args.max_len)
        print("== avant ==\n", report(np.stack([softmax(l) for l in L]), y))

    params = [p for p in model.parameters() if p.requires_grad]
    if not args.full:
        for p in params:
            p.data = p.data.float()   # adaptateurs LoRA en fp32 ; en --full les poids restent en bf16 (AdamW fp32 states)
    opt = torch.optim.AdamW(params, lr=args.lr if not args.full else min(args.lr, 2e-5), weight_decay=0.0)
    steps_total = math.ceil(len(train_rows) / args.bs / args.accum) * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * max(0.0, 1 - s / max(1, steps_total)))

    rng = random.Random(args.seed)
    step, t0 = 0, time.perf_counter()

    def sft_loss(batch_rows):
        """Perte causale standard sur les tokens de la reponse seulement (prompt masque)."""
        texts = [p + r for p, r in batch_rows]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_len, padding_side="right")
        labels = enc["input_ids"].clone()
        for j, (p, _) in enumerate(batch_rows):
            n_prompt = len(tok(p, add_special_tokens=False)["input_ids"])
            labels[j, :n_prompt] = -100
        labels[enc["attention_mask"] == 0] = -100
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model(**enc, labels=labels.to(device)).loss

    for epoch in range(args.epochs):
        rng.shuffle(train_rows)
        model.train()
        for micro, i in enumerate(range(0, len(train_rows), args.bs)):
            if sft_rows and rng.random() < args.sft_ratio:
                (sft_loss(rng.sample(sft_rows, min(args.bs, len(sft_rows)))) / args.accum).backward()
                if (micro + 1) % args.accum == 0:
                    torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
                continue
            chunk = train_rows[i:i + args.bs]
            enc, last = collate(tok, chunk, args.max_len)
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(**enc).logits
            losses = []
            for j, r in enumerate(chunk):
                ids = torch.tensor(label_token_ids(tok, r["labels"]), device=device)
                z = out[j, last[j], ids].float()
                loss = scoring_loss(z, r["gold"], args.loss)
                if args.kl > 0 and r["teacher"] is not None:
                    t = torch.tensor(r["teacher"], device=device, dtype=torch.float32)
                    loss = loss + args.kl * F.kl_div(F.log_softmax(z, -1), t, reduction="sum")
                losses.append(loss)
            (torch.stack(losses).mean() / args.accum).backward()
            if (micro + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % 20 == 0:
                    print(f"epoch {epoch} step {step}/{steps_total} loss {losses[-1].item():.4f} ({time.perf_counter()-t0:.0f}s)")
                if args.save_every and step % args.save_every == 0:
                    ck = Path(args.out) / f"step-{step}"; ck.mkdir(parents=True, exist_ok=True); model.save_pretrained(ck)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.full:
        model.save_pretrained(out / "merged"); tok.save_pretrained(out / "merged")
        print("modele complet sauvegarde dans", out / "merged")
    else:
        model.save_pretrained(out)
        print("adaptateur LoRA sauvegarde dans", out)
    if val_rows:
        L, y = evaluate(model, tok, val_rows, device, args.max_len)
        print("== apres ==\n", report(np.stack([softmax(l) for l in L]), y))
        T = fit_temperature(L, y)
        print(f"== apres + temperature T={T:.3f} ==\n", report(np.stack([softmax(l, T) for l in L]), y))
        Calibration(temperature={"noul": T, "choice": T, "score": T}).save(out / "calibration.json")
    if not args.qlora and not args.full:
        merged = model.merge_and_unload()
        merged.save_pretrained(out / "merged"); tok.save_pretrained(out / "merged")
        print("modele fusionne (pour conversion GGUF) dans", out / "merged")


if __name__ == "__main__":
    main()

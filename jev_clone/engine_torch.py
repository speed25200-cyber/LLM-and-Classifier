"""Moteur System One en memoire (PyTorch / transformers) : UNE passe pour toutes les questions.

C'est la reconstruction du "parallel sampler" de Jev sur un modele ouvert, adaptee du moteur de
`reflex` (kshetrajna12, MIT) : etat encode une fois (cache), branches isolees, lecture directe.
  * `packed`  (modeles a attention pure : Qwen3, Qwen3-VL) : toutes les branches dans UNE sequence,
              masque 4D par blocs (une branche voit l'etat + elle-meme), positions qui repartent a len(etat).
  * `batched` (hybrides GatedDeltaNet : Qwen3.5 / 3.8, Bonsai) : un masque ne peut pas isoler des branches
              dans une couche recurrente, donc les branches forment un batch a droite sur une copie du cache.

Pourquoi en plus du backend llama.cpp : (1) latence, une passe GPU au lieu de N requetes HTTP (cible < 50 ms
pour 10 questions sur un 0,8B, contre 70-500 ms pour l'API Jev) ; (2) exactement le meme calcul que
l'entrainement (`training/`) ; (3) images dans l'etat avec un modele VL (a venir). Sur la RTX 4060 il sert
le clone entraine en bf16 (0,8B : 1,6 Go) ; Bonsai reste sur llama.cpp.

Non execute dans l'environnement de redaction (pas de torch/GPU) : valider avec `tests` sur la machine cible.
"""

from __future__ import annotations

import copy
import hashlib
import random
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np

from jev_clone.prompt import Branch, PromptFormat, build_branches
from jev_clone.readout import Calibration, merge_branches, to_answer
from jev_clone.schema import SystemOneRequest, SystemOneResponse, Usage

CACHED = -2


@dataclass
class Segment:
    ids: list[int]
    parent: int = -1


def build_pack(segments, past_len: int = 0, past_seg: int = -1, device="cpu"):
    """Concatene les segments et construit le masque [1,1,T,P+T] : une cle j est visible de la requete i ssi
    meme segment et j <= i (causal), ou j appartient au segment parent (l'etat entier)."""
    import torch
    ids, seg_of, pos, starts, lens = [], [], [], [], []
    seg_lengths = {past_seg: past_len} if past_len else {}
    for s_idx, seg in enumerate(segments):
        base = seg_lengths[seg.parent] if seg.parent != -1 else 0
        starts.append(len(ids)); lens.append(len(seg.ids))
        ids.extend(seg.ids); seg_of.extend([s_idx] * len(seg.ids)); pos.extend(range(base, base + len(seg.ids)))
        seg_lengths[s_idx] = len(seg.ids)
    T = len(ids)
    q_seg = torch.tensor(seg_of, device=device)
    parent = torch.tensor([s.parent for s in segments], device=device)
    q_parent = parent[q_seg]
    k_seg = torch.cat([torch.full((past_len,), past_seg, device=device), q_seg])
    q_idx = torch.arange(T, device=device)
    k_idx = torch.arange(past_len + T, device=device) - past_len
    same = (k_seg[None, :] == q_seg[:, None]) & (k_idx[None, :] <= q_idx[:, None])
    par = k_seg[None, :] == q_parent[:, None]
    last = torch.tensor([s + n - 1 for s, n in zip(starts, lens)], device=device)
    return {"input_ids": torch.tensor([ids], device=device), "position_ids": torch.tensor([pos], device=device),
            "attention_mask": (same | par)[None, None], "last_index": last}


def right_pad(seqs, pad_id: int, device):
    import torch
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s); mask[i, : len(s)] = 1
    return ids.to(device), mask.to(device), torch.tensor([len(s) - 1 for s in seqs]).to(device)


@dataclass
class StateEntry:
    key: str
    ids: list[int]
    cache: Any
    created: float


class TorchSystemOneEngine:
    def __init__(self, model, tokenizer, fmt: PromptFormat | None = None, calibration: Calibration | None = None,
                 max_pack_tokens: int = 8192, state_cache_entries: int = 8, model_name: str = "jev-clone-torch",
                 strategy: str | None = None):
        import torch
        self.torch = torch
        self.model, self.tok = model, tokenizer
        self.fmt = fmt or PromptFormat()
        self.cal = calibration or Calibration()
        self.max_pack_tokens = max_pack_tokens
        self.model_name = model_name
        self.device = next(model.parameters()).device
        self.strategy = strategy or ("batched" if self.is_hybrid else "packed")
        if self.strategy == "packed" and self.is_hybrid:
            raise ValueError("packed ne peut pas isoler des branches dans des couches recurrentes")
        self._states: OrderedDict[str, StateEntry] = OrderedDict()
        self._n_states = state_cache_entries
        self._lock = threading.Lock()
        self._label_ids: dict[str, int] = {}
        self.model.eval()

    @classmethod
    def load(cls, model_id: str, dtype=None, device: str = "cuda", adapter_path: str | None = None,
             calibration_path: str | None = None, **kw) -> "TorchSystemOneEngine":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype or torch.bfloat16, device_map=device, attn_implementation="sdpa")
        if adapter_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path).merge_and_unload()
        template = tok.chat_template or ""
        fmt = PromptFormat(chat=bool(template), no_think="enable_thinking" in template)
        return cls(model, tok, fmt, Calibration.load(calibration_path), model_name=model_id, **kw)

    # ---- plomberie ---------------------------------------------------------------------------
    @property
    def text_config(self):
        cfg = self.model.config
        return getattr(cfg, "text_config", None) or cfg

    @property
    def is_hybrid(self) -> bool:
        return any(t != "full_attention" for t in getattr(self.text_config, "layer_types", []) or [])

    @property
    def pad_id(self) -> int:
        return self.tok.pad_token_id if self.tok.pad_token_id is not None else 0

    def _causal_lm(self):
        m = self.model
        return m.get_base_model() if hasattr(m, "get_base_model") else m

    def _new_cache(self):
        from transformers import DynamicCache
        return DynamicCache(config=self.text_config)

    def label_id(self, label: str) -> int:
        if label not in self._label_ids:
            ids = self.tok.encode(label, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"etiquette {label!r} n'est pas un token unique : {ids}")
            self._label_ids[label] = ids[0]
        return self._label_ids[label]

    def _encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def restrict(self, row, br: Branch):
        lab = self.torch.tensor([self.label_id(lb) for lb in br.labels], device=row.device)
        return row[lab]

    # ---- cache d'etat --------------------------------------------------------------------------
    def encode_state(self, state) -> tuple[StateEntry, bool]:
        text = self.fmt.prefix(state)
        key = hashlib.sha256(text.encode()).hexdigest()
        if key in self._states:
            self._states.move_to_end(key)
            return self._states[key], True
        ids = self._encode(text)
        cache = self._new_cache()
        with self.torch.inference_mode():
            self.model(input_ids=self.torch.tensor([ids], device=self.device), past_key_values=cache, use_cache=True, logits_to_keep=1)
        entry = StateEntry(key=key, ids=ids, cache=cache, created=time.time())
        self._states[key] = entry
        while len(self._states) > self._n_states:
            self._states.popitem(last=False)
        return entry, False

    # ---- passe unique --------------------------------------------------------------------------
    def _branches_packed(self, entry: StateEntry, chunk: list[list[int]]):
        P = len(entry.ids)
        pack = build_pack([Segment(b, parent=CACHED) for b in chunk], past_len=P, past_seg=CACHED, device=self.device)
        with self.torch.inference_mode():
            logits = self.model(input_ids=pack["input_ids"], position_ids=pack["position_ids"], attention_mask=pack["attention_mask"],
                                past_key_values=entry.cache, use_cache=True, logits_to_keep=pack["last_index"]).logits[0]
            entry.cache.crop(-pack["input_ids"].shape[1])  # on retire les tokens des branches, l'etat reste
        return logits.float()

    def _branches_batched(self, entry: StateEntry, chunk: list[list[int]]):
        torch = self.torch
        P, B = len(entry.ids), len(chunk)
        ids, mask, last = right_pad(chunk, self.pad_id, self.device)
        full_mask = torch.cat([torch.ones((B, P), dtype=mask.dtype, device=self.device), mask], 1)
        cache = copy.deepcopy(entry.cache)
        cache.reorder_cache(torch.zeros(B, dtype=torch.long, device=self.device))  # duplique l'etat B fois (KV + etats recurrents)
        pos = (torch.arange(ids.shape[1], device=self.device) + P).expand(B, ids.shape[1])
        with torch.inference_mode():
            hidden = self._causal_lm().model(input_ids=ids, attention_mask=full_mask, position_ids=pos,
                                             past_key_values=cache, use_cache=True).last_hidden_state
            h = hidden[torch.arange(B, device=self.device), last]
            return self._causal_lm().lm_head(h).float()

    def _chunks(self, branch_ids):
        cur, n, longest = [], 0, 0
        for b in branch_ids:
            new_longest = max(longest, len(b))
            size = n + len(b) if self.strategy == "packed" else (len(cur) + 1) * new_longest
            if cur and size > self.max_pack_tokens:
                yield cur; cur, n, longest = [], 0, 0; new_longest = len(b)
            cur.append(b); n += len(b); longest = new_longest
        if cur:
            yield cur

    def answer(self, req: SystemOneRequest | dict) -> SystemOneResponse:
        if isinstance(req, dict):
            req = SystemOneRequest.model_validate(req)
        with self._lock:
            t0 = time.perf_counter()
            rng = random.Random(0)
            branches: list[Branch] = []
            for qid, q in req.questions.items():
                branches.extend(build_branches(qid, q, self.fmt, req.permutations, rng))
            branch_ids = [self._encode(b.text) for b in branches]
            entry, hit = self.encode_state(req.state)
            rows = []
            for chunk in self._chunks(branch_ids):
                rows.append(self._branches_packed(entry, chunk) if self.strategy == "packed" else self._branches_batched(entry, chunk))
            logits = self.torch.cat(rows, 0)
            per_q: dict[str, list[tuple[Branch, np.ndarray]]] = {}
            for b, row in zip(branches, logits):
                per_q.setdefault(b.qid, []).append((b, self.restrict(row, b).cpu().numpy()))
            answers = {qid: to_answer(q.type, merge_branches(q.type, per_q[qid], self.cal), q) for qid, q in req.questions.items()}
            q_tokens = sum(len(b) for b in branch_ids)
            return SystemOneResponse(answers=answers, model=self.model_name,
                                     usage=Usage(input_tokens=len(entry.ids) + q_tokens, state_tokens=len(entry.ids),
                                                 question_tokens=q_tokens, branches=len(branches), state_cache_hit=hit),
                                     latency_ms=round((time.perf_counter() - t0) * 1000, 2))

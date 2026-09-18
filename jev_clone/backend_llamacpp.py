"""Backend llama.cpp : lecture des probabilites d'etiquettes via l'API HTTP de llama-server.

Pour chaque branche on envoie UNE requete /completion avec :
  * n_predict = 1                     (aucune generation utile : on ne lit que la distribution)
  * grammar = root ::= "A" | "B" ...  (le sampler masque tout token hors etiquettes)
  * post_sampling_probs = true        (les probabilites renvoyees sont renormalisees sur les etiquettes)
  * samplers = [], temperature = 1.0  (aucun top-k / top-p / min-p : distribution brute du modele)
  * cache_prompt = true               (le prefixe d'etat partage est reutilise entre branches)

Fonctionne avec le fork PrismML (Bonsai 2 : PTQ1_0 / PQ2_0) comme avec llama.cpp mainline
(Bonsai 1-bit Q1_0, Ternary Q2_0_g64, Qwen3.5 GGUF...). Verifie contre un llama-server reel
(build prism-b10683) : `completion_probabilities[0].top_probs = [{id, token, prob}, ...]`.
"""

from __future__ import annotations

import concurrent.futures as cf
import time
from dataclasses import dataclass

import numpy as np
import requests

from jev_clone.prompt import Branch, label_grammar
from jev_clone.readout import probs_to_logits


@dataclass
class BranchResult:
    logits: np.ndarray        # log-probabilites restreintes, dans l'ordre de branch.labels
    prompt_tokens: int = 0
    cached_tokens: int = 0
    ms: float = 0.0


def aggregate_label_probs(top: list[dict], labels: list[str]) -> np.ndarray:
    """Somme, pour chaque etiquette, la probabilite des tokens qui en sont un prefixe.

    Avec la grammaire, le sampler n'autorise que des tokens qui sont un prefixe valide d'une etiquette :
    "Yes" peut donc arriver sous la forme des tokens "Y", "Ye" ou "Yes" selon le tokenizer / le modele.
    On les additionne (les etiquettes ne partagent jamais de prefixe : A..Z, Yes/No)."""
    acc = np.zeros(len(labels), dtype=np.float64)
    for e in top:
        tok = e.get("token", "")
        if not tok:
            continue
        pr = float(e["prob"]) if "prob" in e else float(np.exp(e.get("logprob", -1e9)))
        hits = [i for i, lab in enumerate(labels) if lab.startswith(tok)]
        if len(hits) == 1:
            acc[hits[0]] += pr
    if acc.sum() <= 0:
        # la grammaire garantit normalement les etiquettes ; on retombe sur l'uniforme
        acc = np.full(len(labels), 1.0 / len(labels))
    return acc / acc.sum()


class LlamaCppBackend:
    def __init__(self, base_url: str = "http://127.0.0.1:8081", timeout: float = 120.0,
                 max_workers: int = 4, id_slot: int = -1, lora: list[dict] | None = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_workers = max_workers
        self.id_slot = id_slot
        # echelle des adaptateurs LoRA charges par le serveur, par requete : [{"id": 0, "scale": 1.0}]
        # (None = reglage du serveur). Permet, en mode mono, de lire les probabilites System One sur le modele
        # publie (scale 0) tout en generant avec l'adaptateur (scale 1) sur le meme serveur.
        self.lora = lora
        self._session = requests.Session()

    def _with_lora(self, payload: dict) -> dict:
        if self.lora is not None:
            payload = {**payload, "lora": self.lora}
        return payload

    # ---- infos serveur -------------------------------------------------------------------
    def props(self) -> dict:
        r = self._session.get(f"{self.base_url}/props", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def model_name(self) -> str:
        try:
            p = self.props()
            return str(p.get("model_path") or p.get("model_alias") or "")
        except Exception:
            return ""

    def health(self) -> bool:
        try:
            r = self._session.get(f"{self.base_url}/health", timeout=5)
            return r.status_code == 200 and r.json().get("status") == "ok"
        except Exception:
            return False

    def tokenize(self, text: str) -> list[int]:
        r = self._session.post(f"{self.base_url}/tokenize",
                               json={"content": text, "add_special": False, "parse_special": True},
                               timeout=self.timeout)
        r.raise_for_status()
        toks = r.json()["tokens"]
        return [t["id"] if isinstance(t, dict) else t for t in toks]

    # ---- lecture d'une branche --------------------------------------------------------------
    def score_branch(self, prefix: str, branch: Branch) -> BranchResult:
        payload = {
            "prompt": prefix + branch.text,
            "n_predict": 1,
            "n_probs": max(16, 4 * len(branch.labels)),
            "post_sampling_probs": True,
            "samplers": [],
            "temperature": 1.0,
            "grammar": label_grammar(branch.labels),
            "cache_prompt": True,
            "id_slot": self.id_slot,
        }
        payload = self._with_lora(payload)
        t0 = time.perf_counter()
        r = self._session.post(f"{self.base_url}/completion", json=payload, timeout=self.timeout)
        r.raise_for_status()
        d = r.json()
        ms = (time.perf_counter() - t0) * 1000.0
        cp = d.get("completion_probabilities") or []
        if not cp:
            raise RuntimeError("llama-server n'a pas renvoye completion_probabilities (n_probs ignore ?)")
        top = cp[0].get("top_probs") or cp[0].get("top_logprobs") or []
        probs = aggregate_label_probs(top, branch.labels)
        tm = d.get("timings", {}) or {}
        return BranchResult(logits=probs_to_logits(probs),
                            prompt_tokens=int(tm.get("prompt_n", 0) or 0),
                            cached_tokens=int(tm.get("cache_n", 0) or 0), ms=ms)

    def score_branches(self, prefix: str, branches: list[Branch]) -> list[BranchResult]:
        """La 1re branche est envoyee seule (elle remplit le cache du prefixe), les autres en parallele
        sur les slots du serveur (-np N) : chacune ne traite alors que sa propre branche."""
        if not branches:
            return []
        first = self.score_branch(prefix, branches[0])
        rest = branches[1:]
        if not rest:
            return [first]
        if self.max_workers <= 1:
            return [first] + [self.score_branch(prefix, b) for b in rest]
        with cf.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            out = list(ex.map(lambda b: self.score_branch(prefix, b), rest))
        return [first] + out

    # ---- acces bas niveau : gabarit de chat et completion brute par morceaux ----------------------
    def apply_template(self, messages: list[dict], **kw) -> str:
        """Rend les messages avec le gabarit de chat du modele (llama-server /apply-template)."""
        r = self._session.post(f"{self.base_url}/apply-template", json={"messages": messages, **kw}, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["prompt"]

    def complete(self, prompt: str, n_predict: int = 256, stop: list[str] | None = None, temperature: float = 0.7,
                 top_p: float = 0.95, top_k: int = 20, extra: dict | None = None) -> dict:
        """Completion brute (texte -> texte) avec cache de prefixe. Renvoie {content, stop_type, tokens, timings}.
        stop_type : "limit" (n_predict atteint), "word" (un mot d'arret vu), "eos"."""
        payload = {"prompt": prompt, "n_predict": n_predict, "cache_prompt": True, "temperature": temperature,
                   "top_p": top_p, "top_k": top_k, "stop": stop or [], "id_slot": self.id_slot}
        if extra:
            payload.update(extra)
        payload = self._with_lora(payload)
        r = self._session.post(f"{self.base_url}/completion", json=payload, timeout=self.timeout)
        r.raise_for_status()
        d = r.json()
        return {"content": d.get("content", ""), "stop_type": d.get("stop_type", ""), "stopping_word": d.get("stopping_word", ""),
                "tokens": int(d.get("tokens_predicted", 0) or 0), "timings": d.get("timings", {}) or {}}

    # ---- System Two : generation classique (pour la fusion) ---------------------------------
    def chat(self, messages: list[dict], max_tokens: int = 512, thinking_budget: int | None = None,
             temperature: float = 0.7, tools: list | None = None, extra: dict | None = None) -> dict:
        payload: dict = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        if tools:
            payload["tools"] = tools
        if thinking_budget is not None:
            # llama-server (fork PrismML / mainline recent) : 0 = pas de reflexion, N = plafond, -1 = illimite
            payload["thinking_budget_tokens"] = thinking_budget
            if thinking_budget == 0:
                payload["chat_template_kwargs"] = {"enable_thinking": False}
        if extra:
            payload.update(extra)
        payload = self._with_lora(payload)
        r = self._session.post(f"{self.base_url}/v1/chat/completions", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

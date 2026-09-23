"""Backend llama.cpp : lecture des probabilites d'etiquettes via l'API HTTP de llama-server.

Pour chaque branche on envoie UNE requete /completion avec :
  * n_predict = 1                     (aucune generation utile : on ne lit que la distribution)
  * grammar = root ::= "A" | "B" ...  (le sampler masque tout token hors etiquettes)
  * post_sampling_probs = true        (les probabilites renvoyees sont renormalisees sur les etiquettes)
  * samplers = [], temperature = 1.0  (aucun top-k / top-p / min-p : distribution brute du modele)
  * cache_prompt = true               (le prefixe d'etat partage est reutilise entre branches)

Jamais plus de requetes en parallele que de slots du serveur (/props total_slots) : avec -np 1 (classifieur sur CPU),
les branches passent en sequence sur le meme slot et reutilisent l'etat deja lu. Une erreur du serveur remonte avec
son message (ex. depassement du contexte du slot) au lieu d'un simple "500 Server Error". Si le KV partage d'un serveur
lance avec -kvu est plein ("Context size has been exceeded"), les branches refusees repassent une a une.

Fonctionne avec le fork PrismML (Bonsai 2 : PTQ1_0 / PQ2_0) comme avec llama.cpp mainline
(Bonsai 1-bit Q1_0, Ternary Q2_0_g64, Qwen3.5 GGUF...). Verifie contre un llama-server reel
(build prism-b10683) : `completion_probabilities[0].top_probs = [{id, token, prob}, ...]`.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import logging
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import requests

from jev_clone.prompt import Branch, label_grammar
from jev_clone.readout import probs_to_logits

log = logging.getLogger(__name__)
KV_FULL = "context size has been exceeded"   # llama-server : plus de place dans le KV (partage avec -kvu), 500 a chaque requete en cours


def raise_for_status(r: requests.Response) -> None:
    """raise_for_status avec le message de llama-server : la vraie cause (contexte du slot depasse, grammaire...) remonte
    jusqu'a l'interface (puce s1_error, erreur de tour) et au journal."""
    if r.status_code < 400:
        return
    try:
        err = r.json().get("error")
        msg = (err.get("message") or err) if isinstance(err, dict) else (err or r.text)
    except Exception:
        msg = r.text
    msg = " ".join(str(msg).split())[:400]
    log.warning("llama-server %s sur %s : %s", r.status_code, r.url, msg)
    raise requests.HTTPError(f"llama-server {r.status_code} : {msg}", response=r)


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
        self._slots: int | None = None   # slots du serveur (-np), lus une fois sur /props

    def _with_lora(self, payload: dict) -> dict:
        if self.lora is not None:
            payload = {**payload, "lora": self.lora}
        return payload

    # ---- infos serveur -------------------------------------------------------------------
    def props(self) -> dict:
        r = self._session.get(f"{self.base_url}/props", timeout=self.timeout)
        raise_for_status(r)
        return r.json()

    def slots(self) -> int:
        """Nombre de slots du serveur (/props total_slots), lu une fois ; inconnu : max_workers (non memorise)."""
        if self._slots is None:
            try:
                r = self._session.get(f"{self.base_url}/props", timeout=5)
                r.raise_for_status()
                self._slots = max(1, int(r.json().get("total_slots") or self.max_workers))
            except Exception:
                return self.max_workers
        return self._slots

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
        raise_for_status(r)
        toks = r.json()["tokens"]
        return [t["id"] if isinstance(t, dict) else t for t in toks]

    # ---- lecture d'une branche --------------------------------------------------------------
    def _top_probs(self, prompt: str, grammar: str, n_probs: int) -> tuple[list[dict], dict]:
        """Une completion d'un token sous grammaire ; renvoie (top_probs renormalises, timings)."""
        payload = self._with_lora({"prompt": prompt, "n_predict": 1, "n_probs": n_probs, "post_sampling_probs": True,
                                   "samplers": [], "temperature": 1.0, "grammar": grammar, "cache_prompt": True,
                                   "id_slot": self.id_slot})
        r = self._session.post(f"{self.base_url}/completion", json=payload, timeout=self.timeout)
        raise_for_status(r)
        d = r.json()
        cp = d.get("completion_probabilities") or []
        if not cp:
            raise RuntimeError("llama-server n'a pas renvoye completion_probabilities (n_probs ignore ?)")
        return (cp[0].get("top_probs") or cp[0].get("top_logprobs") or []), (d.get("timings", {}) or {})

    def _resolve_values(self, prompt: str, remainders: dict[str, str], terminator: str, mass: float,
                        out: dict[str, float], depth: int, budget: list[int]) -> None:
        """Repartit `mass` entre les options `remainders` (nom -> texte restant a produire) en lisant la
        distribution du token suivant ; les tokens qui sont prefixe de plusieurs options declenchent une
        lecture de continuation (prompt + token). Termine par `terminator` pour separer une option qui est
        prefixe d'une autre ("MANUAL" / "MANUAL_REVIEW")."""
        if len(remainders) == 1:
            out[next(iter(remainders))] += mass
            return
        if depth > 8 or budget[0] <= 0:
            for k in remainders:  # limite atteinte : on repartit uniformement
                out[k] += mass / len(remainders)
            return
        budget[0] -= 1
        grammar = "root ::= " + " | ".join(json.dumps(rem + terminator) for rem in remainders.values())
        top, _ = self._top_probs(prompt, grammar, max(32, min(512, 4 * len(remainders))))
        groups: dict[str, list[str]] = {}
        pt: dict[str, float] = {}
        for e in top:
            tok = e.get("token", "")
            if not tok:
                continue
            pr = float(e["prob"]) if "prob" in e else float(np.exp(e.get("logprob", -1e9)))
            hit = [k for k, rem in remainders.items() if (rem + terminator).startswith(tok)]
            if hit:
                groups[tok] = hit; pt[tok] = pt.get(tok, 0.0) + pr
        total = sum(pt.values())
        if total <= 0:
            for k in remainders:
                out[k] += mass / len(remainders)
            return
        for tok, hit in groups.items():
            share = mass * pt[tok] / total
            if len(hit) == 1:
                out[hit[0]] += share
            else:
                sub = {k: remainders[k][len(tok):] for k in hit}
                self._resolve_values(prompt + tok, sub, terminator, share, out, depth + 1, budget)

    def score_branch(self, prefix: str, branch: Branch) -> BranchResult:
        if branch.terminator:
            t0 = time.perf_counter()
            out = {k: 0.0 for k in branch.labels}
            budget = [64]  # nombre maximal de lectures de continuation par branche
            self._resolve_values(prefix + branch.text, {k: k for k in branch.labels}, branch.terminator, 1.0, out, 0, budget)
            probs = np.array([out[k] for k in branch.labels], dtype=np.float64)
            probs = probs / probs.sum() if probs.sum() > 0 else np.full(len(branch.labels), 1.0 / len(branch.labels))
            return BranchResult(logits=probs_to_logits(probs), prompt_tokens=0, cached_tokens=0,
                                ms=(time.perf_counter() - t0) * 1000.0)
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
        raise_for_status(r)
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
        """La 1re branche est envoyee seule (elle remplit le cache du prefixe), les autres en parallele sur les slots
        du serveur (-np N), jamais plus que ses slots : avec un seul slot (classifieur sur CPU), elles passent en sequence
        et ne traitent que leur propre branche au lieu de re-preremplir l'etat sur d'autres slots."""
        if not branches:
            return []
        try:
            first = self.score_branch(prefix, branches[0])
        except requests.ConnectionError:
            self._slots = None   # serveur arrete ou relance (peut-etre avec un autre -np) : on relira /props
            raise
        rest = branches[1:]
        if not rest:
            return [first]
        workers = min(self.max_workers, self.slots()) if self.max_workers > 1 else 1
        if workers <= 1:
            return [first] + [self.score_branch(prefix, b) for b in rest]
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            out = list(ex.map(lambda b: self._branch_or_none(prefix, b), rest))
        # tampon KV partage plein (serveur lance avec -kvu, autres requetes sur le meme serveur) : les branches refusees
        # repassent une a une et relisent le prefixe deja en cache, au lieu de faire echouer toute la reponse
        return [first] + [r if r is not None else self.score_branch(prefix, b) for r, b in zip(out, rest)]

    def _branch_or_none(self, prefix: str, branch: Branch) -> BranchResult | None:
        try:
            return self.score_branch(prefix, branch)
        except requests.HTTPError as e:
            if KV_FULL in str(e).lower():
                return None
            raise

    # ---- acces bas niveau : gabarit de chat et completion brute par morceaux ----------------------
    def apply_template(self, messages: list[dict], **kw) -> str:
        """Rend les messages avec le gabarit de chat du modele (llama-server /apply-template)."""
        r = self._session.post(f"{self.base_url}/apply-template", json={"messages": messages, **kw}, timeout=self.timeout)
        raise_for_status(r)
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
        raise_for_status(r)
        d = r.json()
        return {"content": d.get("content", ""), "stop_type": d.get("stop_type", ""), "stopping_word": d.get("stopping_word", ""),
                "tokens": int(d.get("tokens_predicted", 0) or 0), "timings": d.get("timings", {}) or {}}

    # ---- System Two : generation classique (pour la fusion) ---------------------------------
    def chat(self, messages: list[dict], max_tokens: int = 512, thinking_budget: int | None = None,
             temperature: float = 0.7, tools: list | None = None, extra: dict | None = None,
             on_delta: Callable[[dict], None] | None = None, should_stop: Callable[[], bool] | None = None) -> dict:
        """/v1/chat/completions. Avec `on_delta`, la reponse est diffusee (SSE) : on_delta recoit
        {"type": "content"|"reasoning", "text"} et {"type": "tool_call", "index", "name", "args_delta"} au fil
        de l'eau ; le retour a la meme forme qu'une reponse non diffusee (message complet, usage, timings).
        `should_stop()` vrai -> la connexion est fermee (llama-server arrete la generation) et on rend ce qui existe."""
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
        if on_delta is not None:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        # Si la demande depasse le contexte du slot (prompt + max_tokens > n_ctx), llama-server repond 500 ;
        # on reduit max_tokens et on reessaie plutot que de faire echouer tout le tour de l'agent.
        for attempt in range(4):
            r = self._session.post(f"{self.base_url}/v1/chat/completions", json=payload, timeout=self.timeout,
                                   stream=on_delta is not None)
            if r.status_code < 400:
                if on_delta is None:
                    return r.json()
                return self._read_stream(r, on_delta, should_stop)
            msg = ""
            try:
                msg = json.dumps(r.json())
            except Exception:
                msg = r.text
            if r.status_code in (400, 500) and any(k in msg.lower() for k in ("context", "n_predict", "exceed", "n_ctx")) and payload["max_tokens"] > 64:
                payload["max_tokens"] = max(64, payload["max_tokens"] // 4)
                continue
            raise_for_status(r)
        raise_for_status(r)
        return r.json()

    @staticmethod
    def _read_stream(r, on_delta: Callable[[dict], None], should_stop: Callable[[], bool] | None) -> dict:
        content: list[str] = []
        reasoning: list[str] = []
        calls: dict[int, dict] = {}
        finish, usage, timings, stopped = None, {}, {}, False
        try:
            # llama-server diffuse en "chunked" : chunk_size=None rend chaque evenement SSE des son arrivee (512 par
            # defaut = rafales de tokens). Sans "chunked", None attendrait la fin du corps : lectures de 64 octets.
            chunked = "chunked" in r.headers.get("Transfer-Encoding", "").lower()
            for raw in r.iter_lines(chunk_size=None if chunked else 64, decode_unicode=True):
                if should_stop is not None and should_stop():
                    stopped = True
                    break
                if not raw or not raw.startswith("data:"):
                    continue
                data = raw[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if chunk.get("error"):
                    raise RuntimeError(str(chunk["error"])[:300])
                usage = chunk.get("usage") or usage
                timings = chunk.get("timings") or timings
                for ch in chunk.get("choices") or []:
                    finish = ch.get("finish_reason") or finish
                    d = ch.get("delta") or {}
                    if d.get("reasoning_content"):
                        reasoning.append(d["reasoning_content"]); on_delta({"type": "reasoning", "text": d["reasoning_content"]})
                    if d.get("content"):
                        content.append(d["content"]); on_delta({"type": "content", "text": d["content"]})
                    for tc in d.get("tool_calls") or []:
                        i = int(tc.get("index", len(calls)))
                        cur = calls.setdefault(i, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                        fn = tc.get("function") or {}
                        if tc.get("id"):
                            cur["id"] = tc["id"]
                        if fn.get("name"):
                            cur["function"]["name"] += fn["name"]
                        if fn.get("arguments"):
                            cur["function"]["arguments"] += fn["arguments"]
                        on_delta({"type": "tool_call", "index": i, "name": cur["function"]["name"], "args_delta": fn.get("arguments") or ""})
        finally:
            r.close()
        msg: dict = {"role": "assistant", "content": "".join(content)}
        if reasoning:
            msg["reasoning_content"] = "".join(reasoning)
        if calls:
            msg["tool_calls"] = [dict(calls[i], id=calls[i]["id"] or f"call_{i}") for i in sorted(calls)]
        return {"choices": [{"index": 0, "message": msg, "finish_reason": "cancelled" if stopped else finish}],
                "usage": usage, "timings": timings, "cancelled": stopped}

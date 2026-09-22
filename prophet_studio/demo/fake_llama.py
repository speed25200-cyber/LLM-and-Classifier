"""Faux llama-server (meme API HTTP) pour les tests et le mode demo de Prophet Studio. Aucune dependance.

  * /health, /props, /tokenize, /apply-template
  * /completion avec grammaire (lecture System One) : probabilites heuristiques deterministes tirees du
    texte de l'etat et de la question (aucun modele : c'est une maquette, pas une mesure)
  * /v1/chat/completions (diffuse ou non) : scenario d'agent scripte (reflexion, appels d'outils, reponse)
  * simulation d'OOM au demarrage : FAKE_LLAMA_OOM_ABOVE_CTX=8192 -> sortie en erreur si -c > 8192

    python -m prophet_studio.demo.fake_llama --port 7880 -c 16384 --tps 45
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ACTION_WORDS = ("cree", "crée", "create", "build", "ecris", "écris", "write", "fais", "make", "genere", "génère", "modifie", "change",
                "corrige", "fix", "ajoute", "add", "installe", "install", "lance", "run", "teste", "test", "refactor", "app", "site", "script")
RISKY = ("rm -rf", "supprime", "delete", "format", "drop table", "del /s", "shutdown", "mkfs", "curl", "| sh", "chmod 777")


def _h(s: str) -> float:
    return int(hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()[:8], 16) / 0xFFFFFFFF


def _norm(p):
    s = sum(p)
    return [x / s for x in p] if s > 0 else [1.0 / len(p)] * len(p)


def parse_s1_prompt(prompt: str) -> dict:
    state = prompt.split("# State\n", 1)[1].split("# Question", 1)[0] if "# State\n" in prompt else ""
    q = prompt.split("# Question\n", 1)[1] if "# Question\n" in prompt else prompt
    question = q.split("\n# Options", 1)[0].strip()
    opts = re.findall(r"^([A-Z])\. (.*)$", q, re.M)
    return {"state": state.lower(), "question": question.lower(), "options": [(l, t.lower()) for l, t in opts]}


def s1_probs(prompt: str, labels: list[str]) -> list[float]:
    """Heuristiques lisibles, deterministes, qui rendent la demo credible."""
    p = parse_s1_prompt(prompt)
    st, qu = p["state"], p["question"]
    req = st
    m = re.search(r'"(?:request|user_request)":\s*"((?:[^"\\]|\\.)*)"', st)
    if m:
        req = m.group(1).lower()
    act = any(w in req for w in ACTION_WORDS)
    risky = any(w in st for w in RISKY)
    if labels == ["Yes", "No"]:
        y = 0.25 + 0.2 * _h(prompt)
        if "short direct answer" in qu:
            y = 0.12 if act else 0.93
        elif "ambiguous" in qu:
            y = 0.08
        elif "multi-step reasoning" in qu:
            y = 0.7 if act or any(w in req for w in ("pourquoi", "why", "explique", "explain", "calcule", "prove")) else 0.2
        elif "would the tool" in qu:
            tool = re.search(r"`([a-z_]+)`", qu)
            t = tool.group(1) if tool else ""
            rel = {"write_file": act, "edit_file": any(w in req for w in ("modifie", "change", "corrige", "fix", "ajoute")),
                   "read_file": True, "glob": True, "grep": "cherche" in req or "find" in req, "run_command": act,
                   "python": "calcule" in req or "python" in req, "browse": any(w in req for w in ("web", "internet", "site web", "cherche en ligne")),
                   "list_files": True}.get(t, False)
            y = 0.86 if rel else 0.1 + 0.1 * _h(t)
        elif "violate" in qu:
            y = 0.8 if risky else 0.03
        elif "satisfy" in qu:
            y = 0.91
        elif "wake" in qu or "addressed to the assistant" in qu:
            y = 0.9
        return [y, 1 - y]
    opts = p["options"]
    if not opts:
        return _norm([1.0 + _h(prompt + l) for l in labels])
    if "risk posture" in qu:
        return _norm([0.05 if risky else 0.9, 0.85 if risky else 0.04, 0.03, 0.03][: len(labels)])
    if "how costly" in qu:
        return _norm([0.1, 0.2, 0.3, 0.4] if risky else [0.82, 0.13, 0.04, 0.01])
    words = set(re.findall(r"[a-zà-ÿ_]{3,}", req))
    scores = []
    for lab, text in opts:
        ow = set(re.findall(r"[a-zà-ÿ_]{3,}", text))
        scores.append(math.exp(2.2 * len(words & ow)) * (1.0 + 0.3 * _h(text)))
    if "observation only" in qu and "kind of request" in qu:
        for i, (lab, text) in enumerate(opts):
            if text.startswith("create_app") and act:
                scores[i] *= 30
            if text.startswith("chat") and not act:
                scores[i] *= 30
    return _norm(scores)


# ---- scenario d'agent -------------------------------------------------------------------------------------------------
DEMO_APP = """<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><title>Minuteur Pomodoro</title>
<style>body{font:16px system-ui;background:#0b0c10;color:#e8e8f0;display:grid;place-items:center;height:100vh;margin:0}
.t{font-size:72px;font-variant-numeric:tabular-nums}button{font:inherit;padding:.6em 1.4em;border-radius:12px;border:0;background:#7c5cff;color:#fff}</style>
</head><body><main><div class="t" id="t">25:00</div><button id="go">Demarrer</button></main>
<script>let s=1500,h;const t=document.getElementById('t');document.getElementById('go').onclick=()=>{clearInterval(h);h=setInterval(()=>{s=Math.max(0,s-1);t.textContent=String(Math.floor(s/60)).padStart(2,'0')+':'+String(s%60).padStart(2,'0')},1000)}</script>
</body></html>
"""


def _last_user(messages: list[dict]) -> tuple[str, int]:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            c = messages[i].get("content")
            if isinstance(c, list):
                c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
            return str(c or ""), i
    return "", -1


def agent_reply(messages: list[dict], tools: list[dict] | None) -> dict:
    """Message assistant suivant : {"reasoning", "content", "tool_calls": [(name, args)]}."""
    user, idx = _last_user(messages)
    low = user.lower()
    tool_names = {t["function"]["name"] for t in tools or []}
    done_steps = [m for m in messages[idx + 1:] if m.get("role") == "tool"]
    n = len(done_steps)
    if not tools:
        if any(w in low for w in ("bonjour", "salut", "hello")):
            return {"reasoning": "", "content": "Bonjour ! Je suis Prophet, votre agent local. Bonsai 2 27B raisonne, le classifieur decide en quelques dizaines de millisecondes. Que construisons-nous ?"}
        return {"reasoning": "", "content": "Voici la reponse courte : le classifieur System One a juge qu'une reponse directe suffisait, donc Bonsai a repondu sans phase de reflexion, en une ou deux secondes."}
    plan_mode = any("PLAN MODE" in (m.get("content") or "") for m in messages if m.get("role") == "system")
    if plan_mode:
        if n == 0 and "glob" in tool_names:
            return {"reasoning": "Mode plan : j'explore d'abord sans rien modifier.", "content": "", "tool_calls": [("glob", {"pattern": "**/*"})]}
        return {"reasoning": "J'ai assez d'elements pour proposer un plan.", "content": "",
                "tool_calls": [("done", {"summary": "Plan propose :\n1. Creer `index.html` (structure + styles).\n2. Ajouter la logique du minuteur en JavaScript.\n3. Verifier dans le navigateur.\n\nValidez pour que je l'execute."})]}
    if any(w in low for w in ("modifie", "change", "corrige", "fix")):
        steps = [("read_file", {"path": "index.html"}),
                 ("edit_file", {"path": "index.html", "old_string": "25:00", "new_string": "50:00"}),
                 ("done", {"summary": "Duree passee a 50 minutes dans `index.html`."})]
    else:
        steps = [("glob", {"pattern": "*"}),
                 ("write_file", {"path": "index.html", "content": DEMO_APP}),
                 ("python", {"code": "import os\nprint(sorted(os.listdir('.')))"}),
                 ("done", {"summary": "J'ai cree **`index.html`** : un minuteur Pomodoro autonome (HTML/CSS/JS, aucune dependance).\n\n"
                                      "- Ouvrez le fichier dans un navigateur pour le lancer.\n- Le bouton *Demarrer* lance un compte a rebours de 25 minutes.\n\n"
                                      "Prochaine etape possible : ajouter une notification sonore a la fin."})]
    steps = [s for s in steps if s[0] in tool_names or s[0] == "done"]
    step = steps[min(n, len(steps) - 1)]
    thoughts = ["Je regarde d'abord ce que contient l'espace de travail, pour ne rien ecraser.",
                "Rien d'existant : j'ecris une application autonome en un seul fichier, facile a ouvrir.",
                "Je verifie que le fichier est bien la avant de conclure.", "Tout est en place, je resume."]
    return {"reasoning": thoughts[min(n, len(thoughts) - 1)], "content": "", "tool_calls": [step]}


def _chunks(text: str, size: int = 4) -> list[str]:
    words = re.findall(r"\S+\s*|\s+", text)
    return ["".join(words[i:i + size // 2 or 1]) for i in range(0, len(words), size // 2 or 1)] if text else []


class FakeLlama:
    def __init__(self, name: str = "fake-bonsai.gguf", ctx: int = 8192, tps: float = 0.0):
        self.name, self.ctx, self.tps = name, ctx, tps
        self.requests: list[tuple[str, dict]] = []

    def handler(self):
        fake = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _json(self, code: int, obj) -> None:
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                if self.path.startswith("/health"):
                    return self._json(200, {"status": "ok"})
                if self.path.startswith("/props"):
                    return self._json(200, {"model_path": fake.name, "default_generation_settings": {"n_ctx": fake.ctx}, "total_slots": 1})
                if self.path.startswith("/v1/models"):
                    return self._json(200, {"object": "list", "data": [{"id": fake.name, "object": "model"}]})
                return self._json(404, {"error": "not found"})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                fake.requests.append((self.path, body))
                if self.path.startswith("/completion"):
                    labels = re.findall(r'"((?:[^"\\]|\\.)*)"', body.get("grammar", "")) or ["Yes", "No"]
                    labels = [l[:-1] if l.endswith('"') else l for l in labels]
                    probs = s1_probs(body.get("prompt", ""), labels)
                    top = [{"id": i, "token": lab, "prob": pr} for i, (lab, pr) in enumerate(zip(labels, probs))]
                    return self._json(200, {"content": labels[max(range(len(labels)), key=lambda i: probs[i])],
                                            "completion_probabilities": [{"top_probs": top}],
                                            "timings": {"prompt_n": 12, "cache_n": max(0, len(body.get("prompt", "")) // 4 - 12)}})
                if self.path.startswith("/tokenize"):
                    return self._json(200, {"tokens": list(range(len(str(body.get("content", "")).split())))})
                if self.path.startswith("/apply-template"):
                    msgs = body.get("messages", [])
                    return self._json(200, {"prompt": "".join(f"<|im_start|>{m['role']}\n{m.get('content', '')}<|im_end|>\n" for m in msgs) + "<|im_start|>assistant\n"})
                if self.path.startswith("/v1/chat/completions"):
                    return self.chat(body)
                return self._json(404, {"error": "not found"})

            def chat(self, body: dict):
                msgs, tools = body.get("messages", []), body.get("tools")
                rep = agent_reply(msgs, tools)
                think = rep.get("reasoning", "") if body.get("thinking_budget_tokens", -1) != 0 else ""
                calls = [{"id": f"call_{int(time.time() * 1000) % 100000}_{i}", "type": "function",
                          "function": {"name": nm, "arguments": json.dumps(a, ensure_ascii=False)}} for i, (nm, a) in enumerate(rep.get("tool_calls", []))]
                ntok = max(1, (len(think) + len(rep.get("content", "")) + sum(len(c["function"]["arguments"]) for c in calls)) // 4)
                timings = {"prompt_n": 900, "prompt_ms": 310.0, "predicted_n": ntok, "predicted_ms": ntok / (fake.tps or 48) * 1000,
                           "predicted_per_second": fake.tps or 48.0}
                if not body.get("stream"):
                    msg = {"role": "assistant", "content": rep.get("content", "")}
                    if think:
                        msg["reasoning_content"] = think
                    if calls:
                        msg["tool_calls"] = calls
                    return self._json(200, {"choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if calls else "stop"}],
                                            "usage": {"completion_tokens": ntok}, "timings": timings})
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                delay = 1.0 / fake.tps * 2 if fake.tps else 0.0

                def send(obj) -> bool:
                    try:
                        self.wfile.write(f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode()); self.wfile.flush()
                        if delay:
                            time.sleep(delay)
                        return True
                    except (BrokenPipeError, ConnectionResetError):
                        return False

                def delta(d):
                    return {"choices": [{"index": 0, "delta": d, "finish_reason": None}]}
                for piece in _chunks(think):
                    if not send(delta({"reasoning_content": piece})):
                        return
                for piece in _chunks(rep.get("content", "")):
                    if not send(delta({"content": piece})):
                        return
                for i, c in enumerate(calls):
                    send(delta({"tool_calls": [{"index": i, "id": c["id"], "type": "function", "function": {"name": c["function"]["name"], "arguments": ""}}]}))
                    a = c["function"]["arguments"]
                    for k in range(0, len(a), 48):
                        if not send(delta({"tool_calls": [{"index": i, "function": {"arguments": a[k:k + 48]}}]})):
                            return
                send({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if calls else "stop"}], "timings": timings,
                      "usage": {"completion_tokens": ntok}})
                try:
                    self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                self.close_connection = True

        return H

    def serve(self, host: str = "127.0.0.1", port: int = 0) -> tuple[ThreadingHTTPServer, threading.Thread]:
        srv = ThreadingHTTPServer((host, port), self.handler())
        srv.daemon_threads = True
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        return srv, th


def main(argv=None):
    ap = argparse.ArgumentParser(description="faux llama-server")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("-c", "--ctx-size", type=int, default=8192)
    ap.add_argument("-m", "--model", default="fake-bonsai.gguf")
    ap.add_argument("--tps", type=float, default=float(os.environ.get("FAKE_LLAMA_TPS", "0")))
    args, _unknown = ap.parse_known_args(argv)
    limit = int(os.environ.get("FAKE_LLAMA_OOM_ABOVE_CTX", "0") or 0)
    print(f"main: loading model '{args.model}' n_ctx={args.ctx_size}", flush=True)
    if limit and args.ctx_size > limit:
        print("ggml_backend_cuda_buffer_type_alloc_buffer: allocating 1536.00 MiB on device 0: cudaMalloc failed: out of memory", file=sys.stderr, flush=True)
        print("llama_init_from_model: failed to initialize the context: failed to allocate buffer for kv cache", file=sys.stderr, flush=True)
        sys.exit(1)
    delay = float(os.environ.get("FAKE_LLAMA_LOAD_S", "0") or 0)
    if delay:
        time.sleep(delay)
    fake = FakeLlama(os.path.basename(args.model), args.ctx_size, args.tps)
    srv = ThreadingHTTPServer((args.host, args.port), fake.handler())
    print(f"main: server is listening on http://{args.host}:{args.port} - starting the main loop", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

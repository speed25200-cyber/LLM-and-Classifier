import numpy as np
import pytest

from jev_clone.backend_llamacpp import BranchResult
from jev_clone.readout import probs_to_logits


class MockBackend:
    """Backend deterministe : la probabilite de chaque etiquette est fixee par un dictionnaire
    {qid: [p_label0, p_label1, ...]} exprime dans l'ordre des `keys` d'origine (non permute)."""

    def __init__(self, table: dict[str, list[float]], chat_reply: str = "", default_noul=(0.3, 0.7)):
        self.table = table
        self.chat_reply = chat_reply
        self.default_noul = list(default_noul)
        self.calls = []
        self.chat_calls = []

    def _row(self, b):
        """probabilites pour la branche b : table[qid] sinon un defaut (noul -> default_noul, sinon uniforme)."""
        if b.qid in self.table:
            return self.table[b.qid]
        if b.kind == "noul":
            return self.default_noul
        return [1.0 / len(b.keys)] * len(b.keys)

    def model_name(self):
        return "mock"

    def score_branches(self, prefix, branches):
        self.calls.append((prefix, [b.text for b in branches]))
        out = []
        for b in branches:
            base = self._row(b)
            # keys d'origine : noul -> [True, False] ; choice -> options ; score -> 0..n-1
            if b.kind == "noul":
                order = {True: 0, False: 1}
            elif b.kind == "score":
                order = {k: k for k in range(len(base))}
            else:
                order = self._choice_order(b)
            p = np.array([base[order[k]] for k in b.keys], dtype=float)
            out.append(BranchResult(logits=probs_to_logits(p / p.sum()), prompt_tokens=10, cached_tokens=5, ms=1.0))
        return out

    def _choice_order(self, b):
        # les keys d'une branche 'choice' sont les noms d'options ; la table est indexee dans l'ordre declare
        declared = self.declared.get(b.qid) if hasattr(self, "declared") else None
        if declared is None or len(declared) != len(b.keys):
            declared = sorted(b.keys)
        return {k: declared.index(k) for k in b.keys}

    def chat(self, messages, **kw):
        self.chat_calls.append((messages, kw))
        return {"choices": [{"message": {"content": self.chat_reply, "reasoning_content": "thinking..."}}]}


@pytest.fixture
def ticket_request():
    return {
        "state": {"ticket": "My payouts have failed three times this week and nobody replied to my emails."},
        "questions": {
            "team": {"type": "choice", "instructions": "Which team should handle this?",
                     "criteria": {"payments": "payouts, refunds", "account": "login, 2FA", "other": None}},
            "escalate": {"type": "noul", "instructions": "Should this be escalated to a manager?"},
            "urgency": {"type": "score", "instructions": "How urgent is this?",
                        "criteria": ["can wait a week", "handle today", "blocked right now"]},
        },
    }


@pytest.fixture
def mock_backend():
    be = MockBackend({"team": [0.9, 0.06, 0.04], "escalate": [0.7, 0.3], "urgency": [0.1, 0.2, 0.7]})
    be.declared = {"team": ["payments", "account", "other"]}
    return be


class ScriptedBackend(MockBackend):
    """Une table par appel de score_branches (le dernier script est reutilise ensuite)."""

    def __init__(self, tables: list[dict], declared: dict | None = None, **kw):
        super().__init__(tables[0], **kw)
        self.tables = tables
        self.declared = declared or {}

    def score_branches(self, prefix, branches):
        self.table = self.tables[min(len(self.calls), len(self.tables) - 1)]
        return super().score_branches(prefix, branches)


class MockS2:
    """Bonsai factice : renvoie une suite de messages (avec ou sans tool_calls) puis le dernier en boucle."""

    def __init__(self, replies: list[dict]):
        self.replies = replies
        self.calls = []

    @staticmethod
    def tool_call(name, args, cid="call_1"):
        import json as _j
        return {"id": cid, "type": "function", "function": {"name": name, "arguments": _j.dumps(args)}}

    def chat(self, messages, **kw):
        self.calls.append((list(messages), kw))
        msg = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        return {"choices": [{"message": {"role": "assistant", **msg}}]}


@pytest.fixture
def big_gguf(monkeypatch):
    """GGUF de grande taille simulee : petit fichier sur le disque, taille vue par le planificateur (catalog.gguf_size). Un
    fichier creux (truncate) occuperait vraiment des dizaines de Go sur le runner Windows."""
    from pathlib import Path
    from prophet_studio import catalog
    sizes: dict[str, int] = {}
    real = catalog.gguf_size
    monkeypatch.setattr(catalog, "gguf_size", lambda p: sizes.get(str(Path(p).resolve()), None) or real(Path(p)))

    def make(path, size: int):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"GGUF" + b"\0" * 60)
        sizes[str(path.resolve())] = int(size)
        return path
    return make

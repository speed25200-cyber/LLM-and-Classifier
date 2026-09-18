import numpy as np
import pytest

from jev_clone.backend_llamacpp import BranchResult
from jev_clone.readout import probs_to_logits


class MockBackend:
    """Backend deterministe : la probabilite de chaque etiquette est fixee par un dictionnaire
    {qid: [p_label0, p_label1, ...]} exprime dans l'ordre des `keys` d'origine (non permute)."""

    def __init__(self, table: dict[str, list[float]], chat_reply: str = ""):
        self.table = table
        self.chat_reply = chat_reply
        self.calls = []
        self.chat_calls = []

    def model_name(self):
        return "mock"

    def score_branches(self, prefix, branches):
        self.calls.append((prefix, [b.text for b in branches]))
        out = []
        for b in branches:
            base = self.table[b.qid]
            # keys d'origine : noul -> [True, False] ; choice -> options ; score -> 0..n-1
            if b.kind == "noul":
                order = {True: 0, False: 1}
            elif b.kind == "score":
                order = {k: k for k in range(len(base))}
            else:
                order = {k: i for i, k in enumerate(sorted(b.keys, key=lambda k: b.keys.index(k)) if len(self.calls) == 0 else b.keys)}
                order = self._choice_order(b)
            p = np.array([base[order[k]] for k in b.keys], dtype=float)
            out.append(BranchResult(logits=probs_to_logits(p / p.sum()), prompt_tokens=10, cached_tokens=5, ms=1.0))
        return out

    def _choice_order(self, b):
        # les keys d'une branche 'choice' sont les noms d'options ; la table est indexee dans l'ordre declare
        declared = self.declared.get(b.qid) if hasattr(self, "declared") else None
        if declared is None:
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

"""Test d'integration contre un vrai llama-server (JEV_TEST_SERVER=http://127.0.0.1:8099).
Le modele peut etre minuscule / aleatoire : on verifie la plomberie (grammaire, probabilites, cache)."""
import os

import pytest

from jev_clone.backend_llamacpp import LlamaCppBackend
from jev_clone.engine import SystemOneEngine

URL = os.environ.get("JEV_TEST_SERVER")
pytestmark = pytest.mark.skipif(not URL, reason="JEV_TEST_SERVER non defini")


def test_live_roundtrip(ticket_request):
    be = LlamaCppBackend(URL, max_workers=2)
    assert be.health()
    eng = SystemOneEngine(be)
    r1 = eng.answer(ticket_request)
    a = r1.answers
    assert abs(sum(a["team"].probabilities.values()) - 1) < 1e-4
    assert 0 <= a["escalate"].noul <= 1 and 0 <= a["urgency"].score <= 2
    assert r1.usage.branches == 3 and r1.latency_ms > 0
    r2 = eng.answer(ticket_request)
    assert r2.usage.state_cache_hit  # le prefixe d'etat est servi depuis le cache
    assert abs(a["team"].probabilities["payments"] - r2.answers["team"].probabilities["payments"]) < 0.05


def test_live_permutations(ticket_request):
    eng = SystemOneEngine(LlamaCppBackend(URL, max_workers=2))
    r = eng.answer(dict(ticket_request, permutations=3))
    assert r.usage.branches == 7 and abs(sum(r.answers["team"].probabilities.values()) - 1) < 1e-4

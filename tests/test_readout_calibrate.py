import numpy as np

from jev_clone.calibrate import fit_temperature, report, threshold_for_precision
from jev_clone.engine import SystemOneEngine
from jev_clone.readout import Calibration, confidence, probs_to_logits, softmax


def test_softmax_temperature_flattens():
    p = softmax(probs_to_logits(np.array([0.9, 0.1])), 1.0)
    assert abs(p[0] - 0.9) < 1e-9
    p2 = softmax(probs_to_logits(np.array([0.9, 0.1])), 2.0)
    assert 0.5 < p2[0] < 0.9


def test_confidence_bounds():
    assert confidence(np.array([1.0, 0.0, 0.0])) > 0.999
    assert abs(confidence(np.array([1 / 3, 1 / 3, 1 / 3]))) < 1e-9
    assert 0 < confidence(np.array([0.7, 0.2, 0.1])) < 1


def test_engine_answers(mock_backend, ticket_request):
    eng = SystemOneEngine(mock_backend, model_name="mock")
    resp = eng.answer(ticket_request)
    a = resp.answers
    assert a["team"].choice == "payments" and abs(sum(a["team"].probabilities.values()) - 1) < 1e-6
    assert abs(a["escalate"].noul - 0.7) < 1e-6
    assert abs(a["urgency"].score - (0.2 + 1.4)) < 1e-6 and a["urgency"].legend["2"] == "blocked right now"
    assert resp.usage.branches == 3 and resp.model == "mock" and resp.latency_ms >= 0
    assert mock_backend.calls[0][0].startswith("<|im_start|>system")


def test_engine_permutations_average(mock_backend, ticket_request):
    ticket_request = dict(ticket_request, permutations=3)
    eng = SystemOneEngine(mock_backend, model_name="mock")
    resp = eng.answer(ticket_request)
    assert resp.usage.branches == 3 * 2 + 1  # noul non permute
    assert resp.answers["team"].choice == "payments"
    assert abs(resp.answers["team"].probabilities["payments"] - 0.9) < 1e-6


def test_calibration_temperature_applies(mock_backend, ticket_request):
    eng = SystemOneEngine(mock_backend, calibration=Calibration(temperature={"choice": 3.0}), model_name="mock")
    resp = eng.answer(ticket_request)
    assert resp.answers["team"].probabilities["payments"] < 0.9
    assert abs(resp.answers["escalate"].noul - 0.7) < 1e-6  # T noul = 1


def test_report_and_fit_temperature():
    rng = np.random.default_rng(0)
    n, K = 600, 4
    # modele calibre a T=1 (les etiquettes sont tirees de sa propre distribution), puis rendu
    # sur-confiant en multipliant ses logits par 3 : la temperature optimale doit revenir vers 3
    base = rng.normal(0, 1.5, (n, K))
    labels = np.array([rng.choice(K, p=softmax(l)) for l in base])
    logits = base * 3.0
    raw = np.stack([softmax(l) for l in logits])
    r0 = report(raw, labels)
    T = fit_temperature(logits, labels)
    r1 = report(np.stack([softmax(l, T) for l in logits]), labels)
    assert 2.0 < T < 4.5, T
    assert r1.ece < r0.ece and r1.nll < r0.nll and r1.accuracy == r0.accuracy
    assert "ECE" in str(r1)


def test_threshold_for_precision():
    conf = np.array([0.99, 0.95, 0.9, 0.8, 0.7, 0.6])
    correct = np.array([1, 1, 1, 0, 1, 0])
    thr = threshold_for_precision(conf, correct, 0.95)
    assert thr == 0.9
    assert threshold_for_precision(np.array([0.5]), np.array([0]), 0.9) is None


def test_calibration_roundtrip(tmp_path):
    cal = Calibration(temperature={"choice": 1.7}, thresholds={"team": 0.8})
    cal.save(tmp_path / "c.json")
    cal2 = Calibration.load(tmp_path / "c.json")
    assert cal2.t("choice") == 1.7 and cal2.thresholds["team"] == 0.8 and cal2.t("noul") == 1.0


def test_aggregate_label_probs():
    from jev_clone.backend_llamacpp import aggregate_label_probs
    top = [{"token": "Y", "prob": 0.3}, {"token": "Yes", "prob": 0.4}, {"token": "No", "prob": 0.2},
           {"token": "N", "prob": 0.1}, {"token": "Maybe", "prob": 0.5}]
    p = aggregate_label_probs(top, ["Yes", "No"])
    assert abs(p[0] - 0.7) < 1e-9 and abs(p[1] - 0.3) < 1e-9
    p = aggregate_label_probs([{"token": "B", "logprob": 0.0}], ["A", "B", "C"])
    assert p[1] == 1.0
    p = aggregate_label_probs([], ["A", "B"])
    assert abs(p[0] - 0.5) < 1e-9

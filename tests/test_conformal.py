import numpy as np

from jev_clone.conformal import ConformalCalibrator, binomial_upper, conformal_quantile, hoeffding_upper, risk_controlled_threshold
from jev_clone.readout import softmax


def _synthetic(n, K, seed, temp=1.0):
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1.5, (n, K))
    labels = np.array([rng.choice(K, p=softmax(l)) for l in base])   # etiquettes tirees du modele : calibre a T=1
    probs = np.stack([softmax(l * temp) for l in base])
    return probs, labels


def test_conformal_coverage_holds():
    cal_p, cal_y = _synthetic(2000, 4, 0)
    test_p, test_y = _synthetic(4000, 4, 1)
    cc = ConformalCalibrator(alpha=0.1)
    q = cc.fit("choice", cal_p, cal_y)
    assert 0 < q < 1
    cov, size = cc.coverage("choice", test_p, test_y)
    assert 0.88 <= cov <= 0.93, cov          # garantie marginale >= 0.90 (a la variance pres)
    assert 1.0 < size < 4.0
    # modele sur-confiant (T=3) : l'ensemble grandit pour garder la couverture
    cc2 = ConformalCalibrator(alpha=0.1)
    over_p, over_y = _synthetic(2000, 4, 2, temp=3.0)
    cc2.fit("choice", over_p, over_y)
    cov2, size2 = cc2.coverage("choice", *_synthetic(4000, 4, 3, temp=3.0))
    assert cov2 >= 0.88 and size2 > size


def test_prediction_set_and_roundtrip(tmp_path):
    cc = ConformalCalibrator(alpha=0.1, qhat={"choice": 0.7}, n={"choice": 500})
    assert cc.prediction_set("choice", {"a": 0.5, "b": 0.35, "c": 0.15}) == ["a", "b"]
    assert cc.prediction_set("choice", {"a": 0.2, "b": 0.2, "c": 0.2, "d": 0.2, "e": 0.2}) == ["a"]  # jamais vide
    assert cc.prediction_set("noul", {"yes": 0.6, "no": 0.4}) == ["yes", "no"]  # pas calibre -> tout
    cc.save(tmp_path / "c.json")
    assert ConformalCalibrator.load(tmp_path / "c.json").qhat["choice"] == 0.7
    assert conformal_quantile(np.array([0.1, 0.2]), 0.1) == 1.0  # n trop petit


def test_risk_controlled_threshold():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.5, 1.0, 600)
    correct = (rng.uniform(size=600) < conf).astype(float)   # modele calibre
    r = risk_controlled_threshold(conf, correct, alpha=0.1, delta=0.1)
    assert r["threshold"] is not None and r["threshold"] > 0.8 and r["error_upper_bound"] <= 0.1
    assert 0.2 < r["coverage"] < 0.6 and r["empirical_error"] < 0.1
    bad = risk_controlled_threshold(conf, np.zeros(600), alpha=0.1, delta=0.1)
    assert bad["threshold"] is None
    assert hoeffding_upper(0.0, 100, 0.1) > 0.0 and hoeffding_upper(0.5, 0, 0.1) == 1.0
    assert abs(binomial_upper(0, 20, 0.1) - (1 - 0.1 ** (1 / 20))) < 1e-6   # forme close pour k = 0
    assert binomial_upper(5, 100, 0.1) > 0.05 and binomial_upper(5, 100, 0.1) < hoeffding_upper(0.05, 100, 0.1)

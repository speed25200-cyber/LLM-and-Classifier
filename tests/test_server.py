from fastapi.testclient import TestClient

from jev_clone.engine import SystemOneEngine
from jev_clone.fusion import FusionRouter, GatePolicy
from jev_clone.server import build_app
from tests.conftest import MockBackend


def test_systemone_endpoint(mock_backend, ticket_request):
    eng = SystemOneEngine(mock_backend, model_name="mock")
    app = build_app(eng, FusionRouter(eng, MockBackend({}, chat_reply='{"team":"other"}'), GatePolicy(0.99)))
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
    r = c.post("/v1/systemone", json=ticket_request)
    assert r.status_code == 200
    body = r.json()
    assert body["answers"]["team"]["choice"] == "payments" and body["answers"]["escalate"]["noul"] == 0.7
    assert c.post("/v1/systemone", json={"state": "x", "questions": {}}).status_code == 422
    f = c.post("/v1/fusion/decide", json=ticket_request).json()
    assert f["path"] == "escalated" and f["decisions"]["team"] == "other"

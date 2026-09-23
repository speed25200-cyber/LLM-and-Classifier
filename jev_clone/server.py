"""Serveur HTTP compatible TypeSafe : POST /v1/systemone (+ /v1/fusion/decide, /health).

    uvicorn jev_clone.server:app --port 8008
    JEV_S1_URL=http://127.0.0.1:8081  (llama-server du clone ; ou le serveur Bonsai en mode mono)
    JEV_S2_URL=http://127.0.0.1:8080  (llama-server Bonsai, optionnel : active /v1/fusion/decide)
    JEV_CALIBRATION=runs/calibration.json  (optionnel ; temperature seule : ignoree, T = 1 ; ancien format : signale)
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException

from jev_clone.backend_llamacpp import LlamaCppBackend
from jev_clone.engine import SystemOneEngine
from jev_clone.fusion import FusionRouter, GatePolicy
from jev_clone.readout import Calibration, load_calibration
from jev_clone.schema import SystemOneRequest


def build_app(s1_engine: SystemOneEngine | None = None, router: FusionRouter | None = None) -> FastAPI:
    app = FastAPI(title="jev-clone", version="0.1.0")
    state = {"engine": s1_engine, "router": router, "cal": None}

    def calibration() -> Calibration:
        """Lue une fois, politique de service (readout.load_calibration) : temperature seule jamais appliquee."""
        if state["cal"] is None:
            state["cal"] = load_calibration(os.environ.get("JEV_CALIBRATION"), agent=False)
        return state["cal"]

    def engine() -> SystemOneEngine:
        if state["engine"] is None:
            state["engine"] = SystemOneEngine(LlamaCppBackend(os.environ.get("JEV_S1_URL", "http://127.0.0.1:8081")),
                                              calibration=calibration())
        return state["engine"]

    def fusion() -> FusionRouter:
        if state["router"] is None:
            s2_url = os.environ.get("JEV_S2_URL")
            s2 = LlamaCppBackend(s2_url, max_workers=1) if s2_url else None
            state["router"] = FusionRouter(engine(), s2, GatePolicy.from_calibration(calibration()),
                                           ledger=os.environ.get("JEV_LEDGER", "runs/ledger.jsonl"))
        return state["router"]

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/v1/systemone")
    def systemone(payload: dict):
        try:
            req = SystemOneRequest.model_validate(payload)
        except Exception as e:  # pydantic ValidationError
            raise HTTPException(status_code=422, detail=str(e))
        return engine().answer(req).model_dump()

    @app.post("/v1/fusion/decide")
    def decide(payload: dict):
        try:
            req = SystemOneRequest.model_validate(payload)
        except Exception as e:
            raise HTTPException(status_code=422, detail=str(e))
        r = fusion().decide(req)
        return {"path": r.path, "decisions": r.decisions, "gated": r.gated,
                "s1": r.s1.model_dump(), "s2_text": r.s2_text, "s2_reasoning": r.s2_reasoning,
                "verification": r.verification, "latency_ms": r.latency_ms, "sources": r.sources, "unresolved": r.unresolved, "s2_error": r.s2_error}

    return app


app = build_app()

"""Stateless scoring API. Caller supplies trusted, past-only history summaries."""
from contextlib import asynccontextmanager
from datetime import datetime
import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator
from threadpoolctl import threadpool_limits

from paymentguard.features import FEATURES, transform


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    transaction_id: str = Field(min_length=1, max_length=128)
    timestamp: datetime
    amount: float = Field(ge=0, le=1e9)
    distance_km: float = Field(ge=0, le=50000)
    new_device: Literal[0, 1]
    international: Literal[0, 1]
    prior_txn_count_1h: int = Field(ge=0, le=1000000)
    prior_mean_amount: float = Field(ge=0, le=1e9)

    @field_validator("timestamp")
    @classmethod
    def aware_timestamp(cls, value):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must include a UTC offset")
        return value


class Prediction(BaseModel):
    transaction_id: str
    fraud_score: float
    threshold: float
    decision: Literal["review", "allow"]
    model_version: str
    data_kind: str


def create_app(model_path=None):
    path = Path(model_path or os.getenv("PAYMENTGUARD_MODEL", "artifacts/model.joblib"))

    @asynccontextmanager
    async def lifespan(app):
        # Only deserialize artifacts you trained or trust; joblib is executable.
        artifact = joblib.load(path) if path.exists() else None
        if artifact is not None and artifact["features"] != FEATURES:
            raise RuntimeError("Artifact feature schema is incompatible.")
        app.state.artifact = artifact
        # Keep OpenMP limits fixed for the lifetime, not per concurrent request.
        with threadpool_limits(limits=1):
            yield

    app = FastAPI(title="PaymentGuard", version="0.1.0", lifespan=lifespan,
                  description="Educational fraud scoring. Synthetic models are labeled in every response.")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        if app.state.artifact is None:
            raise HTTPException(503, "Train a model before scoring.")
        return {"status": "ready", "model_version": app.state.artifact["version"],
                "data_kind": app.state.artifact["data_kind"]}

    @app.post("/predict", response_model=Prediction)
    def predict(tx: Transaction):
        artifact = app.state.artifact
        if artifact is None:
            raise HTTPException(503, "Model unavailable. Run paymentguard train.")
        frame = pd.DataFrame([tx.model_dump()])
        score = float(artifact["model"].predict_proba(transform(frame))[0, 1])
        return Prediction(transaction_id=tx.transaction_id, fraud_score=score,
            threshold=artifact["threshold"], decision="review" if score >= artifact["threshold"] else "allow",
            model_version=artifact["version"], data_kind=artifact["data_kind"])

    return app


app = create_app()

import os
from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from src.deploy_model import load_model
    from src.crm_integration import score_customers
except Exception:
    from deploy_model import load_model
    from crm_integration import score_customers


MODEL_NAME = os.environ.get('MODEL_NAME', 'rf_churn_model')

app = FastAPI(title='Churn Inference API', version='1.0.0')


class BatchPredictRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(min_length=1)


class BatchPredictResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    n_records: int


_cached_artifact: Dict[str, Any] | None = None


def get_artifact() -> Dict[str, Any]:
    global _cached_artifact
    if _cached_artifact is None:
        _cached_artifact = load_model(MODEL_NAME)
    return _cached_artifact


@app.get('/health')
def health() -> Dict[str, str]:
    return {'status': 'ok'}


@app.post('/predict', response_model=BatchPredictResponse)
def predict_batch(payload: BatchPredictRequest) -> BatchPredictResponse:
    artifact = get_artifact()
    feature_names = artifact.get('feature_names')
    threshold = artifact.get('threshold', 0.36)
    model = artifact.get('model')

    df = pd.DataFrame(payload.records)
    if df.empty:
        raise HTTPException(status_code=400, detail='records must contain at least one row')

    if feature_names:
        missing = [col for col in feature_names if col not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f'Missing required features: {missing}')

    try:
        scored = score_customers(df=df, model=model, feature_names=feature_names or df.columns.tolist(), threshold=threshold)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Prediction failed: {exc}') from exc

    keep_cols = [
        'churn_proba',
        'churn_flag',
        'risk_tier',
        'retention_offer',
        'priority_rank',
        'scored_at',
    ]
    out = scored[[c for c in keep_cols if c in scored.columns]].to_dict(orient='records')
    return BatchPredictResponse(predictions=out, n_records=len(out))

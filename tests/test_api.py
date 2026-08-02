from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def ensure_model_exists():
    model_path = Path("models/tfidf_intent_pipeline.joblib")
    data_path = Path("data/customer_intents.csv")
    if not model_path.exists():
        # generate data and train
        import data.generate_data as gen
        from src.train import train_and_serialize

        gen.generate_csv(data_path)
        train_and_serialize(str(data_path), str(model_path))


@pytest.fixture(scope="module")
def client():
    ensure_model_exists()
    from src.app import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert "status" in body
    assert body.get("model_loaded") is True


def test_predict(client):
    payload = {"text": "My app crashes whenever I try to login."}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    j = r.json()
    assert j["text"] == payload["text"]
    assert isinstance(j["predicted_intent"], str)
    assert j["confidence"] >= 0.0
    assert j["latency_ms"] >= 0.0

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def ensure_model_exists():
    model_path = Path("models/tfidf_intent_pipeline.joblib")
    data_path = Path("data/customer_intents.csv")
    if not model_path.exists():
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


def test_predict_validation_error(client):
    # too short text -> validation error (min_length=2)
    r = client.post("/predict", json={"text": "A"})
    assert r.status_code == 422


def test_predict_nonstring(client):
    # Pydantic will coerce numeric input to string for `text: str`, so expect 200
    r = client.post("/predict", json={"text": 12345})
    assert r.status_code == 200
    j = r.json()
    assert j["text"] == "12345"


def test_predict_long_text(client):
    long_text = "error " * 2000
    r = client.post("/predict", json={"text": long_text})
    assert r.status_code == 200
    j = r.json()
    assert j["predicted_intent"] != ""
    assert 0.0 <= j["confidence"] <= 1.0


def test_negation_handling(client):
    payload = {"text": "I am NOT happy with the billing and I do not want to continue."}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    j = r.json()
    assert j["text"] == payload["text"]
    assert isinstance(j["predicted_intent"], str)

import time
from pathlib import Path
from typing import Optional

import joblib
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field

from src.cleaner import clean_text


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "tfidf_intent_pipeline.joblib"

app = FastAPI(title="Intent Classifier API")

# Mount static UI
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/ui")
def ui():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return RedirectResponse(url="/docs")

@app.get("/")
def root():
    # Redirect root to UI if available
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return RedirectResponse(url="/ui")
    return {"status": "healthy", "model_loaded": MODEL is not None}


class InferenceRequest(BaseModel):
    text: str = Field(..., min_length=2, example="I was double charged on my last invoice.")


class InferenceResponse(BaseModel):
    text: str
    predicted_intent: str
    confidence: float
    latency_ms: float


def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


MODEL = load_model()


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": MODEL is not None}


@app.post("/predict", response_model=InferenceResponse)
def predict(req: InferenceRequest):
    start = time.time()
    if MODEL is None:
        return {"text": req.text, "predicted_intent": "", "confidence": 0.0, "latency_ms": 0.0}

    cleaned = clean_text(req.text)
    probs = MODEL.predict_proba([cleaned])[0]
    classes = MODEL.classes_
    top_idx = probs.argmax()
    predicted = classes[top_idx]
    confidence = float(probs[top_idx])
    latency_ms = (time.time() - start) * 1000.0
    return InferenceResponse(text=req.text, predicted_intent=predicted, confidence=confidence, latency_ms=latency_ms)

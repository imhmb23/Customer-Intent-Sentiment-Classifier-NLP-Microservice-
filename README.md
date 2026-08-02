# Customer Intent & Sentiment Classifier

## 1. Overview
A small, self-contained demonstration project that accepts customer text, predicts an intent label (e.g., Billing, Technical Support, Account Cancellation, General Inquiry), and returns a simple confidence score. This repository shows basic data preprocessing, a simple machine learning pipeline, a small FastAPI service, unit tests, and a lightweight browser UI.

This is a clear, compact example for a junior developer portfolio — concise, runnable, and easy to explain in an interview.

## 2. What I built
- Synthetic dataset generation script (so reviewers can reproduce results without external downloads).
- A text-cleaning helper to normalize input before inference.
- A simple scikit-learn pipeline (TF-IDF vectorization + classifier) saved to `models/tfidf_intent_pipeline.joblib`.
- A FastAPI app exposing a health check and a prediction endpoint.
- A minimal browser UI served from the app to try the model interactively.
- Unit and integration tests with `pytest`.

## 3. Technology stack
- Python 3.9+ (recommended)
- scikit-learn, pandas, numpy
- FastAPI + Uvicorn
- pytest for tests
- joblib for model serialization

## 4. Quick start (local)
1. Create and activate a virtual environment (Windows example):

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\activate
```

2. Install pinned dependencies:

```bash
pip install -r requirements.txt
```

3. Generate data and train (one-line helper in `src/train.py`):

```bash
python -m src.train
```

4. Run the API locally:

```bash
uvicorn src.app:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/ui` to use the browser UI, or call the API with `POST /predict`.

## 5. API usage (examples)
- Health check: `GET /health` — returns a small JSON confirming the app is running.
- Prediction: `POST /predict` with JSON body `{ "text": "..." }` — returns predicted intent and a confidence score.

Minimal FastAPI health example (for reviewers):

```python
from fastapi import FastAPI

app = FastAPI()

@app.get('/health')
def health():
    return {"status": "healthy", "model_loaded": True}
```

## 6. Tests
Run the test suite with:

```bash
pytest -q
```

The tests include small unit checks for the cleaner and a couple of API integration tests.

## 7. UI screenshots
Four simple screenshots show the interactive UI (served at `/ui`). Replace these with real images if desired.

![UI 1](docs/images/ui_1.png)
![UI 2](docs/images/ui_2.png)
![UI 3](docs/images/ui_3.png)
![UI 4](docs/images/ui_4.png)

## 8. Files of interest (for reviewers)
- `src/app.py` — FastAPI app and endpoints
- `src/cleaner.py` — text preprocessing utilities
- `src/train.py` — data generation and training helper
- `models/tfidf_intent_pipeline.joblib` — serialized pipeline (small example model)
- `static/index.html` — simple browser UI

## 9. Notes for interview
- This project is intentionally compact so it is easy to explain end-to-end.
- Focus talking points: problem definition, simple preprocessing choices, why a lightweight model was chosen for this demo, how the API and UI connect, and how tests validate behavior.
- Avoid claiming production deployment or enterprise-scale guarantees — present this as a demonstration and learning project.

---

If you'd like, I can (A) add short runnable commands for Windows/macOS, (B) add screenshot files into `docs/images/`, or (C) remove the bundled model and show how to generate it on demand. Which would you prefer?

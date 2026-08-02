# Intent Classifier Assessment

This repository contains a small NLP microservice that classifies customer text inputs into intents and reports confidence and latency. Follow the instructions in `assessment_requirements.md` to generate data, train the model, and run the API.

Quick commands:

```powershell
python -m venv venv
venv\Scripts\pip.exe install -r requirements.txt
python data/generate_data.py
python src/train.py
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
```

from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn import metrics
from src.cleaner import clean_text


def train_and_serialize(data_csv: str = "data/customer_intents.csv", model_path: str = "models/tfidf_intent_pipeline.joblib"):
    data_path = Path(data_csv)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run data/generate_data.py first.")
    df = pd.read_csv(data_path)
    df["clean_text"] = df["text"].fillna("").astype(str).map(clean_text)

    X = df["clean_text"].values
    y = df["intent"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    acc = metrics.accuracy_score(y_test, y_pred)
    report = metrics.classification_report(y_test, y_pred)
    f1_macro = metrics.f1_score(y_test, y_pred, average="macro")

    print("Accuracy:", acc)
    print("Macro F1:", f1_macro)
    print(report)

    model_file = Path(model_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_file)
    print(f"Serialized pipeline to {model_file}")
    return pipeline


if __name__ == "__main__":
    import data.generate_data as gen

    csv_path = Path(__file__).resolve().parent.parent / "data" / "customer_intents.csv"
    if not csv_path.exists():
        print("Dataset not found — generating synthetic data...")
        gen.generate_csv(csv_path)
    train_and_serialize(str(csv_path), str(Path(__file__).resolve().parent.parent / "models" / "tfidf_intent_pipeline.joblib"))

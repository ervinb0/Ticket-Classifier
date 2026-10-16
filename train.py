"""
Train the ticket classifier and save it as a single sklearn Pipeline.

    python train.py                      # uses datasets/sample1.csv
    python train.py --data path/to.csv --out artifacts/model.joblib

The vectorizer and model are saved together, so they can never get out of sync.
Metrics for the validation and test splits are written next to the model.
"""
import argparse
import json
import os
from datetime import datetime, timezone
from multiprocessing import Pool

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from models.text_pre_processor import preprocess_text

REQUIRED_COLUMNS = {"issue_label", "issue_title", "issue_body"}
LABELS = ["bug", "enhancement", "question"]
SEED = 42


def load_dataset(path):
    if not os.path.exists(path):
        raise SystemExit(
            f"Dataset not found at {path}.\n"
            "This project uses the NLBSE'22 issue report classification dataset "
            "(see README > Data). Place the CSV there or pass --data."
        )
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise SystemExit(f"Dataset is missing required columns: {sorted(missing)}")

    unknown = set(df["issue_label"].unique()) - set(LABELS)
    if unknown:
        raise SystemExit(f"Dataset has unexpected labels: {sorted(unknown)}")
    return df


def preprocess_all(texts, cache_path):
    """Preprocessing is the slow step, so cache it and use all CPU cores."""
    if os.path.exists(cache_path):
        cached = pd.read_pickle(cache_path)
        if len(cached) == len(texts):
            print(f"Using cached preprocessed text from {cache_path}")
            return cached

    print(f"Preprocessing {len(texts)} issues (this takes a few minutes)...")
    with Pool() as pool:
        processed = pool.map(preprocess_text, texts.tolist(), chunksize=500)
    processed = pd.Series(processed, index=texts.index)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    processed.to_pickle(cache_path)
    return processed


def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000)),
        ("clf", RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=SEED, n_jobs=-1
        )),
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="datasets/sample1.csv")
    parser.add_argument("--out", default="artifacts/model.joblib")
    parser.add_argument("--cache", default="datasets/cache/processed_text.pkl")
    args = parser.parse_args()

    df = load_dataset(args.data)
    text = (df["issue_title"].fillna("") + " " + df["issue_body"].fillna("")).str.strip()
    df["processed_text"] = preprocess_all(text, args.cache)

    before = len(df)
    df = df[df["processed_text"].notna()]
    print(f"Kept {len(df)} of {before} issues after language/empty filtering")

    X, y = df["processed_text"], df["issue_label"]

    # 70% train / 15% validation / 15% test, stratified so "question" is present in every split
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, random_state=SEED, stratify=y_rest
    )

    pipeline = build_pipeline()
    print(f"Training on {len(X_train)} issues...")
    pipeline.fit(X_train, y_train)

    metrics = {}
    for split, X_s, y_s in [("validation", X_val, y_val), ("test", X_test, y_test)]:
        y_pred = pipeline.predict(X_s)
        print(f"\n{split} set\n" + classification_report(y_s, y_pred, digits=3))
        metrics[split] = classification_report(y_s, y_pred, output_dict=True)

    version = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump({"pipeline": pipeline, "version": version}, args.out, compress=3)

    metrics_path = os.path.join(os.path.dirname(args.out), "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "model_version": version,
            "data": args.data,
            "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
            **metrics,
        }, f, indent=2)

    print(f"\nSaved model to {args.out} (version {version}) and metrics to {metrics_path}")


if __name__ == "__main__":
    main()

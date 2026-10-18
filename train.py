"""
Train the ticket classifier and track the run in MLflow.

    python train.py                      # experiment: evaluate on validation only
    python train.py --model rf           # pick a model (see MODELS)
    python train.py --final              # also evaluate on test and save the model

Experiments never touch the test set or artifacts/model.joblib. Only a --final
run does, once you have picked the model on the validation set.
The vectorizer and model are saved together, so they can never get out of sync.
"""
import argparse
import json
import os
from datetime import datetime, timezone
from multiprocessing import Pool

import joblib
import matplotlib
matplotlib.use("Agg")  # no GUI needed, we only save figures
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from models.text_pre_processor import preprocess_text

REQUIRED_COLUMNS = {"issue_label", "issue_title", "issue_body"}
LABELS = ["bug", "enhancement", "question"]
SEED = 42
EXPERIMENT_NAME = "issue-classifier"

# Model name -> function returning a fresh classifier. Add baselines here.
MODELS = {
    "rf": lambda: RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=SEED, n_jobs=-1
    ),
}


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


def build_pipeline(model_name, max_features):
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=max_features)),
        ("clf", MODELS[model_name]()),
    ])


def evaluate(pipeline, X, y, split):
    """Print the report, log the headline metrics and a confusion matrix to MLflow."""
    y_pred = pipeline.predict(X)
    print(f"\n{split} set\n" + classification_report(y, y_pred, digits=3, zero_division=0))
    report = classification_report(y, y_pred, output_dict=True, zero_division=0)

    mlflow.log_metrics({
        f"{split}_macro_f1": report["macro avg"]["f1-score"],
        f"{split}_accuracy": report["accuracy"],
        f"{split}_question_recall": report["question"]["recall"],
        f"{split}_question_f1": report["question"]["f1-score"],
    })

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y, y_pred, labels=LABELS, ax=ax, colorbar=False)
    ax.set_title(f"{split} confusion matrix")
    mlflow.log_figure(fig, f"confusion_matrix_{split}.png")
    plt.close(fig)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="rf", choices=sorted(MODELS))
    parser.add_argument("--max-features", type=int, default=10000)
    parser.add_argument("--final", action="store_true",
                        help="Evaluate on the test set and save the model. Use once, at the end.")
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

    mlflow.set_experiment(EXPERIMENT_NAME)
    run_name = f"{args.model}-final" if args.final else args.model
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("final", args.final)
        mlflow.log_params({
            "model": args.model,
            "max_features": args.max_features,
            "seed": SEED,
            "n_train": len(X_train),
            "n_val": len(X_val),
        })

        pipeline = build_pipeline(args.model, args.max_features)
        print(f"Training {args.model} on {len(X_train)} issues...")
        pipeline.fit(X_train, y_train)

        metrics = {"validation": evaluate(pipeline, X_val, y_val, "val")}

        if not args.final:
            print("\nExperiment run: test set not evaluated, model not saved. "
                  "Compare runs with `mlflow ui --port 5001`.")
            return

        metrics["test"] = evaluate(pipeline, X_test, y_test, "test")

        version = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        joblib.dump({"pipeline": pipeline, "version": version}, args.out, compress=3)

        metrics_path = os.path.join(os.path.dirname(args.out), "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump({
                "model_version": version,
                "model": args.model,
                "data": args.data,
                "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
                **metrics,
            }, f, indent=2)
        mlflow.log_artifact(metrics_path)
        mlflow.set_tag("model_version", version)

        print(f"\nSaved model to {args.out} (version {version}) and metrics to {metrics_path}")


if __name__ == "__main__":
    main()

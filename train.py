"""
Train the ticket classifier and track the run in MLflow.

    python train.py                      # experiment: evaluate on validation only
    python train.py --model rf           # pick a model (see MODELS)
    python train.py --final              # also evaluate on test and save the model
    python train.py --note "keep '?'"    # describe what the run tries (MLflow tag)

Experiments never touch the test set or artifacts/model.joblib. Only a --final
run does, once you have picked the model on the validation set.
The vectorizer and model are saved together, so they can never get out of sync.
"""
import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from multiprocessing import Pool

import joblib
import matplotlib
matplotlib.use("Agg")  # no GUI needed, we only save figures
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import models.text_pre_processor as text_pre_processor
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
    "dummy": lambda: DummyClassifier(strategy="most_frequent"
    ),
    "logreg": lambda: LogisticRegression(
        class_weight="balanced", random_state=SEED, max_iter=1000, n_jobs=-1
    ),
    "svc": lambda: LinearSVC(class_weight="balanced", random_state=SEED),
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


def remove_duplicates(df, text):
    """
    Drop texts that appear with more than one label (label noise), then keep one
    copy of every remaining duplicate, so no issue can be in train and validation.
    Texts count as equal when they only differ in case, digits or punctuation.
    """
    key = text.str.lower().str.replace(r"[\W\d_]+", " ", regex=True).str.strip()
    conflicting = df.groupby(key)["issue_label"].transform("nunique") > 1
    duplicate = key.duplicated() & ~conflicting

    keep = ~conflicting & ~duplicate
    stats = {"n_conflicting_removed": int(conflicting.sum()), "n_duplicates_removed": int(duplicate.sum())}
    return df[keep], stats


def preprocessing_version():
    """Short hash of the preprocessing code, so runs and caches can be told apart."""
    with open(text_pre_processor.__file__, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:8]


def preprocess_all(texts, cache_dir):
    """
    Preprocessing is the slow step, so cache it and use all CPU cores.
    The cache file name depends on the input texts and on the preprocessing code,
    so changing either one recomputes it instead of silently reusing old results.
    Returns the processed texts and the preprocessing version.
    """
    code_hash = preprocessing_version()
    data_hash = hashlib.md5(pd.util.hash_pandas_object(texts).values.tobytes()).hexdigest()[:8]
    cache_path = os.path.join(cache_dir, f"processed_{data_hash}_{code_hash}.pkl")

    if os.path.exists(cache_path):
        print(f"Using cached preprocessed text from {cache_path}")
        return pd.read_pickle(cache_path), code_hash

    print(f"Preprocessing {len(texts)} issues (this takes a few minutes)...")
    with Pool() as pool:
        processed = pool.map(preprocess_text, texts.tolist(), chunksize=500)
    processed = pd.Series(processed, index=texts.index)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    processed.to_pickle(cache_path)
    return processed, code_hash


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
    parser.add_argument("--cache-dir", default="datasets/cache")
    parser.add_argument("--note", default="",
                        help='What this run tries, e.g. --note "keep question words". Logged as a tag.')
    args = parser.parse_args()

    df = load_dataset(args.data)
    text = (df["issue_title"].fillna("") + " " + df["issue_body"].fillna("")).str.strip()
    df["processed_text"], preprocessing_version = preprocess_all(text, args.cache_dir)

    df, data_stats = remove_duplicates(df, text)
    print(f"Removed {data_stats['n_conflicting_removed']} issues with conflicting labels "
          f"and {data_stats['n_duplicates_removed']} duplicates, {len(df)} left")

    # 70% train / 15% validation / 15% test, stratified so "question" is present in every split.
    # Split before the language filter, so changing the filter never moves issues between splits.
    train_df, rest_df = train_test_split(df, test_size=0.3, random_state=SEED, stratify=df["issue_label"])
    val_df, test_df = train_test_split(rest_df, test_size=0.5, random_state=SEED, stratify=rest_df["issue_label"])

    def usable(split_df):
        """Drop issues the preprocessing rejected (non-English or empty)."""
        split_df = split_df[split_df["processed_text"].notna()]
        return split_df["processed_text"], split_df["issue_label"]

    X_train, y_train = usable(train_df)
    X_val, y_val = usable(val_df)
    X_test, y_test = usable(test_df)
    data_stats["n_filtered"] = len(df) - len(X_train) - len(X_val) - len(X_test)
    print(f"Language/empty filter removed {data_stats['n_filtered']} issues")

    mlflow.set_experiment(EXPERIMENT_NAME)
    run_name = f"{args.model}-final" if args.final else args.model
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("final", args.final)
        if args.note:
            mlflow.set_tag("note", args.note)
        mlflow.log_params({
            "model": args.model,
            "max_features": args.max_features,
            "seed": SEED,
            "preprocessing_version": preprocessing_version,
            "n_train": len(X_train),
            "n_val": len(X_val),
            **data_stats,
        })

        pipeline = build_pipeline(args.model, args.max_features)
        print(f"Training {args.model} on {len(X_train)} issues...")
        start_time = time.perf_counter()
        pipeline.fit(X_train, y_train)
        train_seconds = time.perf_counter() - start_time
        mlflow.log_metric("training_time", train_seconds)

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

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
from datetime import UTC, datetime
from multiprocessing import Pool

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")  # no GUI needed, we only save figures
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from models import text_pre_processor
from models.model_utils import predict_with_threshold
from models.text_pre_processor import preprocess_text

REQUIRED_COLUMNS = {"issue_label", "issue_title", "issue_body"}
LABELS = ["bug", "enhancement", "question"]
SEED = 42
EXPERIMENT_NAME = "issue-classifier"
QUALITY_GATE = {"macro_f1": 0.70, "question_f1": 0.45}

# Model name -> function returning a fresh classifier. Add baselines here.
MODELS = {
    "rf": lambda: RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=SEED, n_jobs=-1
    ),
    "dummy": lambda: DummyClassifier(strategy="most_frequent"),
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


def tune_question_threshold(probabilities, classes, y_true):
    """
    Try different thresholds for predicting "question" and pick the one with the best macro F1.
    Returns the best threshold and the corresponding F1 score, and the full curve for plotting.
    Use out-of-fold probabilities, never validation probabilities,
    to avoid overfitting the threshold to the validation set.
    """

    thresholds = np.linspace(0.2, 0.8, 61)  # 0.2, 0.21, ..., 0.8
    curve = {"threshold": [], "macro_f1": [], "question_recall": [], "question_precision": []}
    for t in thresholds:
        y_pred = predict_with_threshold(probabilities, classes, t)
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        curve["threshold"].append(t)
        curve["macro_f1"].append(report["macro avg"]["f1-score"])
        curve["question_recall"].append(report["question"]["recall"])
        curve["question_precision"].append(report["question"]["precision"])

    best = int(np.argmax(curve["macro_f1"]))
    return float(thresholds[best]), curve


def check_quality_gate(report, gate=QUALITY_GATE):
    """
    Compare a classification_report dict against minimum scores.
    Returns a list of failure messages; an empty list means the model can be saved.
    """
    scores = {
        "macro_f1": report["macro avg"]["f1-score"],
        "question_f1": report["question"]["f1-score"],
    }
    failures = []
    for name, minimum in gate.items():
        score = scores[name]
        if score < minimum:
            failures.append(f"{name} {score:.3f} < {minimum:.2f}")
    return failures


def dedup_key(text):
    """Normalize texts so near-duplicates (case, digits, punctuation) compare equal."""
    return text.str.lower().str.replace(r"[\W\d_]+", " ", regex=True).str.strip()


def remove_duplicates(df, text):
    """
    Drop texts that appear with more than one label (label noise), then keep one
    copy of every remaining duplicate, so no issue can be in train and validation.
    Texts count as equal when they only differ in case, digits or punctuation.
    """
    key = dedup_key(text)
    conflicting = df.groupby(key)["issue_label"].transform("nunique") > 1
    duplicate = key.duplicated() & ~conflicting

    keep = ~conflicting & ~duplicate
    stats = {
        "n_conflicting_removed": int(conflicting.sum()),
        "n_duplicates_removed": int(duplicate.sum()),
    }
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


def build_pipeline(model_name, max_features, ngram_max):
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=(1, ngram_max))),
            ("clf", MODELS[model_name]()),
        ]
    )


def evaluate(pipeline, X, y, split, question_threshold=None):
    """Print the report, log the headline metrics and a confusion matrix to MLflow."""
    if question_threshold is None:
        y_pred = pipeline.predict(X)
    else:
        y_pred = predict_with_threshold(
            pipeline.predict_proba(X), pipeline.classes_, question_threshold
        )
    print(f"\n{split} set\n" + classification_report(y, y_pred, digits=3, zero_division=0))
    report = classification_report(y, y_pred, output_dict=True, zero_division=0)

    mlflow.log_metrics(
        {
            f"{split}_macro_f1": report["macro avg"]["f1-score"],
            f"{split}_accuracy": report["accuracy"],
            f"{split}_question_recall": report["question"]["recall"],
            f"{split}_question_f1": report["question"]["f1-score"],
        }
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y, y_pred, labels=LABELS, ax=ax, colorbar=False)
    ax.set_title(f"{split} confusion matrix")
    mlflow.log_figure(fig, f"confusion_matrix_{split}.png")
    plt.close(fig)
    return report


def plot_threshold_curve(curve, chosen):
    """Log how macro-F1 and question precision/recall change with the threshold."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(curve["threshold"], curve["macro_f1"], label="macro-F1")
    ax.plot(curve["threshold"], curve["question_precision"], label="question precision")
    ax.plot(curve["threshold"], curve["question_recall"], label="question recall")
    ax.axvline(chosen, color="gray", linestyle="--", label=f"chosen t = {chosen:.2f}")
    ax.set_xlabel("question threshold")
    ax.set_ylabel("score (train, out-of-fold)")
    ax.legend()
    mlflow.log_figure(fig, "question_threshold_curve.png")
    plt.close(fig)


def load_splits(data_path, cache_dir):
    """
    Load, preprocess, deduplicate and split the data, exactly as training does.
    Returns train, validation and test DataFrames (usable issues only), and data_stats.
    """
    df = load_dataset(data_path)
    text = (df["issue_title"].fillna("") + " " + df["issue_body"].fillna("")).str.strip()
    df["processed_text"], preprocessing_version = preprocess_all(text, cache_dir)

    df, data_stats = remove_duplicates(df, text)
    print(
        f"Removed {data_stats['n_conflicting_removed']} issues with conflicting labels "
        f"and {data_stats['n_duplicates_removed']} duplicates, {len(df)} left"
    )

    # 70% train / 15% validation / 15% test, stratified so "question" is present in every split.
    # Split before the language filter, so changing the filter never moves issues between splits.
    train_df, rest_df = train_test_split(
        df, test_size=0.3, random_state=SEED, stratify=df["issue_label"]
    )
    val_df, test_df = train_test_split(
        rest_df, test_size=0.5, random_state=SEED, stratify=rest_df["issue_label"]
    )

    def usable(split_df):
        """Drop issues the preprocessing rejected (non-English or empty)."""
        return split_df[split_df["processed_text"].notna()]

    train_df, val_df, test_df = usable(train_df), usable(val_df), usable(test_df)
    data_stats["n_filtered"] = len(df) - len(train_df) - len(val_df) - len(test_df)
    print(f"Language/empty filter removed {data_stats['n_filtered']} issues")

    return train_df, val_df, test_df, data_stats, preprocessing_version


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default="logreg", choices=sorted(MODELS))
    parser.add_argument("--max-features", type=int, default=100000)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument(
        "--final",
        action="store_true",
        help="Evaluate on the test set and save the model. Use once, at the end.",
    )
    parser.add_argument("--data", default="datasets/sample1.csv")
    parser.add_argument("--out", default="artifacts/model.joblib")
    parser.add_argument(
        "--tune-threshold",
        action="store_true",
        help="Tune the 'question' threshold with 5-fold CV on train (needs predict_proba, e.g. logreg).",
    )
    parser.add_argument("--cache-dir", default="datasets/cache")
    parser.add_argument(
        "--note",
        default="",
        help='What this run tries, e.g. --note "keep question words". Logged as a tag.',
    )
    args = parser.parse_args()

    train_df, val_df, test_df, data_stats, preprocessing_version = load_splits(
        args.data, args.cache_dir
    )
    X_train = train_df["processed_text"]
    y_train = train_df["issue_label"]
    X_val = val_df["processed_text"]
    y_val = val_df["issue_label"]
    X_test = test_df["processed_text"]
    y_test = test_df["issue_label"]

    mlflow.set_experiment(EXPERIMENT_NAME)
    run_name = f"{args.model}-ng{args.ngram_max}-{args.max_features // 1000}k"
    if args.final:
        run_name += "-final"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("final", args.final)
        if args.note:
            mlflow.set_tag("note", args.note)
        mlflow.log_params(
            {
                "model": args.model,
                "max_features": args.max_features,
                "seed": SEED,
                "preprocessing_version": preprocessing_version,
                "n_train": len(X_train),
                "n_val": len(X_val),
                "ngram_max": args.ngram_max,
                **data_stats,
            }
        )

        pipeline = build_pipeline(args.model, args.max_features, args.ngram_max)
        print(f"Training {args.model} on {len(X_train)} issues...")
        start_time = time.perf_counter()
        pipeline.fit(X_train, y_train)
        train_seconds = time.perf_counter() - start_time
        mlflow.log_metric("training_time", train_seconds)

        question_threshold = None  # None means plain argmax, no special threshold for "question"
        if args.tune_threshold:
            print("Tuning the question threshold with 5-fold cross-validation on train...")
            oof_probabilities = cross_val_predict(
                build_pipeline(args.model, args.max_features, args.ngram_max),
                X_train,
                y_train,
                cv=5,
                method="predict_proba",
            )
            question_threshold, curve = tune_question_threshold(
                oof_probabilities, pipeline.classes_, y_train
            )
            plot_threshold_curve(curve, question_threshold)

            argmax_f1 = classification_report(
                y_train,
                predict_with_threshold(oof_probabilities, pipeline.classes_, None),
                output_dict=True,
                zero_division=0,
            )["macro avg"]["f1-score"]
            print(
                f"Out-of-fold macro-F1: argmax {argmax_f1:.3f} -> "
                f"threshold {question_threshold:.2f}: {max(curve['macro_f1']):.3f}"
            )
            mlflow.log_metrics(
                {
                    "train_oof_macro_f1_argmax": argmax_f1,
                    "train_oof_macro_f1_tuned": max(curve["macro_f1"]),
                }
            )
        mlflow.log_param("question_threshold", question_threshold)

        metrics = {"validation": evaluate(pipeline, X_val, y_val, "val", question_threshold)}

        if not args.final:
            print(
                "\nExperiment run: test set not evaluated, model not saved. "
                "Compare runs with `mlflow ui --port 5001`."
            )
            return

        metrics["test"] = evaluate(pipeline, X_test, y_test, "test", question_threshold)

        # Check before saving, so a model below the minimum never overwrites the current one
        check_failures = check_quality_gate(metrics["test"])
        mlflow.set_tag("quality_gate", "failed" if check_failures else "passed")
        if check_failures:
            raise SystemExit(
                "Quality gate failed, model not saved:\n  " + "\n  ".join(check_failures)
            )

        version = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        joblib.dump(
            {"pipeline": pipeline, "version": version, "question_threshold": question_threshold},
            args.out,
            compress=3,
        )

        metrics_path = os.path.join(os.path.dirname(args.out), "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(
                {
                    "model_version": version,
                    "model": args.model,
                    "data": args.data,
                    "ngram_max": args.ngram_max,
                    "question_threshold": question_threshold,
                    "n_train": len(X_train),
                    "n_val": len(X_val),
                    "n_test": len(X_test),
                    **metrics,
                    **vars(args),
                },
                f,
                indent=2,
            )
        mlflow.log_artifact(metrics_path)
        mlflow.set_tag("model_version", version)

        print(f"\nSaved model to {args.out} (version {version}) and metrics to {metrics_path}")


if __name__ == "__main__":
    main()

import os

import joblib
import numpy as np

DEFAULT_MODEL_PATH = "artifacts/model.joblib"


def load_model():
    """
    Load the trained pipeline (TF-IDF + classifier) produced by train.py.
    Returns:
        pipeline: sklearn Pipeline taking preprocessed text.
        version: the model version string saved at training time.
        question_threshold: the threshold for predicting "question" saved at training time, or None if not used.
    """
    model_path = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
    if not os.path.exists(model_path):
        raise RuntimeError(
            f"No trained model found at {model_path}. "
            "Run `python train.py` first, or set MODEL_PATH."
        )

    try:
        bundle = joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(f"Error loading model from {model_path}: {e}") from e
    return bundle["pipeline"], bundle["version"], bundle.get("question_threshold")


def predict_with_threshold(probabilities, classes, question_threshold=None):
    """
    Turn predicted probabilities into labels, with a tunable threshold for "question".
    Used by both train.py and app.py, so training and serving decide the same way.
    Args:
        probabilities: array of shape (n_issues, n_classes), e.g. pipeline.predict_proba(X).
        classes: labels in the same order as the columns, e.g. pipeline.classes_.
        question_threshold: predict "question" when its probability is at least this,
            otherwise the most likely other class. None means plain argmax.
    Returns:
        list of predicted labels.
    """
    probabilities = np.asarray(probabilities)
    classes = np.asarray(classes)
    if question_threshold is None:
        return classes[probabilities.argmax(axis=1)].tolist()

    q = list(classes).index("question")
    others = probabilities.copy()
    others[:, q] = -1  # exclude "question" when picking the fallback label
    predictions = classes[others.argmax(axis=1)]
    predictions[probabilities[:, q] >= question_threshold] = "question"
    return predictions.tolist()

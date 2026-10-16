import os

import joblib

DEFAULT_MODEL_PATH = "artifacts/model.joblib"


def load_model():
    """
    Load the trained pipeline (TF-IDF + classifier) produced by train.py.
    Returns:
        pipeline: sklearn Pipeline taking preprocessed text.
        version: the model version string saved at training time.
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
        raise RuntimeError(f"Error loading model from {model_path}: {e}")
    return bundle["pipeline"], bundle["version"]

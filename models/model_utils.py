import joblib
import os

def load_model_and_vectorizer():
    """
    Load the pre-trained model and vectorizer from disk.
    Returns:
        vectorizer: The vectorizer used for text processing.
        model: The trained classification model.
    """
    model_candidates = [
        os.getenv("MODEL_PATH"),
        "artifacts/current/model.pkl",
        "trained_models/model2.pkl",
        "model.pkl",
    ]
    vectorizer_candidates = [
        os.getenv("VECTORIZER_PATH"),
        "artifacts/current/vectorizer.pkl",
        "vectorizers/vectorizer10k2.pkl",
        "vectorizer.pkl",
    ]

    model_path = next((p for p in model_candidates if p and os.path.exists(p)), None)
    vectorizer_path = next((p for p in vectorizer_candidates if p and os.path.exists(p)), None)

    if not model_path or not vectorizer_path:
        raise RuntimeError(
            "Could not find model/vectorizer artifacts. "
            "Set MODEL_PATH and VECTORIZER_PATH, or place files in one of: "
            "artifacts/current/, trained_models+vectorizers, or project root."
        )

    try:
        vectorizer = joblib.load(vectorizer_path)
        model = joblib.load(model_path)
        return vectorizer, model
    except Exception as e:
        raise RuntimeError(f"Error loading model or vectorizer: {e}")

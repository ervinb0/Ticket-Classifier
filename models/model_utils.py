import joblib
from .text_pre_processor import preprocess_text

def load_model_and_vectorizer():
    """
    Load the pre-trained model and vectorizer from disk.
    Returns:
        vectorizer: The vectorizer used for text processing.
        model: The trained classification model.
    """
    try:
        vectorizer = joblib.load("/Users/ervinballa/Desktop/P_AI_Eng/ws2024-principles-of-ai-engineering/vectorizers/vectorizer10k2.pkl")
        model = joblib.load("/Users/ervinballa/Desktop/P_AI_Eng/ws2024-principles-of-ai-engineering/trained_models/model2.pkl")
        return vectorizer, model
    except Exception as e:
        raise RuntimeError(f"Error loading model or vectorizer: {e}")

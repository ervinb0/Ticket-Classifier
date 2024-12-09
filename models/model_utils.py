import joblib
import os
vectorizer_path = os.getenv("VECTORIZER_PATH", "vectorizers/vectorizer10k2.pkl")
model_path = os.getenv("MODEL_PATH", "trained_models/model2.pkl")

#from .text_pre_processor import preprocess_text

#model path = "/Users/ervinballa/Desktop/P_AI_Eng/ws2024-principles-of-ai-engineering/trained_models/model2.pkl"
#vector path = "/Users/ervinballa/Desktop/P_AI_Eng/ws2024-principles-of-ai-engineering/vectorizers/vectorizer10k2.pkl"
def load_model_and_vectorizer():
    """
    Load the pre-trained model and vectorizer from disk.
    Returns:
        vectorizer: The vectorizer used for text processing.
        model: The trained classification model.
    """
    try:
        vectorizer = joblib.load(vectorizer_path)
        model = joblib.load(model_path)
        return vectorizer, model
    except Exception as e:
        raise RuntimeError(f"Error loading model or vectorizer: {e}")

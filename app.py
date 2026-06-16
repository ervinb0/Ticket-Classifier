from flask import Flask, request, jsonify, g
import models.model_utils as model_utils
from models.crud import store_prediction, update_prediction
from models.database import SessionLocal, engine
from models.models import Base
from models.text_pre_processor import preprocess_text

app = Flask(__name__)

_vectorizer = None
_model = None


def get_model_and_vectorizer():
    """Load once on first use so importing the app does not require .pkl files."""
    global _vectorizer, _model
    if _vectorizer is None:
        _vectorizer, _model = model_utils.load_model_and_vectorizer()
    return _vectorizer, _model

# Ensure DB tables exist
Base.metadata.create_all(bind=engine)

# Label mappings
LABEL_MAPPING = {0: "bug", 1: "enhancement", 2: "question"}
INVERSE_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}

def get_db():
    if "db" not in g:
        g.db = SessionLocal()
    return g.db

@app.teardown_appcontext
def close_db(error=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    title = data.get('title', '').strip()
    body = data.get('body', '').strip()
    
    if not title and not body:
        return jsonify({"error": "Title and body cannot both be empty."}), 400
    
    issue_text = title + " " + body
    pre_text = preprocess_text(issue_text)
    if pre_text is None:
        return jsonify({"error": "Input is not in English or is invalid."}), 400

    vectorizer, model = get_model_and_vectorizer()
    vectorized_text = vectorizer.transform([pre_text])
    predicted_label = model.predict(vectorized_text)[0]
    probabilities = model.predict_proba(vectorized_text)[0]
    confidence = max(probabilities)
    labeled_prediction = LABEL_MAPPING[predicted_label]
    issue_id = store_prediction(get_db(), issue_text, labeled_prediction)
    
    return jsonify({
        "id": issue_id,
        "title": title,
        "body": body,
        "predicted_label": labeled_prediction,
        "prediction": labeled_prediction,
        "confidence": float(confidence),
        "probabilities": {
            LABEL_MAPPING[i]: float(prob) for i, prob in enumerate(probabilities)
        }
    })

@app.route('/api/correct', methods=['POST'])
def correct():
    data = request.json
    issue_id = data.get("id", "").strip()
    corrected_label = data.get("corrected_label", "").strip()
    
    if corrected_label not in INVERSE_LABEL_MAPPING:
        return jsonify({'error': f'Invalid label. Valid labels are: {list(INVERSE_LABEL_MAPPING.keys())}'}), 400

    if not issue_id:
        return jsonify({"error": "Missing prediction id."}), 400

    updated = update_prediction(get_db(), issue_id, corrected_label)
    if not updated:
        return jsonify({"error": "Prediction id not found."}), 404

    return jsonify({"message": "Label corrected successfully", "id": issue_id, "corrected_label": corrected_label})

if __name__ == '__main__':
    app.run(debug=True)

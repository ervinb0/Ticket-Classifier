import os

from flask import Flask, request, jsonify, g, send_from_directory
import models.model_utils as model_utils
from models.crud import store_prediction, update_prediction
from models.database import SessionLocal, engine
from models.models import Base
from models.text_pre_processor import preprocess_text

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")

app = Flask(__name__, static_folder=os.path.join(FRONTEND_DIR, "static"))

_model = None
_model_version = None


def get_model():
    """Load once on first use so importing the app does not require a trained model."""
    global _model, _model_version
    if _model is None:
        _model, _model_version = model_utils.load_model()
    return _model, _model_version

# Ensure DB tables exist
Base.metadata.create_all(bind=engine)

VALID_LABELS = ["bug", "enhancement", "question"]

def get_db():
    if "db" not in g:
        g.db = SessionLocal()
    return g.db

@app.teardown_appcontext
def close_db(error=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()

def get_string(data, key):
    value = data.get(key, "")
    return value.strip() if isinstance(value, str) else None

@app.route('/')
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route('/api/health')
def health():
    return jsonify({"status": "ok"})

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    title = get_string(data, 'title')
    body = get_string(data, 'body')
    if title is None or body is None:
        return jsonify({"error": "Title and body must be strings."}), 400
    if not title and not body:
        return jsonify({"error": "Title and body cannot both be empty."}), 400

    issue_text = title + " " + body
    pre_text = preprocess_text(issue_text)
    if pre_text is None:
        return jsonify({"error": "Input is not in English or is invalid."}), 400

    model, model_version = get_model()
    probabilities = model.predict_proba([pre_text])[0]
    class_probs = {label: float(p) for label, p in zip(model.classes_, probabilities)}
    predicted_label = max(class_probs, key=class_probs.get)
    issue_id = store_prediction(get_db(), issue_text, predicted_label)

    return jsonify({
        "id": issue_id,
        "title": title,
        "body": body,
        "predicted_label": predicted_label,
        "confidence": class_probs[predicted_label],
        "probabilities": class_probs,
        "model_version": model_version,
    })

@app.route('/api/correct', methods=['POST'])
def correct():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    issue_id = get_string(data, "id")
    corrected_label = get_string(data, "corrected_label")

    if corrected_label not in VALID_LABELS:
        return jsonify({'error': f'Invalid label. Valid labels are: {VALID_LABELS}'}), 400

    if not issue_id:
        return jsonify({"error": "Missing prediction id."}), 400

    updated = update_prediction(get_db(), issue_id, corrected_label)
    if not updated:
        return jsonify({"error": "Prediction id not found."}), 404

    return jsonify({"message": "Label corrected successfully", "id": issue_id, "corrected_label": corrected_label})

if __name__ == '__main__':
    # Not 5000: macOS AirPlay Receiver listens there
    app.run(port=int(os.getenv("PORT", "8000")), debug=os.getenv("FLASK_DEBUG") == "1")

# app.py
from flask import Flask, request, jsonify, g
from models.text_pre_processor import preprocess_text
from models.database import SessionLocal
from models.crud import store_prediction, update_prediction
from models.model_utils import load_model_and_vectorizer
from models.text_pre_processor import preprocess_text
import sqlite3
import joblib
from langdetect import detect, DetectorFactory
from flask_cors import CORS

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Label mappings
LABEL_MAPPING = {0: "bug", 1: "enhancement", 2: "question"}
INVERSE_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}

def get_db():
    if not hasattr(g, 'db'):
        g.db = sqlite3.connect('database.db')
    return g.db

def store_corrected_data(db, title, body, predicted_label, corrected_label, corrected):
    cursor = db.cursor()
    cursor.execute("""
        INSERT INTO issues (title, body, predicted_label, corrected_label, corrected)
        VALUES (?, ?, ?, ?, ?)""",
        (title, body, predicted_label, corrected_label, corrected)
    )
    db.commit()
    return cursor.lastrowid

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
    
    vectorized_text = vectorizer.transform([pre_text])
    predicted_label = model.predict(vectorized_text)[0]
    probabilities = model.predict_proba(vectorized_text)[0]
    confidence = max(probabilities)
    labeled_prediction = LABEL_MAPPING[predicted_label]
    
    return jsonify({
        "title": title,
        "body": body,
        "prediction": labeled_prediction,
        "confidence": float(confidence),
        "probabilities": {
            LABEL_MAPPING[i]: float(prob) for i, prob in enumerate(probabilities)
        }
    })

@app.route('/api/correct', methods=['POST'])
def correct():
    data = request.json
    title = data.get("title", "").strip()
    body = data.get("body", "").strip()
    predicted_label = data.get("predicted_label", "").strip()
    corrected_label = data.get("corrected_label", "").strip()
    
    if corrected_label not in INVERSE_LABEL_MAPPING:
        return jsonify({'error': f'Invalid label. Valid labels are: {list(INVERSE_LABEL_MAPPING.keys())}'}), 400
    
    corrected = 1 if predicted_label != corrected_label else 0
    db = get_db()
    issue_id = store_corrected_data(db, title, body, predicted_label, corrected_label, corrected)
    
    return jsonify({"message": "Label corrected successfully", "id": issue_id})

if __name__ == '__main__':
    app.run(debug=True)

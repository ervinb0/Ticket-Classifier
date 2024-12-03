from flask import Flask, request, jsonify, g
from models.database import SessionLocal
from models.crud import store_prediction, update_prediction
from models.model_utils import load_model_and_vectorizer
from models.text_pre_processor import preprocess_text
from langdetect import detect, DetectorFactory

app = Flask(__name__)

DetectorFactory.seed = 0
# Mapping of prediction integers to labels
LABEL_MAPPING = {0: "bug", 1: "enhancement", 2: "question"}

# Create the inverse of LABEL_MAPPING
INVERSE_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}

# Load model and vectorizer
vectorizer, model = load_model_and_vectorizer()

# Create a new database session for each request
@app.before_request
def create_db_session():
    g.db = SessionLocal()

@app.teardown_request
def close_db_session(exception=None):
    db = getattr(g, 'db', None)
    if db is not None:
        db.close()

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    title = data.get('title', '')
    body = data.get('body', '')
    
    # Combine the title and body to form the full issue text
    issue_text = title + " " + body
    
    if not issue_text:
        return jsonify({"error": "No text provided"}), 400

    # Check if text is in English
    try:
        language = detect(issue_text)
        if language != 'en':
            return jsonify({"error": "Input text must be in English."}), 400
    except Exception as e:
        return jsonify({"error": "Could not detect language. Please try again."}), 400
    
    # Preprocess and predict
    preprocessed_text = preprocess_text(issue_text)
    if not preprocessed_text:
        return jsonify({"error": "Could not detect meaningful text from your input. Please try again."}), 400
    
    vectorized_text = vectorizer.transform([preprocessed_text])
    predicted_label = model.predict(vectorized_text)[0]

    # Store the prediction in the database
    issue_id = store_prediction(g.db, issue_text, predicted_label)
    # Example usage in your API
    labeled_prediction = LABEL_MAPPING[predicted_label]

    return jsonify({"id": issue_id, "predicted_label": labeled_prediction})

@app.route('/api/correct', methods=['POST'])
def correct():
    data = request.json
    issue_id = data.get("id", "")
    corrected_label = data.get("corrected_label", "")

    if not issue_id or not corrected_label:
        return jsonify({"error": "ID and corrected label are required"}), 400
    
    if corrected_label not in INVERSE_LABEL_MAPPING:
         return jsonify({'error': f'Invalid label. Valid labels are: {list(INVERSE_LABEL_MAPPING.keys())}'}), 400
     
     # Convert the correct label to its numerical value
    corrected_label_num = INVERSE_LABEL_MAPPING[corrected_label]

    # Update the corrected label in the database
    success = update_prediction(g.db, issue_id, corrected_label_num)

    if not success:
        return jsonify({"error": "Prediction ID not found"}), 404

    return jsonify({"message": "Label corrected successfully", "id": issue_id})


if __name__ == '__main__':
    app.run(debug=True)
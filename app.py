from flask import Flask, request, jsonify
import pickle
import uuid
import os
from datetime import datetime
import logging

app = Flask(__name__)

# Setup logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load the trained model
MODEL_PATH = 'models/dummy_model.pkl'

def load_model(model_path):
    """Load the trained model from pickle file"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logging.info("Model loaded successfully")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        return None

# Load model at startup
model = load_model(MODEL_PATH)

# In-memory store for issues
issues_db = {}

@app.route('/api/predict', methods=['POST'])
def predict_issue():
    """Endpoint to predict issue type based on title and body"""
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data or 'title' not in data or 'body' not in data:
            return jsonify({
                'error': 'Missing required fields: title and body'
            }), 400
        
        # Combine title and body for prediction
        text = f"{data['title']} {data['body']}"
        
        # Generate prediction using the model
        if model is not None:
            # Make prediction
            prediction = model.predict([text])[0]
            
            # Generate issue ID
            issue_id = str(uuid.uuid4())
            
            # Store in our database
            issues_db[issue_id] = {
                'title': data['title'],
                'body': data['body'],
                'label': prediction,
                'timestamp': datetime.now().isoformat()
            }
            
            response = {
                'id': issue_id,
                'label': prediction
            }
            
            logging.info(f"Prediction made for issue {issue_id}: {prediction}")
            return jsonify(response)
        else:
            return jsonify({
                'error': 'Model not loaded properly'
            }), 500
            
    except Exception as e:
        logging.error(f"Error in predict_issue: {str(e)}")
        return jsonify({
            'error': 'Internal server error'
        }), 500

@app.route('/api/correct', methods=['POST'])
def correct_issue():
    """Endpoint to correct the predicted issue type"""
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data or 'id' not in data or 'label' not in data:
            return jsonify({
                'error': 'Missing required fields: id and label'
            }), 400
        
        # Check if issue exists
        if data['id'] not in issues_db:
            return jsonify({
                'error': 'Issue not found'
            }), 404
        
        # Update the label
        old_label = issues_db[data['id']]['label']
        issues_db[data['id']]['label'] = data['label']
        issues_db[data['id']]['corrected_at'] = datetime.now().isoformat()
        
        logging.info(
            f"Label corrected for issue {data['id']}: "
            f"{old_label} -> {data['label']}"
        )
        
        return jsonify({
            'id': data['id']
        })
        
    except Exception as e:
        logging.error(f"Error in correct_issue: {str(e)}")
        return jsonify({
            'error': 'Internal server error'
        }), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint to check if the service is healthy"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    app.run(debug=True)

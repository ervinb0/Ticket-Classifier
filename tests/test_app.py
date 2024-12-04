import pytest
import json
from app import app  # Import your Flask app

# Mapping of prediction integers to labels
LABEL_MAPPING = {"bug": 0, "enhancement": 1, "question": 2}

@pytest.fixture
def client():
    # Create a test client for the Flask app
    with app.test_client() as client:
        yield client

def test_predict(client):
    # Test the predict endpoint
    data = {
        "title": "Bug in authentication",
        "body": "The user is unable to log in after recent updates."
    }

    response = client.post('/api/predict', json=data)

    # Assert status code is 200 (OK)
    assert response.status_code == 200

    # Parse the response JSON
    response_json = json.loads(response.data)

    # Assert that the response has the correct keys and values
    assert "id" in response_json
    assert "predicted_label" in response_json
    assert response_json["predicted_label"] in LABEL_MAPPING.keys()

def test_correct(client):
    # Test the correct endpoint
    prediction_data = {
        "title": "Bug in authentication",
        "body": "The user is unable to log in after recent updates."
    }

    # First, predict a label
    response = client.post('/api/predict', json=prediction_data)
    response_json = json.loads(response.data)
    issue_id = response_json["id"]
    predicted_label = response_json["predicted_label"]

    # Now, correct the prediction
    correction_data = {
        "id": issue_id,
        "corrected_label": "question"  # Correcting to a question (2) just for testing
    }
    x = correction_data["corrected_label"]

    response = client.post('/api/correct', json=correction_data)

    # Assert that the response status code is 200
    assert response.status_code == 200

    # Assert that the correction was successfully stored (Check the response or DB)
    response_json = json.loads(response.data)
    assert response_json["id"] == issue_id
    assert response_json["corrected_label"] == LABEL_MAPPING[x]

import pytest
import json
from app import app  # Import Flask app

VALID_LABELS = {"bug", "enhancement", "question"}

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
    assert response_json["predicted_label"] in VALID_LABELS

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
    # Now, correct the prediction
    correction_data = {
        "id": issue_id,
        "corrected_label": "question"
    }

    response = client.post('/api/correct', json=correction_data)

    # Assert that the response status code is 200
    assert response.status_code == 200

    # Assert that the correction was successfully stored (Check the response or DB)
    response_json = json.loads(response.data)
    assert response_json["id"] == issue_id
    assert response_json["corrected_label"] == correction_data["corrected_label"]

def test_predict_returns_probabilities_for_all_labels(client):
    response = client.post('/api/predict', json={"title": "App crashes", "body": ""})
    assert response.status_code == 200
    response_json = response.get_json()
    assert set(response_json["probabilities"]) == VALID_LABELS
    assert abs(sum(response_json["probabilities"].values()) - 1) < 1e-6
    assert response_json["model_version"] == "test-stub"

def test_predict_rejects_empty_input(client):
    response = client.post('/api/predict', json={"title": " ", "body": ""})
    assert response.status_code == 400

def test_predict_rejects_non_json_body(client):
    response = client.post('/api/predict', data="not json", content_type="text/plain")
    assert response.status_code == 400

def test_predict_rejects_non_string_fields(client):
    response = client.post('/api/predict', json={"title": 123, "body": None})
    assert response.status_code == 400

def test_predict_rejects_non_english(client):
    response = client.post('/api/predict', json={
        "title": "Le bouton ne fonctionne pas",
        "body": "Après la mise à jour, le bouton de connexion ne fonctionne plus du tout."
    })
    assert response.status_code == 400

def test_correct_rejects_invalid_label(client):
    response = client.post('/api/correct', json={"id": "abc", "corrected_label": "feature"})
    assert response.status_code == 400

def test_correct_unknown_id_returns_404(client):
    response = client.post('/api/correct', json={"id": "does-not-exist", "corrected_label": "bug"})
    assert response.status_code == 404

def test_index_serves_frontend(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b"Issue Classifier" in response.data

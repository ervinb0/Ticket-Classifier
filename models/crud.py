from sqlalchemy.orm import Session
from .models import Prediction
import uuid

def store_prediction(db: Session, issue_text: str, predicted_label: str):
    """Store a prediction in the database."""
    issue_id = str(uuid.uuid4())  # Generate a unique ID
    prediction = Prediction(
        id=issue_id, text=issue_text, predicted_label=predicted_label
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    return issue_id

def get_prediction(db: Session, issue_id: str):
    """Retrieve a prediction by ID."""
    return db.query(Prediction).filter(Prediction.id == issue_id).first()

def update_prediction(db: Session, issue_id: str, corrected_label: str):
    """Update the corrected label for a prediction."""
    prediction = db.query(Prediction).filter(Prediction.id == issue_id).first()
    if prediction:
        prediction.corrected_label = corrected_label
        db.commit()
        return True
    return False

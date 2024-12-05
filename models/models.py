from sqlalchemy import Column, String, Text
from models.database import Base

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(String, primary_key=True, index=True)
    text = Column(Text, nullable=False)
    predicted_label = Column(String, nullable=False)
    corrected_label = Column(String, nullable=True)

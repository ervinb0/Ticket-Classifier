from models.database import engine, Base
from models.models import Prediction

# Create all tables defined in models
Base.metadata.create_all(bind=engine)

print("Database and tables created successfully!")

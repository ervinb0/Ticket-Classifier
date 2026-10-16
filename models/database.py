import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# SQLite by default; set DATABASE_URL to use another database
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///predictions.db")

# Create an SQLAlchemy engine
engine = create_engine(DATABASE_URL, echo=os.getenv("SQL_ECHO") == "1")

# Create a session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for our models
Base = declarative_base()

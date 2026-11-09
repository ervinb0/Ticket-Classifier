import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# SQLite by default; set DATABASE_URL to use another database
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///predictions.db")

# Create an SQLAlchemy engine
engine = create_engine(DATABASE_URL, echo=os.getenv("SQL_ECHO") == "1")

# Create a session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for our models
Base = declarative_base()


def init_db():
    """Create missing tables. Run once before the server starts, not on every import."""
    from models import models  # noqa: F401  (registers the tables on Base)

    Base.metadata.create_all(bind=engine)

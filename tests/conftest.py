"""
Pytest loads this file before test modules. API tests use a tiny stub pipeline
(not the real model) and an in-memory database, so they run in CI and fresh
clones without the dataset or a trained model.
"""
import os

os.environ.setdefault("DATABASE_URL", "sqlite://")

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

import models.model_utils as model_utils


def _stub_model():
    texts = [
        "authentication bug login user unable",
        "feature enhancement improve request",
        "question configure help",
        "bug crash error fix",
    ]
    labels = ["bug", "enhancement", "question", "bug"]
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", RandomForestClassifier(n_estimators=5, random_state=42)),
    ])
    pipeline.fit(texts, labels)
    return pipeline, "test-stub"


@pytest.fixture(scope="session", autouse=True)
def stub_model():
    original = model_utils.load_model
    model_utils.load_model = _stub_model
    yield
    model_utils.load_model = original

"""
Pytest loads this file before test modules. We stub model/vectorizer when no
artifacts are on disk so `import app` and API tests work in CI / fresh clones.
"""
import os

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

import models.model_utils as model_utils


def _artifacts_present() -> bool:
    model_candidates = [
        os.getenv("MODEL_PATH"),
        "artifacts/current/model.pkl",
        "trained_models/model2.pkl",
        "model.pkl",
    ]
    vectorizer_candidates = [
        os.getenv("VECTORIZER_PATH"),
        "artifacts/current/vectorizer.pkl",
        "vectorizers/vectorizer10k2.pkl",
        "vectorizer.pkl",
    ]
    has_model = any(p for p in model_candidates if p and os.path.exists(p))
    has_vec = any(p for p in vectorizer_candidates if p and os.path.exists(p))
    return has_model and has_vec


def _stub_vectorizer_and_model():
    v = TfidfVectorizer()
    texts = [
        "authentication bug login user unable",
        "feature enhancement improve request",
        "question how configure help",
        "bug crash error fix",
    ]
    v.fit(texts)
    clf = RandomForestClassifier(n_estimators=5, random_state=42, class_weight="balanced")
    clf.fit(v.transform(texts), [0, 1, 2, 0])
    return v, clf


@pytest.fixture(scope="session", autouse=True)
def stub_model_if_no_artifacts():
    if _artifacts_present():
        yield
        return
    original = model_utils.load_model_and_vectorizer

    def _stub():
        return _stub_vectorizer_and_model()

    model_utils.load_model_and_vectorizer = _stub
    yield
    model_utils.load_model_and_vectorizer = original


import pandas as pd
import models.text_pre_processor as text_pre_processor
from train import remove_duplicates, MODELS, build_pipeline, preprocess_all
import pytest

def test_remove_duplicates():
    # Arrange: create a small DataFrame
    df = pd.DataFrame({
        "issue_text": [
            "App crashes",       # Row 0: keep
            "app crashes!",      # Row 1: duplicate of row 0
            "Payment failed",    # Row 2: keep
            "Payment failed.",   # Row 3: duplicate of row 2
            "Error 404",         # Row 4: conflicting label since processing removes numbers/punctuation
            "error 500",         # Row 5: conflicting label since processing removes numbers/punctuation
            "Login timeout",     # Row 6: unique, keep
        ],
        "issue_label": [
            "Bug",
            "Bug",
            "Question",
            "Question",
            "Bug",
            "Question",
            "Bug",
        ],
    })

    # Act: pass the ENTIRE text column as a pandas Series
    cleaned_df, stats = remove_duplicates(
        df, df["issue_text"]
    )

    # Assert: only the expected rows remain
    assert cleaned_df.index.tolist() == [0, 2, 6]

    # Assert: the conflicting rows were removed
    assert stats["n_conflicting_removed"] == 2

    # Assert: the repeated copies were removed
    assert stats["n_duplicates_removed"] == 2

    # Assert: the correct labels remain
    assert cleaned_df["issue_label"].tolist() == [
        "Bug",
        "Question",
        "Bug",
    ]
    
@pytest.mark.parametrize("name", sorted(MODELS))
def test_model_training(name):
        texts = [
            "Application crashes on startup",
            "Unexpected error when opening ticket",
            "Add dark mode to dashboard",
            "Implement search functionality",
            "How do I configure the application",
            "Why does this feature work this way",
        ]

        labels = [
            "bug",
            "bug",
            "enhancement",
            "enhancement",
            "question",
            "question",
        ]
        pipeline = build_pipeline(name, max_features=100)
        pipeline.fit(texts, labels)
        
        predictions = pipeline.predict(texts)
        
        assert len(predictions) == len(labels)
        assert set(predictions).issubset(set(labels))
    


def test_cache_changes_with_preprocessing_code(tmp_path, monkeypatch):
    # Point the "preprocessing code" at a fake file we can edit
    fake_code = tmp_path / "fake_pre_processor.py"
    fake_code.write_text("version 1")
    monkeypatch.setattr(text_pre_processor, "__file__", str(fake_code))
    cache_dir = tmp_path / "cache"
    texts = pd.Series(["The app crashes on startup", "Add a dark mode"])

    _, version_1 = preprocess_all(texts, str(cache_dir))
    _, version_again = preprocess_all(texts, str(cache_dir))  # same code: cache is reused
    fake_code.write_text("version 2")
    _, version_2 = preprocess_all(texts, str(cache_dir))

    assert version_1 == version_again
    assert version_1 != version_2
    assert len(list(cache_dir.iterdir())) == 2  # one cache file per code version

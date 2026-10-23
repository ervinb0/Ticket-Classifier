import pandas as pd
from train import remove_duplicates, MODELS, build_pipeline
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
    

import pandas as pd
from train import remove_duplicates

def test_remove_duplicates():
    # Arrange: create a small DataFrame
    df = pd.DataFrame({
        "issue_text": [
            "App crashes",       # Row 0: keep
            "app crashes!",      # Row 1: duplicate of row 0
            "Payment failed",    # Row 2: keep
            "Payment failed.",   # Row 3: duplicate of row 2
            "Error 404",         # Row 4: conflicting label
            "error 500",         # Row 5: conflicting label
            "Login timeout",     # Row 6: unique, keep
        ],
        "issue_label": [
            "Bug",
            "Bug",
            "Payment",
            "Payment",
            "Bug",
            "Payment",
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
        "Payment",
        "Bug",
    ]

import itertools

import pandas as pd
import pytest

from models import text_pre_processor
from train import MODELS, build_pipeline, dedup_key, load_splits, preprocess_all, remove_duplicates


def make_dataset(path):
    """Write a small fake dataset with near-duplicates planted in it."""
    rows = [
        {
            "issue_title": "App crashes on startup",
            "issue_body": "See logs at https://example.com/logs",
            "issue_label": "bug",
        },
        {
            "issue_title": "App crashes on startup!",
            "issue_body": "See logs at https://example.com/logs",
            "issue_label": "bug",
        },  # near-duplicate of row 0
        {
            "issue_title": "Add dark mode",
            "issue_body": "Feature request for dark mode",
            "issue_label": "enhancement",
        },
        {
            "issue_title": "Add dark mode",
            "issue_body": "Feature request for dark mode",
            "issue_label": "enhancement",
        },  # exact duplicate of row 2
        {
            "issue_title": "How do I configure the app?",
            "issue_body": "I need help configuring the app",
            "issue_label": "question",
        },
        {
            "issue_title": "How do I configure the app?",
            "issue_body": "I need help configuring the app",
            "issue_label": "question",
        },  # exact duplicate of row 4
        {
            "issue_title": "Payment failed",
            "issue_body": "Error 404 when trying to pay",
            "issue_label": "bug",
        },
        {
            "issue_title": "Payment failed",
            "issue_body": "Error 500 when trying to pay",
            "issue_label": "question",
        },  # conflicting label for the same text
    ]

    # 20 distinct texts per label. They differ in words, not numbers,
    # because dedup_key strips digits and would treat "crash 1" and "crash 2" as one text.
    templates = {
        "bug": (
            "The {} {}",
            ["login page", "search bar", "settings menu", "export button", "upload form"],
            [
                "crashes when clicked",
                "freezes after saving",
                "shows a blank screen",
                "throws an error on submit",
            ],
        ),
        "enhancement": (
            "Please add {} to {}",
            ["dark mode", "keyboard shortcuts", "email alerts", "drag and drop"],
            [
                "the dashboard",
                "the editor",
                "the mobile app",
                "the settings page",
                "the admin panel",
            ],
        ),
        "question": (
            "{} {}?",
            ["How do I", "Is it possible to", "What is the best way to", "Why can't I"],
            [
                "reset my password",
                "change the default language",
                "connect a database",
                "invite new users",
                "delete an old project",
            ],
        ),
    }
    generated = []
    for label, (pattern, first, second) in templates.items():
        for a, b in itertools.product(first, second):
            generated.append(
                {
                    "issue_title": pattern.format(a, b),
                    "issue_body": "More details about this issue are in the description below.",
                    "issue_label": label,
                }
            )
    rows += generated

    # Near-duplicates: the same text with other case and extra punctuation.
    # These are what would leak between splits if deduplication ran after splitting.
    for row in generated[::4]:
        rows.append({**row, "issue_title": row["issue_title"].upper() + "!!"})

    pd.DataFrame(rows).to_csv(path, index=False)


def test_splits_do_not_overlap(tmp_path):
    data = tmp_path / "issues.csv"
    make_dataset(data)

    train_df, val_df, test_df, _, _ = load_splits(str(data), str(tmp_path / "cache"))
    splits = {"train": train_df, "val": val_df, "test": test_df}

    # 1. no shared rows
    for (name_a, a), (name_b, b) in itertools.combinations(splits.items(), 2):
        overlap = set(a.index) & set(b.index)
        assert not overlap, f"{name_a} and {name_b} share {len(overlap)} rows"

    # 2. no shared texts after normalization (catches near-duplicates, not just the same row)
    def keys(df):
        text = (df["issue_title"].fillna("") + " " + df["issue_body"].fillna("")).str.strip()
        return set(dedup_key(text))

    for (name_a, a), (name_b, b) in itertools.combinations(splits.items(), 2):
        overlap = keys(a) & keys(b)
        assert not overlap, f"{name_a} and {name_b} share texts: {sorted(overlap)[:3]}"

    # 3. every split has every label (stratification works)
    for name, df in splits.items():
        assert set(df["issue_label"]) == {"bug", "enhancement", "question"}, name


def test_splits_are_reproducible(tmp_path):
    data = tmp_path / "issues.csv"
    make_dataset(data)

    first = load_splits(str(data), str(tmp_path / "cache"))
    second = load_splits(str(data), str(tmp_path / "cache"))

    for split_first, split_second in zip(first[:3], second[:3], strict=True):
        assert split_first.index.tolist() == split_second.index.tolist()


def test_remove_duplicates():
    # Arrange: create a small DataFrame
    df = pd.DataFrame(
        {
            "issue_text": [
                "App crashes",  # Row 0: keep
                "app crashes!",  # Row 1: duplicate of row 0
                "Payment failed",  # Row 2: keep
                "Payment failed.",  # Row 3: duplicate of row 2
                "Error 404",  # Row 4: conflicting label since processing removes numbers/punctuation
                "error 500",  # Row 5: conflicting label since processing removes numbers/punctuation
                "Login timeout",  # Row 6: unique, keep
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
        }
    )

    # Act: pass the ENTIRE text column as a pandas Series
    cleaned_df, stats = remove_duplicates(df, df["issue_text"])

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
    pipeline = build_pipeline(name, max_features=100, ngram_max=1)
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

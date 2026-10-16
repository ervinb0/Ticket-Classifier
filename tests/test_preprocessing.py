from models.text_pre_processor import preprocess_text


def test_removes_links_and_numbers():
    result = preprocess_text("Crash on startup, see https://github.com/org/repo/issues/42 for logs 123")
    assert "http" not in result
    assert "github" not in result
    assert "123" not in result
    assert "crash" in result


def test_lemmatizes_and_removes_stopwords():
    result = preprocess_text("The buttons are not working")
    assert result.split() == ["button", "working"]


def test_short_english_text_is_kept():
    # langdetect misclassifies very short texts, so they must not be rejected
    assert preprocess_text("add dark mode") == "add dark mode"


def test_non_english_text_is_rejected():
    assert preprocess_text("Le bouton de connexion ne fonctionne plus après la mise à jour") is None


def test_empty_after_cleaning_returns_none():
    assert preprocess_text("12345 !!! ???") is None

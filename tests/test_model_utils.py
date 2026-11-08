import pytest

from models.model_utils import predict_with_threshold

CLASSES = ["bug", "enhancement", "question"]


def test_no_threshold_is_argmax():
    probs = [[0.6, 0.3, 0.1], [0.2, 0.3, 0.5]]
    assert predict_with_threshold(probs, CLASSES, None) == ["bug", "question"]


def test_question_above_threshold_wins_even_if_not_most_likely():
    probs = [[0.40, 0.25, 0.35]]
    assert predict_with_threshold(probs, CLASSES, 0.30) == ["question"]


def test_question_below_threshold_falls_back_to_best_other_class():
    # "question" is the most likely class, but below the threshold
    probs = [[0.25, 0.30, 0.45]]
    assert predict_with_threshold(probs, CLASSES, 0.50) == ["enhancement"]


def test_column_order_follows_classes():
    probs = [[0.35, 0.40, 0.25]]  # question, bug, enhancement
    assert predict_with_threshold(probs, ["question", "bug", "enhancement"], 0.30) == ["question"]


def test_missing_question_class_raises():
    with pytest.raises(ValueError):
        predict_with_threshold([[0.5, 0.5]], ["bug", "enhancement"], 0.3)

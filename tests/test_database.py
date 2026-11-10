import pytest

from models.database import normalize_database_url


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("postgres://u:p@host/db", "postgresql+psycopg://u:p@host/db"),
        (
            "postgresql://u:p@host/db?sslmode=require",
            "postgresql+psycopg://u:p@host/db?sslmode=require",
        ),
        ("postgresql+psycopg://u:p@db:5432/tickets", "postgresql+psycopg://u:p@db:5432/tickets"),
        ("sqlite:///predictions.db", "sqlite:///predictions.db"),
        ("", ""),
    ],
)
def test_normalize_database_url(url, expected):
    assert normalize_database_url(url) == expected

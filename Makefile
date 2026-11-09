.PHONY: install train run test lint up down

install:
	python -m venv .venv
	.venv/bin/pip install -r requirements-dev.txt

train:
	.venv/bin/python train.py --tune-threshold --final

run:
	.venv/bin/python app.py

test:
	.venv/bin/python -m pytest

lint:
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .

up:
	docker compose up --build

down:
	docker compose down

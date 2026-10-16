.PHONY: install train run test

install:
	python -m venv .venv
	.venv/bin/pip install -r requirements-dev.txt

train:
	.venv/bin/python train.py

run:
	.venv/bin/python app.py

test:
	.venv/bin/python -m pytest

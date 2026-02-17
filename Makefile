PYTHON ?= python

install:
	$(PYTHON) -m pip install -e .[dev]

run:
	uvicorn app.main:app --reload

test:
	pytest

lint:
	ruff check .

.PHONY: install test lint data evaluate evaluate-answers figures up down

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests scripts

data:
	python scripts/download_data.py

evaluate:
	pip install -e ".[eval]"
	python -m docrag.evaluate

evaluate-answers:
	pip install -e ".[eval]"
	python -m docrag.evaluate_answers

figures:
	pip install -e ".[figures]"
	python scripts/make_figures.py

up:
	docker compose up -d --build

down:
	docker compose down

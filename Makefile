.PHONY: install dev test lint format run clean docker-build docker-run

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/
	mypy src/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

run:
	streamlit run src/viz/app.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .mypy_cache .ruff_cache .coverage htmlcov dist build

docker-build:
	docker build -t llm-adversarial-arena .

docker-run:
	docker run -p 8501:8501 \
		-e OPENAI_API_KEY=$${OPENAI_API_KEY} \
		-e ANTHROPIC_API_KEY=$${ANTHROPIC_API_KEY} \
		llm-adversarial-arena

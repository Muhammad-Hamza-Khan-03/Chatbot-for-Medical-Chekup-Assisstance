.PHONY: help install setup train pipeline query test clean

help:
	@echo "Medical Q&A System - Available Commands:"
	@echo "  make install     - Install dependencies"
	@echo "  make setup       - Setup project (install + download NLTK data)"
	@echo "  make train       - Train a model"
	@echo "  make pipeline    - Run complete pipeline"
	@echo "  make query       - Query and explore data"
	@echo "  make test        - Run tests"
	@echo "  make clean       - Clean generated files"

install:
	pip install -r requirements.txt

setup: install
	python -c "import nltk; nltk.download('punkt', quiet=True)"
	@echo "Setup complete!"

train:
	python scripts/train.py --config config/default_config.yaml

pipeline:
	python scripts/run_pipeline.py --config config/default_config.yaml

query:
	python scripts/query_data.py --data-path data/raw/train.csv --plot-distribution

test:
	pytest tests/ -v

clean:
	rm -rf logs/*
	rm -rf models/checkpoints/*
	rm -rf __pycache__ src/**/__pycache__ tests/__pycache__
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete


# Quick Start Guide

Get up and running with the Medical Q&A System in minutes!

## Prerequisites

- Python 3.8+
- pip
- (Optional) CUDA-capable GPU for faster training

## Installation

1. **Clone and navigate to the project:**
```bash
cd Chatbot-for-Medical-Chekup-Assisstance
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download NLTK data:**
```python
python -c "import nltk; nltk.download('punkt')"
```

Or use the Makefile:
```bash
make setup
```

## Prepare Data

Place your training data CSV file in `data/raw/train.csv`. The CSV should have columns:
- `qtype`: Question type
- `Question`: The question
- `Answer`: The answer

If you already have `train.csv` in the root directory, you can copy it:
```bash
cp train.csv data/raw/train.csv
```

## Quick Examples

### 1. Explore Your Data

```bash
python scripts/query_data.py --data-path data/raw/train.csv --plot-distribution
```

### 2. Run Complete Pipeline

This will load data, train a model, and evaluate it:

```bash
python scripts/run_pipeline.py --config config/default_config.yaml
```

### 3. Train a Specific Model

Train BERT with custom dropout:

```bash
python scripts/train.py --config config/default_config.yaml --model-type bert --dropout 0.3
```

### 4. Train Different Models

**BERT:**
```bash
python scripts/train.py --model-type bert --model-name bert-base-cased
```

**MobileBERT:**
```bash
python scripts/train.py --model-type mobilebert --model-name google/mobilebert-uncased
```

**RoBERTa:**
```bash
python scripts/train.py --model-type roberta --model-name roberta-base
```

## Using Makefile (Optional)

If you have `make` installed:

```bash
make setup      # Install and setup
make pipeline   # Run complete pipeline
make train      # Train model
make query      # Explore data
make test       # Run tests
make clean      # Clean generated files
```

## Configuration

Edit `config/default_config.yaml` to customize:
- Model hyperparameters
- Training settings
- Evaluation options
- Logging preferences

## Example Python Usage

```python
from src.pipeline import MedicalQAPipeline

# Initialize pipeline
pipeline = MedicalQAPipeline(config_path='config/default_config.yaml')

# Run everything
trainer, results = pipeline.run_full_pipeline()

# Access results
print(f"BLEU Score: {results['comprehensive_evaluation']['bleu_score']}")
```

## Troubleshooting

**Issue: CUDA out of memory**
- Solution: Reduce `train_batch_size` in config file

**Issue: Data file not found**
- Solution: Ensure `data/raw/train.csv` exists or update path in config

**Issue: NLTK data missing**
- Solution: Run `python -c "import nltk; nltk.download('punkt')"`

**Issue: Model download fails**
- Solution: Check internet connection, models are downloaded from HuggingFace

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check `scripts/example_usage.py` for more examples
- Customize `config/default_config.yaml` for your needs
- Explore the code in `src/` directory

## Support

For issues or questions, please open an issue on GitHub.


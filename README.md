# Medical Question Answering System

A production-grade medical question-answering system using transformer models (BERT, MobileBERT, RoBERTa) for medical assistance and patient support.

## Features

- **Multiple Model Support**: Train and evaluate BERT, MobileBERT, and RoBERTa models
- **Comprehensive Evaluation**: BLEU and ROUGE score calculation
- **Data Exploration**: SQL-based querying of medical Q&A dataset
- **Experiment Tracking**: Integration with Weights & Biases (wandb)
- **Production-Ready**: Modular architecture with proper logging and configuration management
- **Visualization**: Automatic generation of plots and evaluation metrics

## Documentation

This project includes comprehensive documentation to help you understand the research, implementation, and usage:

- **[Research.md](Research.md)**: Complete research paper documenting the comparative study of transformer models (BERT, MobileBERT, RoBERTa) on medical question answering. Includes methodology, results, analysis, and conclusions.

- **[QUICKSTART.md](QUICKSTART.md)**: Quick start guide for getting up and running with the project.

- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)**: Detailed documentation of the project structure and architecture.

- **README.md** (this file): Overview, installation, and usage instructions.

### Research Overview

The research conducted in this project evaluates three transformer models on the MedQuad dataset (16,407 medical Q&A pairs):

- **BERT**: Best overall balance with 18.5% exact match and 76.4% similar matches
- **MobileBERT**: Lightweight model optimized for resource-constrained environments
- **RoBERTa**: Highest ROUGE scores (F1: 0.593) demonstrating superior semantic understanding

For detailed results, methodology, and analysis, see [Research.md](Research.md).

## Project Structure

```
Chatbot-for-Medical-Chekup-Assisstance/
├── src/                    # Source code
│   ├── data/              # Data loading and preprocessing
│   │   ├── data_loader.py
│   │   └── data_formatter.py
│   ├── models/            # Model training and evaluation
│   │   ├── model_trainer.py
│   │   └── model_evaluator.py
│   ├── utils/             # Utility functions
│   │   ├── logger.py
│   │   └── visualization.py
│   ├── config/            # Configuration management
│   │   └── config.py
│   └── pipeline.py        # Main pipeline
├── scripts/               # Executable scripts
│   ├── train.py          # Training script
│   ├── run_pipeline.py   # Complete pipeline
│   └── query_data.py     # Data exploration
├── config/               # Configuration files
│   └── default_config.yaml
├── data/                 # Data storage
│   ├── raw/              # Raw data files
│   └── processed/        # Processed data
├── models/               # Model checkpoints
│   └── checkpoints/
├── logs/                 # Logs and plots
│   └── plots/
├── tests/                # Unit tests
├── Notebook/             # Jupyter notebooks
│   └── MDQCW1.ipynb     # Main research notebook
├── requirements.txt      # Python dependencies
├── setup.py             # Package setup
├── Research.md          # Research paper and analysis
├── QUICKSTART.md        # Quick start guide
├── PROJECT_STRUCTURE.md  # Project structure documentation
└── README.md            # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Muhammad-Hamza-Khan-03/Chatbot-for-Medical-Chekup-Assisstance
cd Chatbot-for-Medical-Chekup-Assisstance
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package (optional, for development):
```bash
pip install -e .
```

5. Download NLTK data (required for BLEU score calculation):
```python
import nltk
nltk.download('punkt')
```

## Usage

### Configuration

The system uses YAML configuration files. A default configuration is provided at `config/default_config.yaml`. You can create custom configurations or override settings via command-line arguments.

### Quick Start

#### 1. Run Complete Pipeline

Run the entire pipeline (data loading, training, and evaluation):

```bash
python scripts/run_pipeline.py --config config/default_config.yaml
```

#### 2. Train a Model

Train a specific model:

```bash
python scripts/train.py --config config/default_config.yaml --model-type bert --dropout 0.3
```

Available model types:
- `bert` (default: bert-base-cased)
- `mobilebert` (default: google/mobilebert-uncased)
- `roberta` (default: roberta-base)

#### 3. Explore Data

Query and explore the dataset:

```bash
python scripts/query_data.py --data-path data/raw/train.csv --plot-distribution
```

Execute custom SQL queries:

```bash
python scripts/query_data.py --data-path data/raw/train.csv --query "SELECT Question, Answer FROM df WHERE qtype = 'treatment' LIMIT 5"
```

### Programmatic Usage

```python
from src.pipeline import MedicalQAPipeline

# Initialize pipeline
pipeline = MedicalQAPipeline(config_path='config/default_config.yaml')

# Run complete pipeline
trainer, results = pipeline.run_full_pipeline()

# Or run step by step
df = pipeline.load_and_explore_data()
train_data, test_data, test_df, formatted_test_data = pipeline.prepare_data(df)
trainer = pipeline.train_model(train_data, test_data)
results = pipeline.evaluate_model(trainer, test_data, test_df, formatted_test_data)
```

## Configuration

### Default Configuration

The default configuration (`config/default_config.yaml`) includes:

- **Data Settings**: Path to training data, test split ratio
- **Model Settings**: Model type, hyperparameters (learning rate, batch size, dropout)
- **Training Settings**: Output directory, wandb project, evaluation settings
- **Evaluation Settings**: Metrics to calculate, number of samples
- **Logging Settings**: Log level, log directory

### Custom Configuration

Create your own configuration file:

```yaml
data:
  train_path: "data/raw/train.csv"
  test_size: 0.25
  random_state: 42

model:
  model_type: "bert"
  model_name: "bert-base-cased"
  learning_rate: 5e-5
  num_train_epochs: 3
  train_batch_size: 16
  dropout: 0.3
  use_cuda: true

training:
  output_dir: "models/checkpoints"
  wandb_project: "medical-qa-experiments"
  evaluate_during_training: true
```

## Data Format

The system expects a CSV file with the following columns:
- `qtype`: Question type (e.g., treatment, symptoms, prevention)
- `Question`: The medical question
- `Answer`: The answer to the question

Example:
```csv
qtype,Question,Answer
treatment,"What is the treatment for diabetes?","Treatment includes medication, diet, and exercise."
symptoms,"What are symptoms of flu?","Symptoms include fever, cough, and fatigue."
```

## Model Training

### Supported Models

1. **BERT**: `bert-base-cased` - Standard BERT model
2. **MobileBERT**: `google/mobilebert-uncased` - Lightweight BERT variant
3. **RoBERTa**: `roberta-base` - Robustly optimized BERT

### Training Process

1. Data is converted to SQuAD format
2. Split into training and test sets
3. Model is fine-tuned on the training data
4. Evaluation is performed during and after training
5. Metrics (BLEU, ROUGE) are calculated

### Hyperparameters

Key hyperparameters that can be adjusted:
- `learning_rate`: Learning rate (default: 5e-5)
- `num_train_epochs`: Number of training epochs (default: 1)
- `train_batch_size`: Batch size (default: 16)
- `dropout`: Dropout rate (default: 0.3)

## Evaluation Metrics

The system calculates:

1. **Basic Evaluation**: Correct, similar, incorrect predictions
2. **BLEU Score**: Measures n-gram precision
3. **ROUGE Score**: Measures recall-oriented metrics (ROUGE-1, ROUGE-2, ROUGE-L)

For detailed evaluation results, metric analysis, and comparative performance across models, see [Research.md](Research.md).

## Experiment Tracking

To enable Weights & Biases tracking:

1. Install wandb: `pip install wandb`
2. Login: `wandb login`
3. Set `wandb_project` in configuration file

## Logging

Logs are saved to the `logs/` directory. Log levels can be configured:
- `DEBUG`: Detailed information
- `INFO`: General information (default)
- `WARNING`: Warning messages
- `ERROR`: Error messages
- `CRITICAL`: Critical errors

## Development

### Running Tests

```bash
pytest tests/
```

### Code Structure

- **Modular Design**: Each component is in its own module
- **Separation of Concerns**: Data, models, and utilities are separated
- **Configuration Management**: Centralized configuration system
- **Logging**: Comprehensive logging throughout the pipeline

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `train_batch_size` in configuration
2. **NLTK Data Missing**: Run `nltk.download('punkt')`
3. **Model Not Found**: Ensure you have internet connection for downloading models
4. **Data File Not Found**: Check the path in configuration file

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Citation

If you use this code in your research, please cite:

```bibtex
@software{medical_qa_system,
  title = {Medical Question Answering System},
  author = {Muhammad Hamza},
  year = {2025},
  url = {https://github.com/Muhammad-Hamza-Khan-03/Chatbot-for-Medical-Chekup-Assisstance}
}
```

## Acknowledgments

- Hugging Face for transformer models
- SimpleTransformers for easy model training
- Weights & Biases for experiment tracking

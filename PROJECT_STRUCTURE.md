# Project Structure

This document describes the production-grade structure of the Medical Q&A System.

## Directory Tree

```
Chatbot-for-Medical-Chekup-Assisstance/
│
├── src/                          # Main source code
│   ├── __init__.py
│   ├── pipeline.py              # Main pipeline orchestrator
│   │
│   ├── data/                    # Data handling modules
│   │   ├── __init__.py
│   │   ├── data_loader.py       # Data loading and SQL querying
│   │   └── data_formatter.py    # Data format conversion (SQuAD format)
│   │
│   ├── models/                  # Model training and evaluation
│   │   ├── __init__.py
│   │   ├── model_trainer.py     # Model training logic
│   │   └── model_evaluator.py   # Model evaluation and metrics
│   │
│   ├── utils/                   # Utility functions
│   │   ├── __init__.py
│   │   ├── logger.py            # Logging configuration
│   │   └── visualization.py    # Plotting utilities
│   │
│   └── config/                  # Configuration management
│       ├── __init__.py
│       └── config.py            # Config loading and management
│
├── scripts/                      # Executable scripts
│   ├── train.py                 # Training script
│   ├── run_pipeline.py          # Complete pipeline runner
│   ├── query_data.py            # Data exploration script
│   └── example_usage.py         # Usage examples
│
├── config/                       # Configuration files
│   └── default_config.yaml      # Default configuration
│
├── data/                         # Data storage
│   ├── raw/                     # Raw data files
│   │   └── .gitkeep
│   └── processed/               # Processed data
│       └── .gitkeep
│
├── models/                       # Model storage
│   └── checkpoints/             # Saved model checkpoints
│
├── logs/                         # Logs and outputs
│   └── plots/                   # Generated plots
│
├── tests/                        # Unit tests
│   ├── __init__.py
│   └── test_data_loader.py      # Data loader tests
│
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
├── Makefile                      # Common commands
├── .gitignore                    # Git ignore rules
├── README.md                     # Main documentation
├── QUICKSTART.md                 # Quick start guide
└── PROJECT_STRUCTURE.md          # This file
```

## Module Descriptions

### Data Modules (`src/data/`)

- **DataLoader**: Handles loading CSV data, SQL querying, and basic data exploration
- **DataFormatter**: Converts data between DataFrame and SQuAD format for model training

### Model Modules (`src/models/`)

- **ModelTrainer**: Manages model initialization, training, and checkpointing
- **ModelEvaluator**: Handles model evaluation, BLEU/ROUGE score calculation

### Utility Modules (`src/utils/`)

- **logger**: Centralized logging configuration
- **visualization**: Plotting functions for data exploration and evaluation

### Configuration (`src/config/`)

- **config**: YAML/JSON configuration loading and management

### Pipeline (`src/pipeline.py`)

- **MedicalQAPipeline**: Main orchestrator class that coordinates all components

## Scripts

1. **train.py**: Train a model with custom parameters
2. **run_pipeline.py**: Run the complete pipeline (load → train → evaluate)
3. **query_data.py**: Explore and query the dataset
4. **example_usage.py**: Example code snippets

## Configuration

Configuration is managed through YAML files in `config/`. The default configuration includes:
- Data paths and split settings
- Model hyperparameters
- Training settings
- Evaluation options
- Logging preferences

## Data Flow

```
Raw CSV → DataLoader → DataFrame
                    ↓
            DataFormatter → SQuAD Format
                    ↓
            Train/Test Split
                    ↓
            ModelTrainer → Trained Model
                    ↓
            ModelEvaluator → Metrics (BLEU, ROUGE)
```

## Key Features

1. **Modular Design**: Each component is independent and reusable
2. **Configuration-Driven**: All settings in YAML files
3. **Comprehensive Logging**: Detailed logs for debugging
4. **Production-Ready**: Error handling, type hints, documentation
5. **Extensible**: Easy to add new models or evaluation metrics

## Adding New Features

### Adding a New Model

1. Update `ModelTrainer` to support new model type
2. Add model configuration to `default_config.yaml`
3. Test with training script

### Adding a New Metric

1. Add calculation method to `ModelEvaluator`
2. Update evaluation pipeline
3. Add to configuration if needed

### Adding Data Processing

1. Add preprocessing function to `DataFormatter`
2. Update pipeline to use new preprocessing
3. Add tests

## Best Practices

- Always use configuration files for settings
- Log important operations
- Handle errors gracefully
- Write tests for new features
- Follow the existing code structure
- Document new functions and classes


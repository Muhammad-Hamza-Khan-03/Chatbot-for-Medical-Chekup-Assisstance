import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import MedicalQAPipeline
from src.utils.logger import setup_logging
import logging

logger = logging.getLogger(__name__)


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train medical Q&A model")
    parser.add_argument(
        '--config',
        type=str,
        default='config/default_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['bert', 'mobilebert', 'roberta'],
        help='Override model type from config'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        help='Override model name from config'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        help='Override dropout from config'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Initialize pipeline
    pipeline = MedicalQAPipeline(config_path=args.config)
    
    # Load and prepare data
    df = pipeline.load_and_explore_data()
    train_data, test_data, test_df, formatted_test_data = pipeline.prepare_data(df)
    
    # Train model
    trainer = pipeline.train_model(
        train_data,
        test_data,
        model_type=args.model_type,
        model_name=args.model_name,
        dropout=args.dropout
    )
    
    # Evaluate model
    results = pipeline.evaluate_model(trainer, test_data, test_df, formatted_test_data)
    
    logger.info("Training completed successfully")
    logger.info(f"Evaluation results: {results}")


if __name__ == "__main__":
    main()


import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import MedicalQAPipeline
from src.data import DataLoader
from src.utils.logger import setup_logging
import logging

logger = logging.getLogger(__name__)


def example_basic_usage():
    """Example of basic pipeline usage"""
    print("=" * 60)
    print("Example: Basic Pipeline Usage")
    print("=" * 60)
    
    # Setup logging
    setup_logging()
    
    # Initialize pipeline
    pipeline = MedicalQAPipeline(config_path='config/default_config.yaml')
    
    # Run complete pipeline
    trainer, results = pipeline.run_full_pipeline()
    
    print("\nPipeline completed!")
    print(f"Model type: {trainer.model_type}")
    print(f"Evaluation results: {results}")


def example_data_exploration():
    """Example of data exploration"""
    print("=" * 60)
    print("Example: Data Exploration")
    print("=" * 60)
    
    # Setup logging
    setup_logging()
    
    # Load data
    loader = DataLoader('data/raw/train.csv')
    df = loader.load_data()
    
    # Get distribution
    distribution = loader.get_qtype_distribution()
    print("\nQuestion Type Distribution:")
    print(distribution)
    
    # Sample data
    sample = loader.sample_data(n=5)
    print("\nSample Questions:")
    print(sample[['qtype', 'Question']].head())
    
    # Query data
    query = """
    SELECT Question, Answer
    FROM df
    WHERE qtype = 'treatment'
    LIMIT 3
    """
    result = loader.query_data(query)
    print("\nQuery Results (Treatment questions):")
    print(result)


def example_custom_training():
    """Example of custom model training"""
    print("=" * 60)
    print("Example: Custom Model Training")
    print("=" * 60)
    
    # Setup logging
    setup_logging()
    
    # Initialize pipeline
    pipeline = MedicalQAPipeline(config_path='config/default_config.yaml')
    
    # Load and prepare data
    df = pipeline.load_and_explore_data()
    train_data, test_data, test_df, formatted_test_data = pipeline.prepare_data(df)
    
    # Train with custom parameters
    trainer = pipeline.train_model(
        train_data,
        test_data,
        model_type='bert',
        model_name='bert-base-cased',
        dropout=0.3
    )
    
    # Evaluate
    results = pipeline.evaluate_model(trainer, test_data, test_df, formatted_test_data)
    
    print("\nTraining completed!")
    print(f"BLEU Score: {results['comprehensive_evaluation']['bleu_score']:.4f}")
    print(f"ROUGE-1 F1: {results['comprehensive_evaluation']['rouge_score']['rouge-1']['f']:.4f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Example usage of Medical Q&A System")
    parser.add_argument(
        '--example',
        type=str,
        choices=['basic', 'explore', 'custom'],
        default='basic',
        help='Which example to run'
    )
    
    args = parser.parse_args()
    
    if args.example == 'basic':
        example_basic_usage()
    elif args.example == 'explore':
        example_data_exploration()
    elif args.example == 'custom':
        example_custom_training()


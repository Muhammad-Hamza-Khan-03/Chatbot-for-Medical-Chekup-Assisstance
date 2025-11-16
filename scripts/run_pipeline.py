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
    """Main pipeline function"""
    parser = argparse.ArgumentParser(description="Run complete medical Q&A pipeline")
    parser.add_argument(
        '--config',
        type=str,
        default='config/default_config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Initialize and run pipeline
    pipeline = MedicalQAPipeline(config_path=args.config)
    trainer, results = pipeline.run_full_pipeline()
    
    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()


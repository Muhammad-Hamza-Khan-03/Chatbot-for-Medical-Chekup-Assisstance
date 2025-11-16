import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import DataLoader
from src.utils.logger import setup_logging
from src.utils.visualization import VisualizationUtils
import logging

logger = logging.getLogger(__name__)


def main():
    """Main query function"""
    parser = argparse.ArgumentParser(description="Query and explore medical Q&A dataset")
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/raw/train.csv',
        help='Path to data CSV file'
    )
    parser.add_argument(
        '--query',
        type=str,
        help='SQL query to execute on dataset'
    )
    parser.add_argument(
        '--plot-distribution',
        action='store_true',
        help='Plot question type distribution'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Load data
    loader = DataLoader(args.data_path)
    df = loader.load_data()
    
    # Print data info
    info = loader.get_data_info()
    logger.info(f"Dataset info: {info}")
    
    # Plot distribution if requested
    if args.plot_distribution:
        value_counts = loader.get_qtype_distribution()
        visualizer = VisualizationUtils()
        visualizer.plot_qtype_distribution(value_counts)
    
    # Execute query if provided
    if args.query:
        result = loader.query_data(args.query)
        print("\nQuery Results:")
        print(result)
        return result
    
    # Print sample data
    sample = loader.sample_data(n=5)
    print("\nSample Data:")
    print(sample)


if __name__ == "__main__":
    main()


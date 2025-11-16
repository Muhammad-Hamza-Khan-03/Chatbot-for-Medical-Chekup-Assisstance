import logging
import wandb
from pathlib import Path
from typing import Optional, Dict, Any
from sklearn.model_selection import train_test_split

from src.data import DataLoader, DataFormatter
from src.models import ModelTrainer, ModelEvaluator
from src.utils.logger import setup_logging
from src.utils.visualization import VisualizationUtils
from src.config import load_config

logger = logging.getLogger(__name__)


class MedicalQAPipeline:
    """Main pipeline for medical question-answering system"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize pipeline
        
        Args:
            config_path: Optional path to configuration file
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Setup logging
        setup_logging(
            log_level=self.config.logging.log_level,
            log_dir=self.config.logging.log_dir
        )
        
        # Initialize components
        self.data_loader = DataLoader(self.config.data.train_path)
        self.data_formatter = DataFormatter()
        self.visualizer = VisualizationUtils()
        
        # Create directories
        self._create_directories()
        
    def _create_directories(self):
        """Create necessary directories"""
        dirs = [
            self.config.training.output_dir,
            self.config.logging.log_dir,
            'data/processed',
            'models/checkpoints'
        ]
        
        if hasattr(self.config, 'visualization') and self.config.visualization.save_plots:
            dirs.append(self.config.visualization.plot_dir)
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def load_and_explore_data(self):
        """Load data and perform exploration"""
        logger.info("=" * 50)
        logger.info("Loading and exploring data")
        logger.info("=" * 50)
        
        # Load data
        df = self.data_loader.load_data()
        
        # Get data info
        info = self.data_loader.get_data_info()
        logger.info(f"Dataset shape: {info['shape']}")
        
        # Plot qtype distribution
        value_counts = self.data_loader.get_qtype_distribution()
        logger.info(f"Question type distribution:\n{value_counts}")
        
        plot_path = None
        if hasattr(self.config, 'visualization') and self.config.visualization.save_plots:
            plot_path = Path(self.config.visualization.plot_dir) / "qtype_distribution.png"
        
        self.visualizer.plot_qtype_distribution(value_counts, save_path=plot_path)
        
        return df
    
    def prepare_data(self, df):
        """
        Prepare data for training
        
        Args:
            df: DataFrame with training data
            
        Returns:
            Tuple of (train_data, test_data, test_df)
        """
        logger.info("=" * 50)
        logger.info("Preparing data for training")
        logger.info("=" * 50)
        
        # Convert to SQuAD format
        formatted_data = self.data_formatter.convert_df_to_structure(df)
        
        # Split data
        train_data, test_data = train_test_split(
            formatted_data,
            test_size=self.config.data.test_size,
            random_state=self.config.data.random_state
        )
        
        logger.info(f"Training examples: {len(train_data)}")
        logger.info(f"Test examples: {len(test_data)}")
        
        # Get test DataFrame for evaluation
        test_indices = [int(item['qas'][0]['id']) for item in test_data]
        test_df = df.iloc[test_indices].reset_index(drop=True)
        
        # Format test data for prediction
        formatted_test_data = self.data_formatter.convert_to_predict_format(test_df)
        
        return train_data, test_data, test_df, formatted_test_data
    
    def train_model(
        self,
        train_data,
        test_data,
        model_type: Optional[str] = None,
        model_name: Optional[str] = None,
        dropout: Optional[float] = None
    ):
        """
        Train a model
        
        Args:
            train_data: Training data
            test_data: Test data for evaluation
            model_type: Optional model type override
            model_name: Optional model name override
            dropout: Optional dropout override
            
        Returns:
            Trained ModelTrainer instance
        """
        logger.info("=" * 50)
        logger.info("Training model")
        logger.info("=" * 50)
        
        # Initialize wandb if project is specified
        wandb_project = getattr(self.config.training, 'wandb_project', None)
        if wandb_project:
            wandb.init(project=wandb_project)
        
        # Get model parameters
        model_type = model_type or self.config.model.model_type
        model_name = model_name or self.config.model.model_name
        dropout = dropout or self.config.model.dropout
        
        # Create trainer
        trainer = ModelTrainer(
            model_type=model_type,
            model_name=model_name,
            output_dir=self.config.training.output_dir,
            use_cuda=self.config.model.use_cuda,
            learning_rate=self.config.model.learning_rate,
            num_train_epochs=self.config.model.num_train_epochs,
            train_batch_size=self.config.model.train_batch_size,
            dropout=dropout,
            evaluate_during_training=self.config.training.evaluate_during_training,
            save_steps=getattr(self.config.training, 'save_steps', -1),
            n_best_size=getattr(self.config.training, 'n_best_size', 5)
        )
        
        # Create and train model
        trainer.create_model(wandb_project=wandb_project)
        trainer.train(train_data, eval_data=test_data)
        
        return trainer
    
    def evaluate_model(
        self,
        trainer: ModelTrainer,
        test_data,
        test_df,
        formatted_test_data
    ) -> Dict[str, Any]:
        """
        Evaluate a trained model
        
        Args:
            trainer: Trained ModelTrainer instance
            test_data: Test data in SQuAD format
            test_df: Test DataFrame
            formatted_test_data: Formatted test data for prediction
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info("=" * 50)
        logger.info("Evaluating model")
        logger.info("=" * 50)
        
        evaluator = ModelEvaluator(trainer.model)
        
        # Basic evaluation
        eval_results = evaluator.evaluate(test_data)
        
        # Plot evaluation results
        plot_path = None
        if hasattr(self.config, 'visualization') and self.config.visualization.save_plots:
            plot_path = Path(self.config.visualization.plot_dir) / f"eval_results_{trainer.model_type}.png"
        
        self.visualizer.plot_evaluation_results(eval_results, save_path=plot_path)
        
        # Comprehensive evaluation with BLEU and ROUGE
        n_samples = getattr(self.config.evaluation, 'n_samples', None)
        comprehensive_results = evaluator.evaluate_comprehensive(
            test_df,
            formatted_test_data,
            n_samples=n_samples
        )
        
        # Combine results
        all_results = {
            'basic_evaluation': eval_results,
            'comprehensive_evaluation': comprehensive_results
        }
        
        return all_results
    
    def run_full_pipeline(self):
        """Run the complete pipeline"""
        try:
            # Load and explore data
            df = self.load_and_explore_data()
            
            # Prepare data
            train_data, test_data, test_df, formatted_test_data = self.prepare_data(df)
            
            # Train model
            trainer = self.train_model(train_data, test_data)
            
            # Evaluate model
            results = self.evaluate_model(trainer, test_data, test_df, formatted_test_data)
            
            logger.info("=" * 50)
            logger.info("Pipeline completed successfully")
            logger.info("=" * 50)
            
            return trainer, results
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}", exc_info=True)
            raise


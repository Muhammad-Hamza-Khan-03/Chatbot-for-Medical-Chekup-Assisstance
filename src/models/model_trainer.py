import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from simpletransformers.question_answering import QuestionAnsweringModel
import torch

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train question-answering models"""
    
    def __init__(
        self,
        model_type: str,
        model_name: str,
        output_dir: str = "models/checkpoints",
        use_cuda: bool = True,
        **kwargs
    ):
        """
        Initialize ModelTrainer
        
        Args:
            model_type: Type of model ('bert', 'mobilebert', 'roberta')
            model_name: HuggingFace model name
            output_dir: Directory to save model checkpoints
            use_cuda: Whether to use CUDA
            **kwargs: Additional training arguments
        """
        self.model_type = model_type
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_cuda = use_cuda and torch.cuda.is_available()
        
        # Default training arguments
        self.train_args = {
            'learning_rate': 5e-5,
            'num_train_epochs': 1,
            'train_batch_size': 16,
            'overwrite_output_dir': True,
            'reprocess_input_data': True,
            'evaluate_during_training': True,
            'save_steps': -1,
            'n_best_size': 5,
            'dropout': 0.3,
            **kwargs
        }
        
        self.model: Optional[QuestionAnsweringModel] = None
        
    def create_model(self, wandb_project: Optional[str] = None):
        """
        Create and initialize the model
        
        Args:
            wandb_project: Optional wandb project name for logging
        """
        if wandb_project:
            self.train_args['wandb_project'] = wandb_project
        
        logger.info(f"Initializing {self.model_type} model: {self.model_name}")
        logger.info(f"Using CUDA: {self.use_cuda}")
        logger.info(f"Training arguments: {self.train_args}")
        
        self.model = QuestionAnsweringModel(
            self.model_type,
            self.model_name,
            args=self.train_args,
            use_cuda=self.use_cuda
        )
        
    def train(
        self,
        train_data: List[Dict[str, Any]],
        eval_data: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Train the model
        
        Args:
            train_data: Training data in SQuAD format
            eval_data: Optional evaluation data
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        logger.info(f"Training model on {len(train_data)} examples")
        if eval_data:
            logger.info(f"Evaluating on {len(eval_data)} examples")
        
        self.model.train_model(train_data, eval_data=eval_data)
        logger.info("Training completed")
        
    def save_model(self, path: Optional[str] = None):
        """
        Save the trained model
        
        Args:
            path: Optional custom path to save model
        """
        if self.model is None:
            raise ValueError("No model to save. Train model first.")
        
        save_path = Path(path) if path else self.output_dir / f"{self.model_type}_{self.model_name.replace('/', '_')}"
        logger.info(f"Saving model to {save_path}")
        # Model is already saved during training, but we can save additional metadata here
        return save_path
    
    def load_model(self, model_path: str):
        """
        Load a pre-trained model
        
        Args:
            model_path: Path to the saved model
        """
        logger.info(f"Loading model from {model_path}")
        self.model = QuestionAnsweringModel(
            self.model_type,
            model_path,
            args=self.train_args,
            use_cuda=self.use_cuda
        )


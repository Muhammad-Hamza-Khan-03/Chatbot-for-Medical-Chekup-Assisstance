import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration class for managing settings"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize Config from dictionary
        
        Args:
            config_dict: Dictionary with configuration values
        """
        self._config = config_dict
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        result = {}
        for key, value in self._config.items():
            if isinstance(value, dict):
                result[key] = Config(value).to_dict()
            else:
                result[key] = getattr(self, key)
        return result
    
    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        """
        Load configuration from file
        
        Args:
            config_path: Path to config file (JSON or YAML)
            
        Returns:
            Config object
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        logger.info(f"Loading configuration from {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            elif config_path.suffix == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        return cls(config_dict)
    
    def save(self, config_path: str):
        """
        Save configuration to file
        
        Args:
            config_path: Path to save config file
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self.to_dict()
        
        with open(config_path, 'w') as f:
            if config_path.suffix in ['.yaml', '.yml']:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            elif config_path.suffix == '.json':
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        logger.info(f"Saved configuration to {config_path}")


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file or return default
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        Config object
    """
    if config_path:
        return Config.from_file(config_path)
    else:
        # Return default config
        default_config = {
            'data': {
                'train_path': 'data/raw/train.csv',
                'test_size': 0.25,
                'random_state': 42
            },
            'model': {
                'model_type': 'bert',
                'model_name': 'bert-base-cased',
                'learning_rate': 5e-5,
                'num_train_epochs': 1,
                'train_batch_size': 16,
                'dropout': 0.3,
                'use_cuda': True
            },
            'training': {
                'output_dir': 'models/checkpoints',
                'wandb_project': None,
                'evaluate_during_training': True
            },
            'evaluation': {
                'n_samples': None,
                'calculate_bleu': True,
                'calculate_rouge': True
            },
            'logging': {
                'log_level': 'INFO',
                'log_dir': 'logs'
            }
        }
        return Config(default_config)


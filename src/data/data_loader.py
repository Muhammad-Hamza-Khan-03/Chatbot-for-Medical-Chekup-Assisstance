import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Tuple
from pandasql import sqldf

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and explore medical Q&A dataset"""
    
    def __init__(self, data_path: str):
        """
        Initialize DataLoader
        
        Args:
            data_path: Path to the CSV file
        """
        self.data_path = Path(data_path)
        self.df: Optional[pd.DataFrame] = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load CSV data into DataFrame
        
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If data file doesn't exist
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        logger.info(f"Loading data from {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(self.df)} rows with {len(self.df.columns)} columns")
        logger.info(f"Data info:\n{self.df.info()}")
        
        return self.df
    
    def get_data_info(self) -> dict:
        """
        Get basic information about the dataset
        
        Returns:
            Dictionary with dataset information
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        return {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'null_counts': self.df.isnull().sum().to_dict(),
            'memory_usage': self.df.memory_usage(deep=True).sum()
        }
    
    def query_data(self, query: str) -> pd.DataFrame:
        """
        Execute SQL query on the DataFrame
        
        Args:
            query: SQL query string
            
        Returns:
            Query results as DataFrame
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        logger.debug(f"Executing query: {query}")
        result = sqldf(query, globals())
        return result
    
    def get_qtype_distribution(self) -> pd.Series:
        """
        Get distribution of question types
        
        Returns:
            Series with qtype value counts
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        return self.df['qtype'].value_counts()
    
    def sample_data(self, n: int = 10, random_state: int = 42) -> pd.DataFrame:
        """
        Sample random rows from dataset
        
        Args:
            n: Number of samples
            random_state: Random seed
            
        Returns:
            Sampled DataFrame
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        return self.df.sample(n=n, random_state=random_state)


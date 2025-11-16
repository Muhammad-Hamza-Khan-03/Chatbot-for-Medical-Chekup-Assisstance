import pandas as pd
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class DataFormatter:
    """Format data for question-answering models"""
    
    @staticmethod
    def convert_df_to_structure(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Convert DataFrame to SQuAD-like format for training
        
        Args:
            df: DataFrame with 'Question' and 'Answer' columns
            
        Returns:
            List of dictionaries in SQuAD format
        """
        data = []
        for idx, row in df.iterrows():
            qas = [{
                "id": str(idx),
                "is_impossible": False,
                "question": row['Question'],
                "answers": [
                    {
                        "text": row['Answer'],
                        "answer_start": 0
                    }
                ]
            }]
            
            data.append({
                "context": row['Answer'],
                "qas": qas
            })
        
        logger.info(f"Converted {len(data)} examples to SQuAD format")
        return data
    
    @staticmethod
    def convert_to_predict_format(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Convert DataFrame to prediction format
        
        Args:
            df: DataFrame with 'Question' and 'Answer' columns
            
        Returns:
            List of dictionaries in prediction format
        """
        to_predict = []
        for index, row in df.iterrows():
            context = row['Answer']
            qas = [{
                "question": row['Question'],
                "id": str(index),
            }]
            to_predict.append({"context": context, "qas": qas})
        
        logger.info(f"Converted {len(to_predict)} examples to prediction format")
        return to_predict


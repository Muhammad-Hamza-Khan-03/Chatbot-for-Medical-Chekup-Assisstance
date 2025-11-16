import logging
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
from simpletransformers.question_answering import QuestionAnsweringModel
from rouge import Rouge
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate question-answering models"""
    
    def __init__(self, model: QuestionAnsweringModel):
        """
        Initialize ModelEvaluator
        
        Args:
            model: Trained QuestionAnsweringModel
        """
        self.model = model
        self.rouge = Rouge()
        
    def evaluate(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate model on test data
        
        Args:
            test_data: Test data in SQuAD format
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating model on {len(test_data)} examples")
        result, texts = self.model.eval_model(test_data)
        logger.info(f"Evaluation results: {result}")
        return result
    
    def predict(
        self,
        data: List[Dict[str, Any]],
        n_samples: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Make predictions on data and return results
        
        Args:
            data: Data in prediction format
            n_samples: Number of samples to predict
            
        Returns:
            List of dictionaries with predictions
        """
        results = []
        for i in range(min(n_samples, len(data))):
            actual_qs = data[i]['qas'][0]['question']
            answers = self.model.predict([data[i]])
            pred_ans = str(max(answers[0][0]['answer']))
            
            results.append({
                'question': actual_qs,
                'predicted_answer': pred_ans
            })
        
        return results
    
    def get_reference_and_predicted_answers(
        self,
        test_df: pd.DataFrame,
        formatted_test_data: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[str]]:
        """
        Get reference and predicted answers for metric calculation
        
        Args:
            test_df: DataFrame with actual answers
            formatted_test_data: Formatted test data for prediction
            
        Returns:
            Tuple of (reference_answers, predicted_answers)
        """
        reference_answers = []
        predicted_answers = []
        
        logger.info(f"Generating predictions for {len(test_df)} examples")
        for i in range(len(test_df)):
            actual_ans = test_df.iloc[i, 2]  # Assuming Answer is in column 2
            answers = self.model.predict([formatted_test_data[i]])
            pred_ans = str(max(answers[0][0]['answer']))
            
            reference_answers.append(actual_ans)
            predicted_answers.append(pred_ans)
            
            if (i + 1) % 10 == 0:
                logger.debug(f"Processed {i + 1}/{len(test_df)} examples")
        
        return reference_answers, predicted_answers
    
    def calculate_bleu_score(
        self,
        reference_answers: List[str],
        predicted_answers: List[str],
        use_smoothing: bool = True,
        smoothing_method: int = 2
    ) -> float:
        """
        Calculate BLEU score
        
        Args:
            reference_answers: List of reference answers
            predicted_answers: List of predicted answers
            use_smoothing: Whether to use smoothing function
            smoothing_method: Smoothing method (1 or 2)
            
        Returns:
            BLEU score
        """
        # Convert to list of lists for BLEU calculation
        refs = [[ref.split()] for ref in reference_answers]
        preds = [pred.split() for pred in predicted_answers]
        
        if use_smoothing:
            smoothing_function = SmoothingFunction().method1 if smoothing_method == 1 else SmoothingFunction().method2
            bleu_score = corpus_bleu(refs, preds, smoothing_function=smoothing_function)
        else:
            bleu_score = corpus_bleu(refs, preds)
        
        logger.info(f"BLEU Score: {bleu_score}")
        return bleu_score
    
    def calculate_rouge_score(
        self,
        reference_answers: List[str],
        predicted_answers: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate ROUGE score
        
        Args:
            reference_answers: List of reference answers
            predicted_answers: List of predicted answers
            
        Returns:
            Dictionary with ROUGE scores
        """
        logger.info("Calculating ROUGE scores")
        rouge_score = self.rouge.get_scores(
            reference_answers,
            predicted_answers,
            avg=True,
            ignore_empty=True
        )
        logger.info(f"ROUGE Score: {rouge_score}")
        return rouge_score
    
    def evaluate_comprehensive(
        self,
        test_df: pd.DataFrame,
        formatted_test_data: List[Dict[str, Any]],
        n_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation including BLEU and ROUGE scores
        
        Args:
            test_df: DataFrame with test data
            formatted_test_data: Formatted test data for prediction
            n_samples: Optional limit on number of samples to evaluate
            
        Returns:
            Dictionary with all evaluation metrics
        """
        if n_samples:
            test_df = test_df[:n_samples]
            formatted_test_data = formatted_test_data[:n_samples]
        
        # Get reference and predicted answers
        ref_answers, pred_answers = self.get_reference_and_predicted_answers(
            test_df, formatted_test_data
        )
        
        # Calculate metrics
        bleu_score = self.calculate_bleu_score(ref_answers, pred_answers)
        bleu_score_smooth = self.calculate_bleu_score(
            ref_answers, pred_answers, use_smoothing=True, smoothing_method=2
        )
        rouge_score = self.calculate_rouge_score(ref_answers, pred_answers)
        
        results = {
            'bleu_score': bleu_score,
            'bleu_score_smooth': bleu_score_smooth,
            'rouge_score': rouge_score,
            'n_samples': len(test_df)
        }
        
        return results


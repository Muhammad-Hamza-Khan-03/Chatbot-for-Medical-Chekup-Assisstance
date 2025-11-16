import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class VisualizationUtils:
    """Utility class for creating visualizations"""
    
    @staticmethod
    def plot_qtype_distribution(
        value_counts,
        save_path: Optional[str] = None,
        figsize: tuple = (10, 5)
    ):
        """
        Plot distribution of question types
        
        Args:
            value_counts: Series with qtype value counts
            save_path: Optional path to save the plot
            figsize: Figure size tuple
        """
        plt.figure(figsize=figsize)
        value_counts.plot(kind='bar')
        plt.title('Distribution of Question Types')
        plt.xlabel('Question Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_evaluation_results(
        results: Dict[str, Any],
        save_path: Optional[str] = None,
        figsize: tuple = (10, 5)
    ):
        """
        Plot evaluation results
        
        Args:
            results: Dictionary with evaluation metrics
            save_path: Optional path to save the plot
            figsize: Figure size tuple
        """
        # Filter out non-numeric values
        plot_data = {k: v for k, v in results.items() if isinstance(v, (int, float))}
        
        if not plot_data:
            logger.warning("No numeric values to plot")
            return
        
        plt.figure(figsize=figsize)
        plt.bar(plot_data.keys(), plot_data.values())
        plt.xlabel('Evaluation Metrics')
        plt.ylabel('Counts')
        plt.title('Evaluation Results')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        else:
            plt.show()
        
        plt.close()


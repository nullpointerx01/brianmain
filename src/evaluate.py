"""
Model Evaluation Module for Brain Tumor Detection
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_curve, auc, precision_recall_curve
)
import itertools

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CLASS_NAMES, MODEL_PATH
from src.data_preprocessing import DataPreprocessor
from src.model import load_trained_model


class ModelEvaluator:
    """Class for comprehensive model evaluation"""
    
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.class_names = CLASS_NAMES
        
    def load_model(self):
        """Load the trained model"""
        self.model = load_trained_model(self.model_path)
        return self
    
    def evaluate(self, test_generator):
        """
        Perform comprehensive evaluation
        
        Args:
            test_generator: Test data generator
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            self.load_model()
        
        # Get predictions
        print("Generating predictions...")
        predictions = self.model.predict(test_generator, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_generator.classes
        
        # Calculate metrics
        results = {
            'y_true': y_true,
            'y_pred': y_pred,
            'predictions': predictions,
            'classification_report': classification_report(
                y_true, y_pred, 
                target_names=self.class_names,
                output_dict=True
            ),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        return results
    
    def plot_confusion_matrix(self, cm, save_path='confusion_matrix.png'):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix', fontsize=16)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Confusion matrix saved to {save_path}")
    
    def plot_roc_curves(self, y_true, predictions, save_path='roc_curves.png'):
        """Plot ROC curves for each class"""
        from sklearn.preprocessing import label_binarize
        
        # Binarize labels
        y_true_bin = label_binarize(y_true, classes=range(len(self.class_names)))
        
        plt.figure(figsize=(10, 8))
        
        colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
        
        for i, (class_name, color) in enumerate(zip(self.class_names, colors)):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], predictions[:, i])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color=color, lw=2,
                    label=f'{class_name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Multi-class Classification', fontsize=14)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"ROC curves saved to {save_path}")
    
    def plot_precision_recall_curves(self, y_true, predictions, save_path='pr_curves.png'):
        """Plot Precision-Recall curves"""
        from sklearn.preprocessing import label_binarize
        
        y_true_bin = label_binarize(y_true, classes=range(len(self.class_names)))
        
        plt.figure(figsize=(10, 8))
        colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
        
        for i, (class_name, color) in enumerate(zip(self.class_names, colors)):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], predictions[:, i])
            plt.plot(recall, precision, color=color, lw=2, label=class_name)
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves', fontsize=14)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Precision-Recall curves saved to {save_path}")
    
    def print_classification_report(self, report):
        """Print formatted classification report"""
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        
        # Print per-class metrics
        print(f"\n{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
        print("-"*60)
        
        for class_name in self.class_names:
            metrics = report[class_name]
            print(f"{class_name:<15} {metrics['precision']:>10.3f} {metrics['recall']:>10.3f} "
                  f"{metrics['f1-score']:>10.3f} {metrics['support']:>10.0f}")
        
        print("-"*60)
        
        # Print overall metrics
        print(f"\n{'Accuracy':<15} {report['accuracy']:>40.3f}")
        print(f"{'Macro Avg':<15} {report['macro avg']['precision']:>10.3f} "
              f"{report['macro avg']['recall']:>10.3f} {report['macro avg']['f1-score']:>10.3f}")
        print(f"{'Weighted Avg':<15} {report['weighted avg']['precision']:>10.3f} "
              f"{report['weighted avg']['recall']:>10.3f} {report['weighted avg']['f1-score']:>10.3f}")
        
        print("="*60)
    
    def generate_evaluation_report(self, save_dir='.'):
        """Generate complete evaluation report"""
        print("="*60)
        print("BRAIN TUMOR DETECTION - MODEL EVALUATION")
        print("="*60)
        
        # Load test data
        preprocessor = DataPreprocessor()
        _, _, test_generator = preprocessor.create_data_generators()
        
        # Evaluate
        results = self.evaluate(test_generator)
        
        # Print classification report
        self.print_classification_report(results['classification_report'])
        
        # Plot confusion matrix
        self.plot_confusion_matrix(
            results['confusion_matrix'],
            os.path.join(save_dir, 'confusion_matrix.png')
        )
        
        # Plot ROC curves
        self.plot_roc_curves(
            results['y_true'],
            results['predictions'],
            os.path.join(save_dir, 'roc_curves.png')
        )
        
        # Plot PR curves
        self.plot_precision_recall_curves(
            results['y_true'],
            results['predictions'],
            os.path.join(save_dir, 'pr_curves.png')
        )
        
        return results


def evaluate_model(model_path=MODEL_PATH):
    """Main evaluation function"""
    evaluator = ModelEvaluator(model_path)
    evaluator.load_model()
    results = evaluator.generate_evaluation_report()
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Brain Tumor Detection Model')
    parser.add_argument('--model', '-m', type=str, default=str(MODEL_PATH),
                       help='Path to the trained model')
    parser.add_argument('--output', '-o', type=str, default='.',
                       help='Output directory for evaluation plots')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        print("Please train the model first using: python src/train.py")
        sys.exit(1)
    
    evaluate_model(args.model)

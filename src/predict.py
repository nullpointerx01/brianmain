"""
Prediction Module for Brain Tumor Detection
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CLASS_NAMES, MODEL_PATH, IMG_SIZE
from src.data_preprocessing import preprocess_single_image
from src.model import load_trained_model


class BrainTumorPredictor:
    """Class for making predictions on brain MRI images"""
    
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.class_names = CLASS_NAMES
        
    def load_model(self):
        """Load the trained model"""
        print(f"Loading model from {self.model_path}...")
        self.model = load_trained_model(self.model_path)
        print("✓ Model loaded successfully")
        return self
    
    def predict(self, image_path):
        """
        Make prediction on a single image
        
        Args:
            image_path: Path to the MRI image
            
        Returns:
            dict: Prediction results with class name and confidence scores
        """
        if self.model is None:
            self.load_model()
        
        # Preprocess image
        image = preprocess_single_image(image_path)
        
        # Make prediction
        predictions = self.model.predict(image, verbose=0)
        
        # Get results
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = self.class_names[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx]
        
        # Create detailed results
        results = {
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'all_probabilities': {
                self.class_names[i]: float(predictions[0][i]) 
                for i in range(len(self.class_names))
            },
            'is_tumor_detected': predicted_class != 'notumor'
        }
        
        return results
    
    def predict_batch(self, image_paths):
        """
        Make predictions on multiple images
        
        Args:
            image_paths: List of paths to MRI images
            
        Returns:
            list: List of prediction results
        """
        results = []
        for path in image_paths:
            result = self.predict(path)
            result['image_path'] = path
            results.append(result)
        return results
    
    def print_prediction(self, results):
        """Print prediction results in a formatted way"""
        print("\n" + "="*50)
        print("BRAIN TUMOR DETECTION RESULTS")
        print("="*50)
        
        if results['is_tumor_detected']:
            print(f"⚠️  TUMOR DETECTED: {results['predicted_class'].upper()}")
        else:
            print("✓  NO TUMOR DETECTED")
        
        print(f"\nConfidence: {results['confidence']*100:.2f}%")
        
        print("\nAll Class Probabilities:")
        for class_name, prob in results['all_probabilities'].items():
            bar = "█" * int(prob * 30)
            print(f"  {class_name:15} {prob*100:6.2f}% {bar}")
        
        print("="*50)


def predict_image(image_path, model_path=MODEL_PATH):
    """
    Convenience function to predict a single image
    
    Args:
        image_path: Path to the MRI image
        model_path: Path to the trained model
        
    Returns:
        dict: Prediction results
    """
    predictor = BrainTumorPredictor(model_path)
    predictor.load_model()
    results = predictor.predict(image_path)
    predictor.print_prediction(results)
    return results


def get_tumor_info(tumor_type):
    """Get information about detected tumor type"""
    tumor_info = {
        'glioma': {
            'name': 'Glioma',
            'description': 'Gliomas are tumors that arise from glial cells in the brain.',
            'severity': 'High',
            'common_symptoms': ['Headaches', 'Seizures', 'Memory problems', 'Personality changes'],
            'recommendation': 'Immediate consultation with a neuro-oncologist is recommended.'
        },
        'meningioma': {
            'name': 'Meningioma',
            'description': 'Meningiomas arise from the meninges, the membranes surrounding the brain.',
            'severity': 'Varies (Most are benign)',
            'common_symptoms': ['Headaches', 'Vision problems', 'Hearing loss', 'Memory loss'],
            'recommendation': 'Follow-up imaging and consultation with a neurosurgeon.'
        },
        'pituitary': {
            'name': 'Pituitary Tumor',
            'description': 'Pituitary tumors develop in the pituitary gland at the base of the brain.',
            'severity': 'Varies (Most are benign)',
            'common_symptoms': ['Vision problems', 'Hormonal imbalances', 'Headaches', 'Fatigue'],
            'recommendation': 'Consultation with an endocrinologist and neurosurgeon.'
        },
        'notumor': {
            'name': 'No Tumor',
            'description': 'No tumor detected in the MRI scan.',
            'severity': 'None',
            'common_symptoms': [],
            'recommendation': 'Continue regular health checkups.'
        }
    }
    return tumor_info.get(tumor_type, tumor_info['notumor'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict brain tumor from MRI image')
    parser.add_argument('--image', '-i', type=str, required=True,
                       help='Path to the MRI image')
    parser.add_argument('--model', '-m', type=str, default=str(MODEL_PATH),
                       help='Path to the trained model')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        print("Please train the model first using: python src/train.py")
        sys.exit(1)
    
    results = predict_image(args.image, args.model)
    
    # Print tumor information
    tumor_info = get_tumor_info(results['predicted_class'])
    print(f"\n📋 Tumor Information:")
    print(f"   Type: {tumor_info['name']}")
    print(f"   Description: {tumor_info['description']}")
    print(f"   Severity: {tumor_info['severity']}")
    print(f"   Recommendation: {tumor_info['recommendation']}")

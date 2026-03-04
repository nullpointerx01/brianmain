"""
Unit Tests for Brain Tumor Detection Model
"""

import os
import sys
import unittest
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestConfig(unittest.TestCase):
    """Test configuration settings"""
    
    def test_config_imports(self):
        """Test that config can be imported"""
        from src.config import IMG_SIZE, NUM_CLASSES, CLASS_NAMES
        self.assertEqual(IMG_SIZE, 224)
        self.assertEqual(NUM_CLASSES, 4)
        self.assertEqual(len(CLASS_NAMES), 4)
    
    def test_class_names(self):
        """Test class names are correct"""
        from src.config import CLASS_NAMES
        expected = ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.assertEqual(CLASS_NAMES, expected)


class TestDataPreprocessor(unittest.TestCase):
    """Test data preprocessing functions"""
    
    def test_preprocessor_initialization(self):
        """Test DataPreprocessor can be initialized"""
        from src.data_preprocessing import DataPreprocessor
        preprocessor = DataPreprocessor(img_size=224)
        self.assertEqual(preprocessor.img_size, 224)
    
    def test_preprocess_single_image_function(self):
        """Test single image preprocessing function exists"""
        from src.data_preprocessing import preprocess_single_image
        self.assertTrue(callable(preprocess_single_image))


class TestModel(unittest.TestCase):
    """Test model architecture"""
    
    def test_create_custom_cnn(self):
        """Test custom CNN creation"""
        from src.model import create_custom_cnn
        model = create_custom_cnn()
        
        # Check model is created
        self.assertIsNotNone(model)
        
        # Check input shape
        self.assertEqual(model.input_shape, (None, 224, 224, 3))
        
        # Check output shape
        self.assertEqual(model.output_shape, (None, 4))
    
    def test_model_compilation(self):
        """Test model can be compiled"""
        from src.model import create_custom_cnn, compile_model
        model = create_custom_cnn()
        model = compile_model(model)
        
        # Check optimizer is set
        self.assertIsNotNone(model.optimizer)
    
    def test_brain_tumor_cnn_class(self):
        """Test BrainTumorCNN wrapper class"""
        from src.model import BrainTumorCNN
        
        cnn = BrainTumorCNN(model_type='custom')
        cnn.build()
        
        self.assertIsNotNone(cnn.model)


class TestPredictor(unittest.TestCase):
    """Test prediction module"""
    
    def test_predictor_initialization(self):
        """Test BrainTumorPredictor can be initialized"""
        from src.predict import BrainTumorPredictor
        
        predictor = BrainTumorPredictor()
        self.assertIsNotNone(predictor)
        self.assertEqual(len(predictor.class_names), 4)
    
    def test_tumor_info_function(self):
        """Test tumor information function"""
        from src.predict import get_tumor_info
        
        for tumor_type in ['glioma', 'meningioma', 'pituitary', 'notumor']:
            info = get_tumor_info(tumor_type)
            self.assertIn('name', info)
            self.assertIn('description', info)
            self.assertIn('severity', info)
            self.assertIn('recommendation', info)


class TestUtils(unittest.TestCase):
    """Test utility functions"""
    
    def test_timer_context_manager(self):
        """Test Timer context manager"""
        from utils.helpers import Timer
        import time
        
        with Timer("Test") as t:
            time.sleep(0.1)
        
        self.assertIsNotNone(t)
    
    def test_format_time(self):
        """Test time formatting function"""
        from utils.helpers import format_time
        
        self.assertIn('s', format_time(30))
        self.assertIn('m', format_time(120))
        self.assertIn('h', format_time(3700))


class TestInputValidation(unittest.TestCase):
    """Test input validation"""
    
    def test_image_size_validation(self):
        """Test that image size is within acceptable range"""
        from src.config import IMG_SIZE
        self.assertGreaterEqual(IMG_SIZE, 32)
        self.assertLessEqual(IMG_SIZE, 512)
    
    def test_batch_size_validation(self):
        """Test that batch size is reasonable"""
        from src.config import BATCH_SIZE
        self.assertGreater(BATCH_SIZE, 0)
        self.assertLessEqual(BATCH_SIZE, 128)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)

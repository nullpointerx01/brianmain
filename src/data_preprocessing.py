"""
Data Preprocessing Module for Brain Tumor Detection
"""

import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import (
    TRAIN_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE, 
    CLASS_NAMES, AUGMENTATION_CONFIG, VALIDATION_SPLIT
)


class DataPreprocessor:
    """Class to handle all data preprocessing operations"""
    
    def __init__(self, img_size=IMG_SIZE):
        self.img_size = img_size
        self.class_names = CLASS_NAMES
        
    def load_image(self, image_path):
        """Load and preprocess a single image"""
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # Normalize pixel values
        img = img / 255.0
        
        return img
    
    def load_dataset(self, data_dir):
        """Load entire dataset from directory"""
        images = []
        labels = []
        
        print(f"Loading data from {data_dir}...")
        
        for class_idx, class_name in enumerate(tqdm(self.class_names)):
            class_dir = os.path.join(data_dir, class_name)
            
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} does not exist")
                continue
                
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                
                try:
                    img = self.load_image(img_path)
                    images.append(img)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue
        
        return np.array(images), np.array(labels)
    
    def create_data_generators(self):
        """Create training and validation data generators with augmentation"""
        
        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=AUGMENTATION_CONFIG['rotation_range'],
            width_shift_range=AUGMENTATION_CONFIG['width_shift_range'],
            height_shift_range=AUGMENTATION_CONFIG['height_shift_range'],
            shear_range=AUGMENTATION_CONFIG['shear_range'],
            zoom_range=AUGMENTATION_CONFIG['zoom_range'],
            horizontal_flip=AUGMENTATION_CONFIG['horizontal_flip'],
            fill_mode=AUGMENTATION_CONFIG['fill_mode'],
            validation_split=VALIDATION_SPLIT
        )
        
        # Test data generator (no augmentation)
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Training generator
        train_generator = train_datagen.flow_from_directory(
            TRAIN_DIR,
            target_size=(self.img_size, self.img_size),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        # Validation generator
        validation_generator = train_datagen.flow_from_directory(
            TRAIN_DIR,
            target_size=(self.img_size, self.img_size),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        # Test generator
        test_generator = test_datagen.flow_from_directory(
            TEST_DIR,
            target_size=(self.img_size, self.img_size),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
        
        return train_generator, validation_generator, test_generator
    
    def visualize_samples(self, images, labels, num_samples=9):
        """Visualize sample images from the dataset"""
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        
        indices = np.random.choice(len(images), num_samples, replace=False)
        
        for i, ax in enumerate(axes.flat):
            idx = indices[i]
            ax.imshow(images[idx])
            ax.set_title(f"Class: {self.class_names[labels[idx]]}")
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('sample_images.png')
        plt.show()
        
    def get_class_distribution(self, labels):
        """Get class distribution from labels"""
        unique, counts = np.unique(labels, return_counts=True)
        distribution = dict(zip([self.class_names[i] for i in unique], counts))
        
        print("\nClass Distribution:")
        for class_name, count in distribution.items():
            print(f"  {class_name}: {count}")
        
        return distribution
    
    def apply_clahe(self, image):
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        # Convert to LAB color space
        lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return enhanced / 255.0


def preprocess_single_image(image_path, img_size=IMG_SIZE):
    """Preprocess a single image for prediction"""
    preprocessor = DataPreprocessor(img_size)
    img = preprocessor.load_image(image_path)
    return np.expand_dims(img, axis=0)


if __name__ == "__main__":
    # Test the data preprocessing
    preprocessor = DataPreprocessor()
    
    print("Creating data generators...")
    try:
        train_gen, val_gen, test_gen = preprocessor.create_data_generators()
        print(f"Training samples: {train_gen.samples}")
        print(f"Validation samples: {val_gen.samples}")
        print(f"Test samples: {test_gen.samples}")
    except Exception as e:
        print(f"Note: {e}")
        print("Please ensure you have downloaded the dataset and placed it in the data/ directory")

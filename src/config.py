"""
Configuration file for Brain Tumor Detection System
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "Training"
TEST_DIR = DATA_DIR / "Testing"

# Model directory
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "brain_tumor_model.h5"

# Image configuration
IMG_SIZE = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
BATCH_SIZE = 32

# Training configuration
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# Class labels
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
NUM_CLASSES = len(CLASS_NAMES)

# Data augmentation parameters
AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.2,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'fill_mode': 'nearest'
}

# Model configuration
MODEL_CONFIG = {
    'dropout_rate': 0.5,
    'l2_regularization': 0.01,
    'use_batch_norm': True
}

# Logging
LOG_DIR = BASE_DIR / "logs"
TENSORBOARD_LOG_DIR = LOG_DIR / "tensorboard"

# Create directories if they don't exist
for directory in [DATA_DIR, TRAIN_DIR, TEST_DIR, MODEL_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

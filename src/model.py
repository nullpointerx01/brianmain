"""
CNN Model Architecture for Brain Tumor Detection
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)

from config import (
    IMG_SHAPE, NUM_CLASSES, LEARNING_RATE, 
    MODEL_CONFIG, MODEL_PATH, TENSORBOARD_LOG_DIR
)


def create_custom_cnn(input_shape=IMG_SHAPE, num_classes=NUM_CLASSES):
    """
    Create a custom CNN architecture for brain tumor classification
    """
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fourth Convolutional Block
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fifth Convolutional Block
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fully Connected Layers
        layers.Flatten(),
        layers.Dense(512, activation='relu', 
                    kernel_regularizer=regularizers.l2(MODEL_CONFIG['l2_regularization'])),
        layers.BatchNormalization(),
        layers.Dropout(MODEL_CONFIG['dropout_rate']),
        
        layers.Dense(256, activation='relu',
                    kernel_regularizer=regularizers.l2(MODEL_CONFIG['l2_regularization'])),
        layers.BatchNormalization(),
        layers.Dropout(MODEL_CONFIG['dropout_rate']),
        
        # Output Layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def create_transfer_learning_model(base_model_name='vgg16', input_shape=IMG_SHAPE, num_classes=NUM_CLASSES):
    """
    Create a transfer learning model using pre-trained weights
    
    Args:
        base_model_name: 'vgg16', 'resnet50', or 'efficientnet'
    """
    # Select base model
    if base_model_name == 'vgg16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == 'resnet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == 'efficientnet':
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Unknown base model: {base_model_name}")
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Create new model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(MODEL_CONFIG['dropout_rate']),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(MODEL_CONFIG['dropout_rate']),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def compile_model(model, learning_rate=LEARNING_RATE):
    """Compile the model with optimizer and loss function"""
    optimizer = Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model


def get_callbacks(model_path=MODEL_PATH):
    """Get training callbacks"""
    callbacks = [
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Save best model
        ModelCheckpoint(
            filepath=str(model_path),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard logging
        TensorBoard(
            log_dir=str(TENSORBOARD_LOG_DIR),
            histogram_freq=1
        )
    ]
    
    return callbacks


def load_trained_model(model_path=MODEL_PATH):
    """Load a trained model from file"""
    return tf.keras.models.load_model(str(model_path))


def get_model_summary(model):
    """Print model summary"""
    model.summary()
    return model


class BrainTumorCNN:
    """
    Wrapper class for Brain Tumor Detection CNN
    """
    def __init__(self, model_type='custom', base_model='vgg16'):
        self.model_type = model_type
        self.base_model = base_model
        self.model = None
        self.history = None
        
    def build(self):
        """Build the model"""
        if self.model_type == 'custom':
            self.model = create_custom_cnn()
        else:
            self.model = create_transfer_learning_model(self.base_model)
        
        self.model = compile_model(self.model)
        return self
    
    def train(self, train_data, val_data, epochs=50):
        """Train the model"""
        callbacks = get_callbacks()
        
        self.history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks
        )
        
        return self.history
    
    def predict(self, image):
        """Make prediction on a single image"""
        return self.model.predict(image)
    
    def evaluate(self, test_data):
        """Evaluate model on test data"""
        return self.model.evaluate(test_data)
    
    def save(self, path=MODEL_PATH):
        """Save the model"""
        self.model.save(str(path))
        
    def load(self, path=MODEL_PATH):
        """Load a saved model"""
        self.model = load_trained_model(path)
        return self


if __name__ == "__main__":
    # Test model creation
    print("Creating Custom CNN Model...")
    model = create_custom_cnn()
    model = compile_model(model)
    model.summary()
    
    print("\n" + "="*50 + "\n")
    
    print("Creating Transfer Learning Model (VGG16)...")
    tl_model = create_transfer_learning_model('vgg16')
    tl_model = compile_model(tl_model)
    tl_model.summary()

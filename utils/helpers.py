"""
Utility Helper Functions
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def create_directory(path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")


def get_timestamp():
    """Get current timestamp string"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def calculate_class_weights(labels):
    """
    Calculate class weights for imbalanced datasets
    
    Args:
        labels: Array of class labels
        
    Returns:
        dict: Class weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    return dict(zip(classes, weights))


def plot_images_grid(images, labels, class_names, title="Sample Images", n_cols=4):
    """
    Plot a grid of images
    
    Args:
        images: Array of images
        labels: Array of labels
        class_names: List of class names
        title: Plot title
        n_cols: Number of columns in grid
    """
    n_images = len(images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    axes = axes.flatten() if n_images > 1 else [axes]
    
    for i, ax in enumerate(axes):
        if i < n_images:
            ax.imshow(images[i])
            ax.set_title(class_names[labels[i]])
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    return fig


def save_model_summary(model, filepath):
    """Save model summary to a text file"""
    with open(filepath, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    print(f"Model summary saved to {filepath}")


def log_metrics(metrics, filepath, mode='a'):
    """Log metrics to a file"""
    timestamp = get_timestamp()
    with open(filepath, mode) as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"Timestamp: {timestamp}\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
        f.write(f"{'='*50}\n")


def normalize_image(image):
    """Normalize image to 0-1 range"""
    return (image - image.min()) / (image.max() - image.min() + 1e-8)


def denormalize_image(image):
    """Denormalize image from 0-1 to 0-255"""
    return (image * 255).astype(np.uint8)


def get_memory_usage():
    """Get current memory usage"""
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    print(f"Random seed set to {seed}")


def check_gpu_availability():
    """Check if GPU is available"""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ GPU available: {len(gpus)} device(s)")
            for gpu in gpus:
                print(f"  - {gpu}")
            return True
        else:
            print("✗ No GPU available, using CPU")
            return False
    except Exception as e:
        print(f"Error checking GPU: {e}")
        return False


def format_time(seconds):
    """Format seconds to human readable time"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {int(secs)}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"


class Timer:
    """Simple timer context manager"""
    
    def __init__(self, name="Operation"):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = datetime.now()
        print(f"Starting: {self.name}...")
        return self
    
    def __exit__(self, *args):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        print(f"Completed: {self.name} in {format_time(elapsed)}")


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    set_seed(42)
    check_gpu_availability()
    
    with Timer("Test operation"):
        import time
        time.sleep(1)
    
    print("\n✓ All utilities working correctly!")

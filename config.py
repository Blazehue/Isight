"""
Configuration Module for Sign Language Detection

This module contains configuration parameters for the sign language detection system.

Author: Blazehue
Date: January 2026
"""

# Model Training Configuration
TRAINING_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'cross_validation_folds': 5,
}

# Detection Configuration
DETECTION_CONFIG = {
    'confidence_threshold': 0.80,
    'buffer_size': 5,
    'min_detection_confidence': 0.7,
    'min_tracking_confidence': 0.7,
    'model_complexity': 1,
}

# Data Collection Configuration
DATA_COLLECTION_CONFIG = {
    'default_samples': 150,
    'save_directory': 'training_data',
}

# Visualization Configuration
VISUALIZATION_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 100,
}

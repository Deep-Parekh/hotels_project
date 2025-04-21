"""
Utility Functions and Classes
===========================

Provides supporting functionality for the recommendation system:
- Training utilities
- Model checkpointing
- Early stopping
- Evaluation metrics
- Data preparation scripts
- Visualization tools
"""

from .helpers import (
    EarlyStopping,
    ModelCheckpoint,
    fit,
    train_one_epoch,
    evaluate,
    plot_training_history,
    plot_learning_rate,
    create_training_summary
)

from .train_model import ModelTrainer
from .prepare_data import prepare_data

__all__ = [
    # Training utilities
    'EarlyStopping',
    'ModelCheckpoint',
    'fit',
    'train_one_epoch',
    'evaluate',
    
    # Visualization
    'plot_training_history',
    'plot_learning_rate',
    'create_training_summary',
    
    # Main classes
    'ModelTrainer',
    
    # Data preparation
    'prepare_data',
]

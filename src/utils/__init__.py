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

from .data_utils import count_lines, validate_directories, save_chunk_metadata
from .prepare_data import prepare_data
from .prepare_random_sample import prepare_random_sample
from .train_model import ModelTrainer
from .helpers import (
    fit,
    plot_training_history,
    plot_learning_rate,
    create_training_summary,
    ModelCheckpoint,
    EarlyStopping
)

__all__ = [
    'prepare_data',
    'prepare_random_sample',
    'ModelTrainer',
    'fit',
    'plot_training_history',
    'plot_learning_rate',
    'create_training_summary',
    'ModelCheckpoint',
    'EarlyStopping',
    'count_lines',
    'validate_directories',
    'save_chunk_metadata'
]

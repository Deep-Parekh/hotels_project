"""
Data Processing Module
=====================

Handles all data-related operations for the hotel recommendation system:
- Loading raw hotel review data
- Preprocessing text and ratings
- Creating datasets for different model types
- Data validation and cleaning
"""

from .loader import DataLoader
from .preprocessor import DataPreprocessor
from .dataset import (
    HotelReviewDataset,
    CFMatrixDataset
)

__all__ = [
    'DataLoader',
    'DataPreprocessor',
    'HotelReviewDataset',
    'CFMatrixDataset',
]

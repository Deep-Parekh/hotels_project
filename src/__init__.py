"""
Hotel Recommendation System
==========================

A comprehensive recommendation system for hotels using multiple approaches:
- Content-based filtering
- Collaborative filtering
- Hybrid recommendations

This package provides tools for:
- Data loading and preprocessing
- Model training and evaluation
- Recommendation generation
"""

from . import data
from . import models
from . import utils

__all__ = [
    'data',
    'models',
    'utils',
]

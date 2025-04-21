"""
Model Implementations
====================

Contains implementations of different recommendation models:
- Content-based neural network
- Collaborative filtering matrix factorization
- Hybrid recommendation model

Each model is designed to handle specific aspects of hotel recommendations,
from text processing to user-item interactions.
"""

from .content_based import ContentBasedModel
from .collaborative import MatrixFactorization
from .hybrid_recommender import HybridRecommender

__all__ = [
    'ContentBasedModel',
    'MatrixFactorization',
    'HybridRecommender',
]

"""Collaborative filtering model."""

import torch
import torch.nn as nn

class MatrixFactorization(nn.Module):
    """Matrix factorization model for collaborative filtering."""
    
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 32):
        super().__init__()
        
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        
        # Initialize embeddings
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)
    
    def forward(self, user_ids, item_ids):
        user_embeds = self.user_embeddings(user_ids)
        item_embeds = self.item_embeddings(item_ids)
        
        # Compute dot product
        ratings = (user_embeds * item_embeds).sum(dim=1)
        return ratings

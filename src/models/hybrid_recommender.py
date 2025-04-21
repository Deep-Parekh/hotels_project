import torch
import torch.nn as nn
from .content_based import ContentBasedModel
from .collaborative import MatrixFactorization

class HybridRecommender(nn.Module):
    """
    Combine CF embeddings + content features via MLP.
    """
    def __init__(self,
                 num_users, num_items,
                 content_input_dim,
                 cf_emb_dim=32,
                 hidden_dim=128,
                 dropout_p=0.3):
        super().__init__()
        # Collaborative part
        self.cf = MatrixFactorization(num_users, num_items, emb_dim=cf_emb_dim)
        # Content‑based part
        self.content_net = ContentBasedModel(content_input_dim, hidden_dim, dropout_p)
        # Fusion MLP
        fusion_input_dim = cf_emb_dim + 1 + 1  # user·item + content → adjust if you want e.g. content embedding
        self.fusion = nn.Sequential(
            nn.Linear(cf_emb_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, user_idx, item_idx, content_feats):
        cf_pred = self.cf(user_idx, item_idx).unsqueeze(-1)  # shape (B,1)
        content_pred = self.content_net(content_feats).unsqueeze(-1)
        x = torch.cat([cf_pred, content_pred], dim=-1)
        return self.fusion(x).squeeze(-1)

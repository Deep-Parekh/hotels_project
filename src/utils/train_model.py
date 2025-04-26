"""
Training script for hotel recommendation models.
Handles both content-based and collaborative filtering approaches.
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Union

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.data.dataset import ChunkedHotelReviewDataset, ChunkedCFMatrixDataset
from src.models.content_based import ContentBasedModel
from src.models.collaborative import MatrixFactorization
from src.models.hybrid_recommender import HybridRecommender
from src.utils.helpers import fit, plot_training_history, plot_learning_rate, create_training_summary, ModelCheckpoint

class ModelTrainer:
    """Handles training of recommendation models."""
    
    def __init__(self, config=None):
        """
        Initialize the trainer.
        
        Args:
            config (dict): Configuration for training
        """
        self.config = config or {
            'content_based': {
                'batch_size': 64,
                'learning_rate': 0.001,
                'num_epochs': 10,
                'hidden_dim': 128,
                'dropout': 0.3,
                'chunk_size': 10000
            },
            'collaborative': {
                'batch_size': 256,
                'learning_rate': 0.001,
                'num_epochs': 10,
                'embedding_dim': 32,
                'chunk_size': 10000
            },
            'hybrid': {
                'batch_size': 128,
                'learning_rate': 0.001,
                'num_epochs': 10,
                'cf_embedding_dim': 32,
                'hidden_dim': 128,
                'dropout': 0.3
            }
        }
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Setup checkpoint directory
        self.checkpoint_dir = Path('models/checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def prepare_content_based_data(self):
        """Prepare datasets for content-based model using chunked loading."""
        print("\nPreparing content-based datasets...")
        
        # Create chunked datasets
        full_dataset = ChunkedHotelReviewDataset(
            data_dir=project_root / "data/processed",
            chunk_size=self.config['content_based']['chunk_size']
        )
        
        # Create data loaders
        train_loader = DataLoader(
            full_dataset,
            batch_size=self.config['content_based']['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # For validation and testing, we use smaller chunks
        val_dataset = ChunkedHotelReviewDataset(
            data_dir=project_root / "data/processed",
            chunk_size=10000,  # Smaller chunks for validation
            shuffle=False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['content_based']['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        print(f"Dataset information:")
        print(f"- Total samples: {full_dataset.total_samples:,}")
        print(f"- Feature dimensions: {full_dataset.get_feature_dims()}")
        
        return train_loader, val_loader, val_loader, full_dataset.get_feature_dims()

    def prepare_cf_data(self):
        """Prepare datasets for collaborative filtering using chunked loading."""
        print("\nPreparing collaborative filtering datasets...")
        
        # Create chunked datasets
        full_dataset = ChunkedCFMatrixDataset(
            data_dir=project_root / "data/processed",
            chunk_size=self.config['collaborative']['chunk_size']
        )
        
        # Create data loaders
        train_loader = DataLoader(
            full_dataset,
            batch_size=self.config['collaborative']['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # For validation and testing, we use smaller chunks
        val_dataset = ChunkedCFMatrixDataset(
            data_dir=project_root / "data/processed",
            chunk_size=10000,  # Smaller chunks for validation
            shuffle=False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['collaborative']['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        num_users, num_hotels = full_dataset.get_dims()
        print(f"Dataset dimensions:")
        print(f"- Number of users: {num_users:,}")
        print(f"- Number of hotels: {num_hotels:,}")
        print(f"- Total interactions: {full_dataset.total_samples:,}")
        
        return train_loader, val_loader, val_loader, (num_users, num_hotels)

    def train_content_based(self):
        """Train content-based model with organized checkpointing."""
        print("\nTraining content-based model...")
        
        # Prepare data
        train_loader, val_loader, test_loader, input_dim = self.prepare_content_based_data()
        
        # Initialize model
        cfg = self.config['content_based']
        model = ContentBasedModel(
            input_dim=input_dim,
            hidden_dim=cfg['hidden_dim'],
            dropout=cfg['dropout']
        ).to(self.device)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )
        
        # Define metrics
        metrics = {
            'rmse': lambda outputs, targets: torch.sqrt(nn.MSELoss()(outputs, targets)),
            'mae': lambda outputs, targets: nn.L1Loss()(outputs, targets)
        }
        
        # Setup early stopping
        early_stopping = {
            'patience': 7,
            'min_delta': 1e-4,
            'mode': 'min',
            'verbose': True
        }
        
        # Setup checkpointing
        checkpoint = ModelCheckpoint(
            checkpoint_dir=self.checkpoint_dir,
            model_type='content_based',
            mode='min',
            save_best_only=False,  # Save historical checkpoints
            max_checkpoints=5      # Keep last 5 checkpoints
        )
        
        # Train using enhanced fit
        history = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=self.device,
            num_epochs=cfg['num_epochs'],
            scheduler=scheduler,
            metrics=metrics,
            early_stopping=early_stopping,
            checkpoint=checkpoint
        )
        
        # Visualize results
        self.visualize_training(history, 'content_based')
        
        return history, model

    def train_collaborative(self):
        """Train collaborative filtering model with organized checkpointing."""
        print("\nTraining collaborative filtering model...")
        
        # Prepare data
        train_loader, val_loader, test_loader, (num_users, num_hotels) = self.prepare_cf_data()
        
        # Initialize model
        cfg = self.config['collaborative']
        model = MatrixFactorization(
            num_users=num_users,
            num_items=num_hotels,
            embedding_dim=cfg['embedding_dim']
        ).to(self.device)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )
        
        # Define metrics
        metrics = {
            'rmse': lambda outputs, targets: torch.sqrt(nn.MSELoss()(outputs, targets)),
            'mae': lambda outputs, targets: nn.L1Loss()(outputs, targets)
        }
        
        # Setup checkpointing
        checkpoint = ModelCheckpoint(
            checkpoint_dir=self.checkpoint_dir,
            model_type='collaborative',
            mode='min',
            save_best_only=False,
            max_checkpoints=5
        )
        
        # Track learning rates
        learning_rates = []
        def lr_callback(optimizer):
            learning_rates.append(optimizer.param_groups[0]['lr'])
        
        # Add callback to scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2,
            callback=lr_callback
        )
        
        # Train using helpers.fit
        history = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=self.device,
            num_epochs=cfg['num_epochs'],
            scheduler=scheduler,
            metrics=metrics,
            early_stopping_patience=5,
            checkpoint_path='models/checkpoints/collaborative_best.pt'
        )
        
        # Visualize results
        self.visualize_training(history, 'collaborative', learning_rates)
        
        return history, model

    def visualize_training(self, history: Dict[str, List[float]], 
                          model_type: str,
                          learning_rates: List[float] = None):
        """
        Visualize training results and save plots.
        
        Args:
            history (dict): Training history dictionary
            model_type (str): Either 'content_based' or 'collaborative'
            learning_rates (list): Optional list of learning rates used during training
        """
        # Create plots directory if it doesn't exist
        plots_dir = Path('models/plots')
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot training history
        plot_training_history(
            history,
            save_path=str(plots_dir / f'{model_type}_training_history.png')
        )
        
        # Plot learning rate if provided
        if learning_rates:
            plot_learning_rate(
                history,
                learning_rates,
                save_path=str(plots_dir / f'{model_type}_learning_rate.png')
            )
        
        # Create and save training summary
        summary = create_training_summary(
            history,
            model_name=model_type,
            save_path=str(plots_dir / f'{model_type}_training_summary.csv')
        )
        
        print("\nTraining Summary:")
        print(summary.to_string(index=False))

    def load_best_model(self, model_type: str):
        """Load the best model of specified type."""
        if model_type == 'content_based':
            model_class = ContentBasedModel
            model_kwargs = {
                'input_dim': self.config['content_based']['input_dim'],
                'hidden_dim': self.config['content_based']['hidden_dim'],
                'dropout': self.config['content_based']['dropout']
            }
        elif model_type == 'collaborative':
            model_class = MatrixFactorization
            model_kwargs = {
                'num_users': self.config['collaborative']['num_users'],
                'num_items': self.config['collaborative']['num_items'],
                'embedding_dim': self.config['collaborative']['embedding_dim']
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model, checkpoint = ModelCheckpoint.load_best_model(
            self.checkpoint_dir,
            model_type,
            model_class,
            model_kwargs
        )
        
        return model, checkpoint

def main():
    """Main training script"""
    trainer = ModelTrainer()
    
    # Train models
    print("Training Content-Based Model...")
    cb_history, cb_model = trainer.train_content_based()
    
    print("\nTraining Collaborative Filtering Model...")
    cf_history, cf_model = trainer.train_collaborative()
    
    # Load best models
    best_cb_model, cb_checkpoint = trainer.load_best_model('content_based')
    best_cf_model, cf_checkpoint = trainer.load_best_model('collaborative')
    
    print("\nTraining complete! Models saved in organized checkpoint directories")

if __name__ == "__main__":
    main()

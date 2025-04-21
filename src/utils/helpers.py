import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Union, Tuple, Type
import pandas as pd
import torch.nn as nn
from pathlib import Path
import json
from datetime import datetime
import copy
import logging

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train the model for a single epoch.
    Args:
        model      : nn.Module
        dataloader : DataLoader yielding batches where the last element is the target
        optimizer  : torch.optim.Optimizer
        criterion  : loss function
        device     : torch.device
    Returns:
        Average training loss over the epoch.
    """
    model.train()
    running_loss = 0.0
    for batch in tqdm(dataloader, desc="Train", leave=False):
        *inputs, targets = batch
        inputs = [inp.to(device) for inp in inputs]
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(*inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * targets.size(0)

    return running_loss / len(dataloader.dataset)

def evaluate(model, dataloader, criterion, device, metrics=None):
    """
    Evaluate the model on a validation/test set.
    Args:
        model      : nn.Module
        dataloader : DataLoader
        criterion  : loss function
        device     : torch.device
        metrics    : dict of {name: fn(outputs, targets) -> scalar}
    Returns:
        dict: {
          "loss": <avg loss>,
          **{metric_name: avg_metric, …}
        }
    """
    model.eval()
    running_loss = 0.0
    metric_sums = {name: 0.0 for name in (metrics or {})}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval ", leave=False):
            *inputs, targets = batch
            inputs = [inp.to(device) for inp in inputs]
            targets = targets.to(device)

            outputs = model(*inputs)
            loss = criterion(outputs, targets)

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size

            for name, fn in (metrics or {}).items():
                metric_sums[name] += fn(outputs, targets) * batch_size

    avg_loss = running_loss / len(dataloader.dataset)
    results = {"loss": avg_loss}
    for name, total in metric_sums.items():
        results[name] = total / len(dataloader.dataset)

    return results

class EarlyStopping:
    """Early stopping handler with model state preservation.
    
    Args:
        patience (int): Number of epochs to wait for improvement before stopping
        min_delta (float): Minimum change in monitored value to qualify as an improvement
        mode (str): 'min' or 'max' depending on whether we want to minimize or maximize the metric
        verbose (bool): Whether to print messages about early stopping status
    """
    def __init__(self, 
                 patience: int = 7, 
                 min_delta: float = 0.0, 
                 mode: str = 'min',
                 verbose: bool = True):
        if mode not in ['min', 'max']:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
            
        self.patience = patience
        self.min_delta = abs(min_delta)
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.early_stop = False
        self.best_state = None
        
        # For tracking improvement
        self.is_better = lambda a, b: a < b if mode == 'min' else a > b
    
    def __call__(self, value: float, model_state: Dict[str, Any]) -> bool:
        """
        Check if training should stop.
        
        Args:
            value (float): Current value to monitor
            model_state (dict): Current model state dict
            
        Returns:
            bool: True if training should stop
        """
        if self.is_better(value, self.best_value):
            improvement = abs(value - self.best_value) > self.min_delta
            if improvement:
                if self.verbose:
                    print(f'Improvement found: {self.best_value:.6f} -> {value:.6f}')
                self.best_value = value
                self.counter = 0
                self.best_state = copy.deepcopy(model_state)
                return False
        
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
            if self.verbose:
                print(f'Early stopping triggered after {self.patience} epochs without improvement')
            return True
        
        if self.verbose:
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
        return False
    
    def get_best_state(self) -> Optional[Dict[str, Any]]:
        """Return the best model state encountered."""
        return self.best_state

class ModelCheckpoint:
    """Enhanced model checkpointing with organized subfolder structure.
    
    Saves checkpoints in the following structure:
    models/
    └── checkpoints/
        ├── content_based/
        │   ├── best/
        │   │   ├── model_best.pt
        │   │   └── metadata.json
        │   └── history/
        │       ├── checkpoint_epoch_1.pt
        │       └── checkpoint_epoch_2.pt
        ├── collaborative/
        │   ├── best/
        │   │   ├── model_best.pt
        │   │   └── metadata.json
        │   └── history/
        └── hybrid/
            ├── best/
            │   ├── model_best.pt
            │   └── metadata.json
            └── history/
    """
    
    VALID_MODEL_TYPES = {'content_based', 'collaborative', 'hybrid'}
    
    def __init__(self,
                 checkpoint_dir: Union[str, Path],
                 model_type: str,
                 mode: str = 'min',
                 save_best_only: bool = True,
                 save_weights_only: bool = False,
                 max_checkpoints: int = 5,
                 verbose: bool = True):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Base directory for all checkpoints
            model_type: Type of model ('content_based', 'collaborative', or 'hybrid')
            mode: 'min' or 'max' for the monitored metric
            save_best_only: If True, only save when model improves
            save_weights_only: If True, only save model weights
            max_checkpoints: Maximum number of historical checkpoints to keep
            verbose: Whether to print messages about checkpointing
        """
        if model_type not in self.VALID_MODEL_TYPES:
            raise ValueError(f"model_type must be one of {self.VALID_MODEL_TYPES}")
        
        self.base_dir = Path(checkpoint_dir)
        self.model_type = model_type
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.max_checkpoints = max_checkpoints
        self.verbose = verbose
        
        # Setup directory structure
        self.model_dir = self.base_dir / model_type
        self.best_dir = self.model_dir / 'best'
        self.history_dir = self.model_dir / 'history'
        
        # Create directories
        self.best_dir.mkdir(parents=True, exist_ok=True)
        if not self.save_best_only:
            self.history_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.is_better = lambda a, b: a < b if mode == 'min' else a > b
        
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints if exceeding max_checkpoints."""
        if not self.save_best_only:
            checkpoints = sorted(self.history_dir.glob('checkpoint_epoch_*.pt'))
            while len(checkpoints) >= self.max_checkpoints:
                oldest = checkpoints.pop(0)
                oldest.unlink()
                if self.verbose:
                    print(f"Removed old checkpoint: {oldest}")
    
    def _save_checkpoint(self, 
                        filepath: Path, 
                        model_state: Dict[str, Any],
                        epoch: int,
                        value: float,
                        optimizer_state: Optional[Dict[str, Any]] = None,
                        scheduler_state: Optional[Dict[str, Any]] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save checkpoint with all relevant information."""
        checkpoint = {
            'epoch': epoch,
            'value': value,
            'timestamp': datetime.now().isoformat()
        }
        
        if self.save_weights_only:
            checkpoint['model_state_dict'] = model_state
        else:
            checkpoint.update({
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer_state,
                'scheduler_state_dict': scheduler_state
            })
        
        if metadata:
            checkpoint['metadata'] = metadata
            
        torch.save(checkpoint, filepath)
        
        if self.verbose:
            print(f"Saved checkpoint to {filepath}")
    
    def save_checkpoint(self,
                       model_state: Dict[str, Any],
                       value: float,
                       epoch: int,
                       optimizer_state: Optional[Dict[str, Any]] = None,
                       scheduler_state: Optional[Dict[str, Any]] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Save model checkpoint if conditions are met.
        
        Args:
            model_state: Model state dict
            value: Value to monitor for improvement
            epoch: Current epoch number
            optimizer_state: Optimizer state dict
            scheduler_state: Scheduler state dict
            metadata: Additional metadata to save
            
        Returns:
            bool: True if this was the best model so far
        """
        is_best = self.is_better(value, self.best_value)
        
        # Save checkpoint
        if is_best or not self.save_best_only:
            if is_best:
                self.best_value = value
                # Save best model
                best_model_path = self.best_dir / 'model_best.pt'
                self._save_checkpoint(
                    best_model_path,
                    model_state,
                    epoch,
                    value,
                    optimizer_state,
                    scheduler_state,
                    metadata
                )
                
                # Save metadata separately
                metadata_path = self.best_dir / 'metadata.json'
                with open(metadata_path, 'w') as f:
                    json.dump({
                        'best_epoch': epoch,
                        'best_value': float(value),
                        'timestamp': datetime.now().isoformat(),
                        'model_type': self.model_type,
                        **(metadata or {})
                    }, f, indent=4)
            
            if not self.save_best_only:
                # Save regular checkpoint
                checkpoint_path = self.history_dir / f'checkpoint_epoch_{epoch:03d}.pt'
                self._save_checkpoint(
                    checkpoint_path,
                    model_state,
                    epoch,
                    value,
                    optimizer_state,
                    scheduler_state,
                    metadata
                )
                self._cleanup_old_checkpoints()
        
        return is_best

    @classmethod
    def load_best_model(cls,
                       checkpoint_dir: Union[str, Path],
                       model_type: str,
                       model_class: Type[nn.Module],
                       model_kwargs: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Load the best model of a specific type.
        
        Args:
            checkpoint_dir: Base checkpoint directory
            model_type: Type of model to load
            model_class: Model class to instantiate
            model_kwargs: Arguments for model initialization
            
        Returns:
            tuple: (loaded_model, checkpoint_data)
        """
        if model_type not in cls.VALID_MODEL_TYPES:
            raise ValueError(f"model_type must be one of {cls.VALID_MODEL_TYPES}")
        
        best_model_path = Path(checkpoint_dir) / model_type / 'best' / 'model_best.pt'
        
        if not best_model_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {best_model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(best_model_path)
        
        # Initialize model
        model = model_class(**model_kwargs)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, checkpoint

def fit(model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        num_epochs: int = 20,
        scheduler: Optional[Any] = None,
        metrics: Optional[Dict[str, callable]] = None,
        early_stopping: Optional[Dict[str, Any]] = None,
        checkpoint: Optional[Dict[str, Any]] = None) -> Dict[str, list]:
    """
    Enhanced training loop with early stopping and checkpointing.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: PyTorch optimizer
        criterion: Loss function
        device: Device to train on
        num_epochs: Maximum number of epochs
        scheduler: Learning rate scheduler
        metrics: Dictionary of metric functions
        early_stopping: Dict of early stopping parameters
        checkpoint: Dict of checkpointing parameters
        
    Returns:
        Dict containing training history
    """
    # Initialize early stopping and checkpointing
    early_stopper = None
    checkpointer = None
    
    if early_stopping:
        early_stopper = EarlyStopping(**early_stopping)
    
    if checkpoint:
        checkpointer = ModelCheckpoint(**checkpoint)
    
    history = {"train_loss": [], "val_loss": []}
    if metrics:
        for name in metrics:
            history[f"val_{name}"] = []
    
    for epoch in range(1, num_epochs + 1):
        # Training phase
        model.train()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validation phase
        model.eval()
        val_res = evaluate(model, val_loader, criterion, device, metrics)
        val_loss = val_res["loss"]
        
        # Update history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if metrics:
            for name in metrics:
                history[f"val_{name}"].append(val_res[name])
        
        # Logging
        msg = f"Epoch {epoch}/{num_epochs} — train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}"
        if metrics:
            for name in metrics:
                msg += f", val_{name}: {val_res[name]:.4f}"
        print(msg)
        
        # Model checkpointing
        if checkpointer:
            is_best = checkpointer.save_checkpoint(
                model_state=model.state_dict(),
                value=val_loss,
                epoch=epoch,
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict() if scheduler else None,
                metadata={'metrics': val_res}
            )
        
        # Early stopping check
        if early_stopper and early_stopper(val_loss, model.state_dict()):
            print(f"Early stopping triggered (no improvement in {early_stopper.patience} epochs)")
            # Load best model state
            model.load_state_dict(early_stopper.get_best_state())
            break
        
        # Scheduler step
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
    
    return history

def plot_training_history(history: Dict[str, List[float]], 
                         figsize: tuple = (12, 6),
                         save_path: str = None):
    """
    Plot training and validation metrics over epochs.
    
    Args:
        history (dict): Dictionary containing lists of metrics per epoch
                       (e.g., {'train_loss': [...], 'val_loss': [...], 'val_rmse': [...]}
        figsize (tuple): Figure size (width, height)
        save_path (str): If provided, save the plot to this path
    """
    # Set style
    sns.set_style("whitegrid")
    plt.figure(figsize=figsize)
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame(history)
    epochs = range(1, len(df) + 1)
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, df['train_loss'], 'b-', label='Training Loss', marker='o')
    plt.plot(epochs, df['val_loss'], 'r-', label='Validation Loss', marker='o')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot other metrics if they exist
    plt.subplot(1, 2, 2)
    metrics = [col for col in df.columns if col not in ['train_loss', 'val_loss']]
    for metric in metrics:
        plt.plot(epochs, df[metric], marker='o', label=metric)
    plt.title('Model Metrics Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()

def plot_learning_rate(history: Dict[str, List[float]], 
                      learning_rates: List[float],
                      figsize: tuple = (10, 5),
                      save_path: str = None):
    """
    Plot learning rate changes alongside loss.
    
    Args:
        history (dict): Training history dictionary
        learning_rates (list): List of learning rates used during training
        figsize (tuple): Figure size (width, height)
        save_path (str): If provided, save the plot to this path
    """
    sns.set_style("whitegrid")
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Plot loss
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'b--', label='Validation Loss')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    # Plot learning rate
    ax2 = ax1.twinx()
    ax2.set_ylabel('Learning Rate', color='tab:red')
    ax2.plot(epochs, learning_rates, 'r-', label='Learning Rate')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.title('Training Loss and Learning Rate Over Epochs')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()

def create_training_summary(history: Dict[str, List[float]], 
                          model_name: str,
                          save_path: str = None) -> pd.DataFrame:
    """
    Create a summary DataFrame of training metrics and optionally save it.
    
    Args:
        history (dict): Training history dictionary
        model_name (str): Name of the model
        save_path (str): If provided, save the summary to this path
        
    Returns:
        pd.DataFrame: Summary of training metrics
    """
    # Create summary dictionary
    summary = {
        'Model': model_name,
        'Final Train Loss': history['train_loss'][-1],
        'Final Val Loss': history['val_loss'][-1],
        'Best Val Loss': min(history['val_loss']),
        'Best Epoch': history['val_loss'].index(min(history['val_loss'])) + 1,
        'Total Epochs': len(history['train_loss'])
    }
    
    # Add other metrics if they exist
    for key in history:
        if key not in ['train_loss', 'val_loss']:
            summary[f'Final {key}'] = history[key][-1]
            summary[f'Best {key}'] = min(history[key])
    
    # Create DataFrame
    df = pd.DataFrame([summary])
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Summary saved to {save_path}")
    
    return df

"""
PyTorch Dataset classes for the HotelRec recommendation system.
Handles both content-based and collaborative filtering approaches.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import pandas as pd
import numpy as np
from pathlib import Path
import json
from collections import defaultdict
from typing import List, Optional, Union, Iterator, Dict, Tuple, Callable
from tqdm import tqdm

# TODO: update both the datasets based on actual data in HotelRec.txt

class ChunkedDataset(IterableDataset):
    """
    Base class for chunked data loading.
    Implements memory-efficient data loading by reading chunks on-demand.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        chunk_size: int = 10000,
        shuffle: bool = True,
        transform: Optional[Callable] = None
    ):
        """
        Initialize chunked dataset.
        
        Args:
            data_dir: Directory containing JSONL chunks
            chunk_size: Number of samples to load in memory at once
            shuffle: Whether to shuffle samples within chunks
            transform: Optional transform to apply to features
        """
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.transform = transform
        
        # Get all chunk files and their metadata
        self.chunk_files = sorted(self.data_dir.glob("reviews_chunk_*.jsonl"))
        self.meta_files = sorted(self.data_dir.glob("reviews_chunk_*.meta.json"))
        
        if not self.chunk_files:
            raise FileNotFoundError(f"No chunk files found in {data_dir}")
        
        # Load metadata for all chunks
        self.chunk_metadata = []
        total_samples = 0
        for meta_file in self.meta_files:
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
                metadata['start_idx'] = total_samples
                total_samples += metadata['n_reviews']
                metadata['end_idx'] = total_samples
                self.chunk_metadata.append(metadata)
        
        self.total_samples = total_samples
        print(f"Found {len(self.chunk_files):,} chunks with {total_samples:,} total samples")
    
    def _load_chunk(self, chunk_file: Path) -> pd.DataFrame:
        """Load a single chunk file."""
        return pd.read_json(chunk_file, lines=True)
    
    def _prepare_chunk(self, chunk_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare a chunk of data for training.
        To be implemented by subclasses.
        """
        raise NotImplementedError
    
    def __iter__(self) -> Iterator:
        """
        Create an iterator over the chunks.
        Implements memory-efficient iteration by loading one chunk at a time.
        """
        worker_info = torch.utils.data.get_worker_info()
        
        # Divide chunks among workers if using multiple workers
        if worker_info is not None:
            chunks_per_worker = len(self.chunk_files) // worker_info.num_workers
            worker_id = worker_info.id
            start_chunk = worker_id * chunks_per_worker
            end_chunk = start_chunk + chunks_per_worker if worker_id < worker_info.num_workers - 1 else len(self.chunk_files)
            chunk_files = self.chunk_files[start_chunk:end_chunk]
        else:
            chunk_files = self.chunk_files
        
        # Iterate over assigned chunks
        for chunk_file in chunk_files:
            # Load and prepare chunk
            chunk_df = self._load_chunk(chunk_file)
            prepared_chunk = self._prepare_chunk(chunk_df)
            
            # Create iterator for the chunk
            indices = range(len(prepared_chunk))
            if self.shuffle:
                indices = torch.randperm(len(prepared_chunk)).tolist()
            
            # Yield samples from the chunk
            for idx in indices:
                yield self._get_item(prepared_chunk, idx)
    
    def _get_item(self, chunk_df: pd.DataFrame, idx: int) -> tuple:
        """
        Get a single item from a chunk.
        To be implemented by subclasses.
        """
        raise NotImplementedError

class ChunkedHotelReviewDataset(ChunkedDataset):
    """
    Memory-efficient dataset for content-based model.
    Loads and processes data in chunks.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Load first chunk to get feature information
        first_chunk = self._load_chunk(self.chunk_files[0])
        self.rating_cols = [col for col in first_chunk.columns if col.startswith('rating_')]
        self.text_cols = ['processed_text', 'processed_title'] if 'processed_text' in first_chunk.columns else ['text', 'title']
        
        # Initialize feature statistics
        self._init_feature_stats()
    
    def _init_feature_stats(self):
        """Initialize running statistics for feature normalization."""
        print("\nCalculating feature statistics...")
        
        # Initialize accumulators
        self.feature_sums = defaultdict(float)
        self.feature_sq_sums = defaultdict(float)
        total_samples = 0
        
        # Calculate statistics in chunks
        for chunk_file in tqdm(self.chunk_files, desc="Processing chunks"):
            chunk_df = self._load_chunk(chunk_file)
            chunk_df = self._prepare_features(chunk_df, update_stats=True)
            total_samples += len(chunk_df)
        
        # Calculate means and stds
        self.feature_means = {
            col: self.feature_sums[col] / total_samples
            for col in self.feature_cols
        }
        
        self.feature_stds = {
            col: np.sqrt(
                self.feature_sq_sums[col] / total_samples -
                (self.feature_sums[col] / total_samples) ** 2
            )
            for col in self.feature_cols
        }
        
        print("Feature statistics calculated")
    
    def _prepare_features(self, df: pd.DataFrame, update_stats: bool = False) -> pd.DataFrame:
        """Prepare numerical features for the model."""
        # Calculate text lengths
        for col in self.text_cols:
            df[f'{col}_length'] = df[col].str.len()
        
        # Define feature columns
        self.feature_cols = [
            'rating_sleep_quality', 'rating_value', 'rating_rooms',
            'rating_service', 'rating_cleanliness', 'rating_location'
        ]
        self.feature_cols.extend([f'{col}_length' for col in self.text_cols])
        
        # Fill missing values
        for col in self.feature_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
            else:
                df[col] = 0.0
        
        if update_stats:
            # Update running statistics
            for col in self.feature_cols:
                self.feature_sums[col] += df[col].sum()
                self.feature_sq_sums[col] += (df[col] ** 2).sum()
        else:
            # Normalize features
            for col in self.feature_cols:
                df[col] = (df[col] - self.feature_means[col]) / self.feature_stds[col]
        
        return df
    
    def _prepare_chunk(self, chunk_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare a chunk of data for training."""
        return self._prepare_features(chunk_df)
    
    def _get_item(self, chunk_df: pd.DataFrame, idx: int) -> tuple:
        """Get a single item from a chunk."""
        row = chunk_df.iloc[idx]
        
        # Get numerical features
        features = torch.tensor(
            row[self.feature_cols].values,
            dtype=torch.float32
        )
        
        # Get target rating
        rating = torch.tensor(row["rating"], dtype=torch.float32)
        
        # Apply transforms if any
        if self.transform:
            features = self.transform(features)
            
        return features, rating
    
    def get_feature_dims(self) -> int:
        """Return the number of feature dimensions."""
        return len(self.feature_cols)

class ChunkedCFMatrixDataset(ChunkedDataset):
    """
    Memory-efficient dataset for collaborative filtering.
    Loads and processes data in chunks while maintaining consistent user/hotel mappings.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Create global user and hotel mappings
        print("\nCreating user and hotel mappings...")
        self.user_to_idx = {}
        self.hotel_to_idx = {}
        
        for chunk_file in tqdm(self.chunk_files, desc="Processing chunks"):
            chunk_df = self._load_chunk(chunk_file)
            
            # Update mappings
            for user in chunk_df['author'].unique():
                if user not in self.user_to_idx:
                    self.user_to_idx[user] = len(self.user_to_idx)
            
            for hotel in chunk_df['hotel_url'].unique():
                if hotel not in self.hotel_to_idx:
                    self.hotel_to_idx[hotel] = len(self.hotel_to_idx)
        
        self.num_users = len(self.user_to_idx)
        self.num_hotels = len(self.hotel_to_idx)
        
        print(f"Found {self.num_users:,} users and {self.num_hotels:,} hotels")
    
    def _prepare_chunk(self, chunk_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare a chunk of data for training."""
        chunk_df['user_idx'] = chunk_df['author'].map(self.user_to_idx)
        chunk_df['hotel_idx'] = chunk_df['hotel_url'].map(self.hotel_to_idx)
        return chunk_df
    
    def _get_item(self, chunk_df: pd.DataFrame, idx: int) -> tuple:
        """Get a single item from a chunk."""
        row = chunk_df.iloc[idx]
        return (
            torch.tensor(row["user_idx"], dtype=torch.long),
            torch.tensor(row["hotel_idx"], dtype=torch.long),
            torch.tensor(row["rating"], dtype=torch.float32),
        )
    
    def get_dims(self) -> tuple[int, int]:
        """Return the dimensions of the user-item matrices."""
        return self.num_users, self.num_hotels

def get_content_loader(data_path, batch_size=64, shuffle=True, num_workers=4):
    """Create a DataLoader for the content-based model."""
    ds = ChunkedHotelReviewDataset(data_path)
    return DataLoader(
        ds, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=num_workers, 
        pin_memory=True
    )

def get_cf_loader(data_path, batch_size=256, shuffle=True, num_workers=4):
    """Create a DataLoader for the collaborative filtering model."""
    ds = ChunkedCFMatrixDataset(data_path)
    return DataLoader(
        ds, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=num_workers, 
        pin_memory=True
    )

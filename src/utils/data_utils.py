"""
Shared utilities for data preparation and sampling.
"""

from pathlib import Path
from typing import Optional
import json
from tqdm import tqdm

def count_lines(file_path: Path) -> int:
    """
    Count number of lines in file efficiently.
    
    Args:
        file_path (Path): Path to the file
        
    Returns:
        int: Number of lines in the file
    """
    print("\nCounting total lines in file...")
    lines = 0
    with open(file_path, 'rb') as f:
        buf_size = 1024 * 1024
        read_f = f.raw.read
        buf = read_f(buf_size)
        while buf:
            lines += buf.count(b'\n')
            buf = read_f(buf_size)
    return lines

def validate_directories(
    project_root: Path,
    raw_dir: Path,
    processed_dir: Path,
    force_clean: bool = False
) -> None:
    """
    Validate and create necessary directories.
    
    Args:
        project_root (Path): Project root directory
        raw_dir (Path): Raw data directory
        processed_dir (Path): Processed data directory
        force_clean (bool): Whether to clean processed directory
    """
    # Ensure directories exist
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    if force_clean and processed_dir.exists():
        import shutil
        shutil.rmtree(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

def save_chunk_metadata(
    chunk_data: dict,
    processed_dir: Path,
    chunk_idx: int
) -> None:
    """
    Save metadata for a processed chunk.
    
    Args:
        chunk_data (dict): Chunk processing results
        processed_dir (Path): Directory to save metadata
        chunk_idx (int): Chunk index
    """
    metadata = {
        'chunk_index': chunk_idx,
        'n_reviews': chunk_data['n_reviews'],
        'file_size': chunk_data['file_size'],
        'columns': chunk_data['columns'],
        'dtypes': chunk_data['dtypes'],
        'timestamp': chunk_data['timestamp']
    }
    
    meta_file = processed_dir / f"reviews_chunk_{chunk_idx:04d}.meta.json"
    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2)

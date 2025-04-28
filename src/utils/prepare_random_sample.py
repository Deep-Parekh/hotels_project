"""
Script for preparing a random sample from the HotelRec dataset.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional
import numpy as np
from tqdm import tqdm
import json
import random
from datetime import datetime
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.data import DataLoader
from src.utils.data_utils import count_lines, validate_directories, save_chunk_metadata

def process_chunk(reviews: list, loader: DataLoader, chunk_idx: int, output_dir: Path, include_tfidf: bool) -> None:
    """Process and save a chunk of reviews."""
    # Convert reviews to DataFrame
    df_raw = pd.DataFrame(reviews)
    
    # Process the reviews
    df_proc = loader.preprocessor.process_reviews(df_raw, include_tfidf=include_tfidf)
    
    # Remove all unnecessary fields before saving
    drop_cols = [
        'text', 'title', 'processed_text', 'processed_title', 'property_dict',
        'hotel_url', 'author'
    ]
    df_proc = df_proc.drop(columns=drop_cols)
    
    # Save as JSONL
    chunk_file = output_dir / f"reviews_chunk_{chunk_idx:04d}.jsonl"
    df_proc.to_json(chunk_file, orient='records', lines=True)
    
    # Prepare metadata
    chunk_data = {
        'n_reviews': len(df_proc),
        'file_size': chunk_file.stat().st_size,
        'columns': list(df_proc.columns),
        'dtypes': df_proc.dtypes.astype(str).to_dict(),
        'timestamp': datetime.now().isoformat()
    }
    
    # Save metadata
    save_chunk_metadata(chunk_data, output_dir, chunk_idx)
    
    print(f"Chunk {chunk_idx:04d}: Processed {len(df_proc):,} reviews → {chunk_file.name}")

def prepare_random_sample(
    raw_file: str = "HotelRec.txt",
    n_samples: int = 2_000_000,
    chunk_size: int = 100_000,
    include_tfidf: bool = False,
    force_clean: bool = False,
    random_seed: Optional[int] = 42
) -> None:
    """
    Prepare a random sample from the HotelRec dataset for model training.
    
    Args:
        raw_file (str): Name of the raw data file
        n_samples (int): Number of reviews to sample
        chunk_size (int): Number of reviews per chunk
        include_tfidf (bool): Whether to include TF-IDF features
        force_clean (bool): Whether to clean processed directory before starting
        random_seed (int): Random seed for reproducibility
    """
    # Setup paths
    data_dir = project_root / "data"
    raw_dir = data_dir / "raw" / "Full_HotelRec"    
    random_processed_dir = data_dir / "random_processed"
    
    print("\n" + "="*60)
    print("Starting Random Sample Preparation")
    print("="*60)
    print(f"Configuration:")
    print(f"- Raw file: {raw_file}")
    print(f"- Number of samples: {n_samples:,}")
    print(f"- Chunk size: {chunk_size:,}")
    print(f"- Include TF-IDF: {include_tfidf}")
    print(f"- Random seed: {random_seed}")
    print("="*60 + "\n")
    
    # Set random seed
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    # Validate directories
    validate_directories(
        project_root=project_root,
        raw_dir=raw_dir,
        processed_dir=random_processed_dir,
        force_clean=force_clean
    )
    
    # Check if raw file exists
    raw_file_path = raw_dir / raw_file
    if not raw_file_path.exists():
        print(f"\nError: Raw data file not found: {raw_file_path}")
        print("Please place the HotelRec.txt file in the data/raw directory.")
        sys.exit(1)
    
    try:
        # Count total lines
        total_lines = count_lines(raw_file_path)
        print(f"Total reviews available: {total_lines:,}")
        
        if n_samples > total_lines:
            print(f"Warning: Requested samples ({n_samples:,}) exceeds total reviews.")
            print(f"Will use all available reviews instead.")
            n_samples = total_lines
        
        # Generate random indices
        print("\nGenerating random sample indices...")
        sample_indices = set(np.random.choice(
            total_lines, 
            size=n_samples, 
            replace=False
        ))
        
        # Initialize loader
        loader = DataLoader(
            data_dir=raw_dir,
            chunk_size=chunk_size,
            processed_dir=random_processed_dir
        )
        
        # Process the samples in chunks
        print("\nProcessing random samples...")
        reviews = []
        chunk_idx = 0
        
        with tqdm(total=n_samples, desc="Reviews sampled") as pbar:
            with open(raw_file_path, 'r', encoding='utf-8') as f:
                for line_idx, line in enumerate(f):
                    if line_idx in sample_indices:
                        try:
                            review = json.loads(line)
                            reviews.append(review)
                            pbar.update(1)
                            
                            # Process chunk when it reaches chunk_size
                            if len(reviews) >= chunk_size:
                                process_chunk(
                                    reviews,
                                    loader,
                                    chunk_idx,
                                    random_processed_dir,
                                    include_tfidf
                                )
                                chunk_idx += 1
                                reviews = []
                        except json.JSONDecodeError:
                            print(f"\nWarning: Skipping malformed JSON at line {line_idx}")
                
                # Process remaining reviews
                if reviews:
                    process_chunk(
                        reviews,
                        loader,
                        chunk_idx,
                        random_processed_dir,
                        include_tfidf
                    )
        
        # Get final statistics
        processed_files = list(random_processed_dir.glob("reviews_chunk_*.jsonl"))
        total_size = sum(f.stat().st_size for f in processed_files)
        
        print("\n" + "="*60)
        print("Random Sample Processing Complete")
        print("="*60)
        print(f"Processed data information:")
        print(f"- Number of chunks: {len(processed_files)}")
        print(f"- Total size: {total_size / (1024**3):.2f} GB")
        print(f"- Samples processed: {n_samples:,}")
        print(f"\nProcessed data is available in: {random_processed_dir}")
        print("="*60)
        
        # After all chunks are processed, save the encodings
        loader.save_encodings()
        
    except Exception as e:
        print(f"\nError during sample preparation: {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Prepare random sample from HotelRec dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--raw-file",
        default="HotelRec.txt",
        help="Name of the raw data file"
    )
    
    parser.add_argument(
        "--n-samples",
        type=int,
        default=2_000_000,
        help="Number of reviews to sample"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Number of reviews per chunk"
    )
    
    parser.add_argument(
        "--include-tfidf",
        action="store_true",
        help="Include TF-IDF features in processed data"
    )
    
    parser.add_argument(
        "--force-clean",
        action="store_true",
        help="Clean processed directory before starting"
    )
    
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    prepare_random_sample(
        raw_file=args.raw_file,
        n_samples=args.n_samples,
        chunk_size=args.chunk_size,
        include_tfidf=args.include_tfidf,
        force_clean=args.force_clean,
        random_seed=args.random_seed
    )

if __name__ == "__main__":
    main()

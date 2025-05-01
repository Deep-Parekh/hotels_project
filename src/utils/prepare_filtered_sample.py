"""
Script for preparing a filtered subset from the HotelRec dataset.
Filters reviews to only include hotels with >5 reviews and authors with >5 reviews.
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
import pandas as pd
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.data import DataLoader
from src.utils.data_utils import count_lines, validate_directories, save_chunk_metadata

class ReviewCounter:
    """Class to count reviews per hotel and author."""
    
    def __init__(self):
        self.hotel_counts: Dict[str, int] = defaultdict(int)
        self.author_counts: Dict[str, int] = defaultdict(int)
        self.total_reviews: int = 0
    
    def update(self, review: dict) -> None:
        """Update counts for a single review."""
        hotel = review["hotel_url"]
        author = review["author"]
        self.hotel_counts[hotel] += 1
        self.author_counts[author] += 1
        self.total_reviews += 1
    
    def get_qualifying_entities(self, min_reviews: int = 5) -> Tuple[Set[str], Set[str]]:
        """Get sets of hotels and authors with at least min_reviews."""
        qualifying_hotels = {
            hotel for hotel, count in self.hotel_counts.items() 
            if count >= min_reviews
        }
        qualifying_authors = {
            author for author, count in self.author_counts.items() 
            if count >= min_reviews
        }
        return qualifying_hotels, qualifying_authors
    
    def get_statistics(self) -> Dict:
        """Get summary statistics about the dataset."""
        return {
            "total_reviews": self.total_reviews,
            "unique_hotels": len(self.hotel_counts),
            "unique_authors": len(self.author_counts),
            "hotel_stats": {
                "min_reviews": min(self.hotel_counts.values()),
                "max_reviews": max(self.hotel_counts.values()),
                "avg_reviews": np.mean(list(self.hotel_counts.values())),
                "median_reviews": np.median(list(self.hotel_counts.values()))
            },
            "author_stats": {
                "min_reviews": min(self.author_counts.values()),
                "max_reviews": max(self.author_counts.values()),
                "avg_reviews": np.mean(list(self.author_counts.values())),
                "median_reviews": np.median(list(self.author_counts.values()))
            }
        }

def count_reviews(raw_file: Path, progress: bool = True) -> ReviewCounter:
    """
    Count reviews per hotel and author in the raw data file.
    
    Args:
        raw_file: Path to the raw data file
        progress: Whether to show progress bar
        
    Returns:
        ReviewCounter object with counts
    """
    counter = ReviewCounter()
    total_lines = count_lines(raw_file)
    
    with tqdm(total=total_lines, desc="Counting reviews", disable=not progress) as pbar:
        with open(raw_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    review = json.loads(line)
                    counter.update(review)
                except json.JSONDecodeError:
                    continue
                pbar.update(1)
    
    return counter

def process_filtered_chunk(
    reviews: List[dict],
    loader: DataLoader,
    chunk_idx: int,
    output_dir: Path,
    include_tfidf: bool
) -> None:
    """
    Process and save a chunk of filtered reviews.
    
    Args:
        reviews: List of review dictionaries
        loader: DataLoader instance
        chunk_idx: Index of the current chunk
        output_dir: Directory to save processed chunks
        include_tfidf: Whether to include TF-IDF features
    """
    # Convert reviews to DataFrame
    df_raw = pd.DataFrame(reviews)
    
    # Process the reviews
    df_proc = loader.preprocessor.process_reviews(df_raw, include_tfidf=include_tfidf)
    
    # Remove unnecessary fields
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

def prepare_filtered_sample(
    raw_file: str = "HotelRec.txt",
    target_size: int = 200_000,
    chunk_size: int = 200_000,
    min_reviews: int = 5,
    include_tfidf: bool = False,
    force_clean: bool = False,
    random_seed: Optional[int] = 42
) -> None:
    """
    Prepare a filtered subset from the HotelRec dataset.
    
    Args:
        raw_file: Name of the raw data file
        target_size: Target number of reviews to include
        chunk_size: Number of reviews per chunk
        min_reviews: Minimum number of reviews required for hotels and authors
        include_tfidf: Whether to include TF-IDF features
        force_clean: Whether to clean processed directory before starting
        random_seed: Random seed for reproducibility
    """
    # Setup paths
    data_dir = project_root / "data"
    raw_dir = data_dir / "raw" / "Full_HotelRec"    
    filtered_processed_dir = data_dir / "filtered_processed"
    
    print("\n" + "="*60)
    print("Starting Filtered Sample Preparation")
    print("="*60)
    print(f"Configuration:")
    print(f"- Raw file: {raw_file}")
    print(f"- Target size: {target_size:,}")
    print(f"- Chunk size: {chunk_size:,}")
    print(f"- Min reviews: {min_reviews}")
    print(f"- Include TF-IDF: {include_tfidf}")
    print(f"- Random seed: {random_seed}")
    print("="*60 + "\n")
    
    # Set random seed
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Validate directories
    validate_directories(
        project_root=project_root,
        raw_dir=raw_dir,
        processed_dir=filtered_processed_dir,
        force_clean=force_clean
    )
    
    # Check if raw file exists
    raw_file_path = raw_dir / raw_file
    if not raw_file_path.exists():
        print(f"\nError: Raw data file not found: {raw_file_path}")
        print("Please place the HotelRec.txt file in the data/raw directory.")
        sys.exit(1)
    
    try:
        # First pass: Count reviews
        print("\nCounting reviews per hotel and author...")
        counter = count_reviews(raw_file_path)
        stats = counter.get_statistics()
        
        # Get qualifying hotels and authors
        qualifying_hotels, qualifying_authors = counter.get_qualifying_entities(min_reviews)
        
        print("\nDataset Statistics:")
        print(f"Total Reviews: {stats['total_reviews']:,}")
        print(f"Unique Hotels: {stats['unique_hotels']:,}")
        print(f"Unique Authors: {stats['unique_authors']:,}")
        print(f"\nQualifying Hotels: {len(qualifying_hotels):,}")
        print(f"Qualifying Authors: {len(qualifying_authors):,}")
        
        # Initialize loader
        loader = DataLoader(
            data_dir=raw_dir,
            chunk_size=chunk_size,
            processed_dir=filtered_processed_dir
        )
        
        # Second pass: Process qualifying reviews
        print("\nProcessing qualifying reviews...")
        reviews = []
        chunk_idx = 0
        
        with tqdm(total=target_size, desc="Reviews processed") as pbar:
            with open(raw_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        review = json.loads(line)
                        if (review["hotel_url"] in qualifying_hotels and 
                            review["author"] in qualifying_authors):
                            reviews.append(review)
                            pbar.update(1)
                            
                            # Process chunk when it reaches chunk_size
                            if len(reviews) >= chunk_size:
                                process_filtered_chunk(
                                    reviews,
                                    loader,
                                    chunk_idx,
                                    filtered_processed_dir,
                                    include_tfidf
                                )
                                chunk_idx += 1
                                reviews = []
                                
                                if len(reviews) >= target_size:
                                    break
                    except json.JSONDecodeError:
                        continue
        
        # Process remaining reviews
        if reviews:
            process_filtered_chunk(
                reviews,
                loader,
                chunk_idx,
                filtered_processed_dir,
                include_tfidf
            )
        
        # Save final statistics
        stats_path = filtered_processed_dir / "dataset_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print("\n" + "="*60)
        print("Filtered Sample Processing Complete")
        print("="*60)
        print(f"Processed data information:")
        print(f"- Number of chunks: {chunk_idx + 1}")
        print(f"- Total reviews processed: {len(reviews):,}")
        print(f"\nProcessed data is available in: {filtered_processed_dir}")
        print("="*60)
        
        # After all chunks are processed, save the encodings
        loader.save_encodings()
        
    except Exception as e:
        print(f"\nError during sample preparation: {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Prepare filtered sample from HotelRec dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--raw-file",
        default="HotelRec.txt",
        help="Name of the raw data file"
    )
    
    parser.add_argument(
        "--target-size",
        type=int,
        default=200_000,
        help="Target number of reviews to include"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200_000,
        help="Number of reviews per chunk"
    )
    
    parser.add_argument(
        "--min-reviews",
        type=int,
        default=5,
        help="Minimum number of reviews required for hotels and authors"
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
    
    prepare_filtered_sample(
        raw_file=args.raw_file,
        target_size=args.target_size,
        chunk_size=args.chunk_size,
        min_reviews=args.min_reviews,
        include_tfidf=args.include_tfidf,
        force_clean=args.force_clean,
        random_seed=args.random_seed
    )

if __name__ == "__main__":
    main() 
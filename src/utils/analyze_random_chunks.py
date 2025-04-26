"""
Script to analyze statistics from randomly sampled hotel review chunks.
Provides information about unique hotels, authors, and reviews per hotel.
"""

import os
from pathlib import Path
import json
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Set, Tuple
import argparse
from concurrent.futures import ProcessPoolExecutor
import psutil
import warnings

class ChunkAnalyzer:
    def __init__(self, data_dir: str, num_workers: int = 4, chunk_size: int = 5000):
        self.data_dir = Path(data_dir)
        self.num_workers = num_workers
        self.chunk_size = chunk_size
        self.chunk_files = sorted(self.data_dir.glob("reviews_chunk_*.jsonl"))
        
        if not self.chunk_files:
            raise FileNotFoundError(f"No chunk files found in {data_dir}")
        
        # Check available memory
        available_memory = psutil.virtual_memory().available / (1024 ** 3)  # in GB
        
        print(f"\nMemory Analysis:")
        print(f"Available Memory: {available_memory:.1f} GB")
        print(f"Number of chunks: {len(self.chunk_files)}")
        print(f"Workers: {num_workers}")
        
        if available_memory < 4:
            warnings.warn(f"\nWarning: Available memory ({available_memory:.1f}GB) is less than "
                        f"estimated required memory (4GB).\n"
                        f"Consider reducing num_workers or increasing chunk_size.")
        
        # Initialize counters
        self.unique_hotels: Set[str] = set()
        self.unique_authors: Set[str] = set()
        self.reviews_per_hotel: Dict[str, int] = defaultdict(int)
        self.reviews_per_author: Dict[str, int] = defaultdict(int)
        self.total_reviews = 0

    def process_chunk(self, chunk_file: Path) -> Tuple[Set[str], Set[str], Dict[str, int], Dict[str, int], int]:
        """Process a single chunk file and return its statistics."""
        chunk_hotels = set()
        chunk_authors = set()
        chunk_hotel_counts = defaultdict(int)
        chunk_author_counts = defaultdict(int)
        total_rows = 0
        
        try:
            # Read chunk in smaller chunks to manage memory
            for chunk_df in pd.read_json(chunk_file, lines=True, chunksize=self.chunk_size):
                # Process only required columns
                chunk_df = chunk_df[['hotel_url', 'author']]
                
                # Update hotels
                chunk_hotels.update(chunk_df['hotel_url'].unique())
                for hotel, count in chunk_df['hotel_url'].value_counts().items():
                    chunk_hotel_counts[hotel] += count
                
                # Update authors
                chunk_authors.update(chunk_df['author'].unique())
                for author, count in chunk_df['author'].value_counts().items():
                    chunk_author_counts[author] += count
                
                total_rows += len(chunk_df)
                
                # Clear memory
                del chunk_df
                
        except Exception as e:
            print(f"Error processing {chunk_file}: {str(e)}")
            raise
            
        return (
            chunk_hotels, 
            chunk_authors, 
            dict(chunk_hotel_counts),
            dict(chunk_author_counts),
            total_rows
        )
    
    def merge_chunk_results(self, results: List[Tuple[Set[str], Set[str], Dict[str, int], Dict[str, int], int]]):
        """Merge results from all chunks."""
        for (hotels, authors, hotel_counts, author_counts, chunk_size) in results:
            self.unique_hotels.update(hotels)
            self.unique_authors.update(authors)
            
            # Update review counts
            for hotel, count in hotel_counts.items():
                self.reviews_per_hotel[hotel] += count
            for author, count in author_counts.items():
                self.reviews_per_author[author] += count
            
            self.total_reviews += chunk_size
    
    def analyze(self):
        """Analyze all chunks and compute statistics."""
        print("Starting chunk analysis...")
        
        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self.process_chunk, chunk_file) 
                      for chunk_file in self.chunk_files]
            
            # Collect results with progress bar
            results = []
            for future in tqdm(futures, total=len(futures), desc="Processing chunks"):
                results.append(future.result())
        
        # Merge results from all chunks
        self.merge_chunk_results(results)
        
        # Calculate statistics
        reviews_per_hotel_array = np.array(list(self.reviews_per_hotel.values()))
        reviews_per_author_array = np.array(list(self.reviews_per_author.values()))
        
        stats = {
            'total_reviews': self.total_reviews,
            'unique_hotels': len(self.unique_hotels),
            'unique_authors': len(self.unique_authors),
            'reviews_per_hotel': {
                'mean': reviews_per_hotel_array.mean(),
                'median': np.median(reviews_per_hotel_array),
                'min': reviews_per_hotel_array.min(),
                'max': reviews_per_hotel_array.max(),
                'std': reviews_per_hotel_array.std()
            },
            'reviews_per_author': {
                'mean': reviews_per_author_array.mean(),
                'median': np.median(reviews_per_author_array),
                'min': reviews_per_author_array.min(),
                'max': reviews_per_author_array.max(),
                'std': reviews_per_author_array.std()
            },
            'top_hotels': dict(sorted(self.reviews_per_hotel.items(), 
                                    key=lambda x: x[1], 
                                    reverse=True)[:10]),
            'top_authors': dict(sorted(self.reviews_per_author.items(), 
                                     key=lambda x: x[1], 
                                     reverse=True)[:10])
        }
        
        return stats
    
    def save_statistics(self, stats: Dict, output_file: str):
        """Save statistics to a JSON file."""
        # Convert all numpy types to native Python types
        def convert(obj):
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            else:
                return obj

        stats = convert(stats)

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Statistics saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Analyze hotel review chunks statistics')
    parser.add_argument('--data-dir', type=str, default='data/random_processed',
                      help='Directory containing the chunk files')
    parser.add_argument('--output-file', type=str, default='data/statistics/chunk_statistics.json',
                      help='Output file for statistics')
    parser.add_argument('--num-workers', type=int, default=4,
                      help='Number of worker processes')
    parser.add_argument('--chunk-size', type=int, default=5000,
                      help='Number of rows to process at once')
    args = parser.parse_args()
    
    # Create analyzer and process chunks
    analyzer = ChunkAnalyzer(args.data_dir, args.num_workers, args.chunk_size)
    stats = analyzer.analyze()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total Reviews: {stats['total_reviews']:,}")
    print(f"Unique Hotels: {stats['unique_hotels']:,}")
    print(f"Unique Authors: {stats['unique_authors']:,}")
    
    print("\nReviews per Hotel:")
    for metric, value in stats['reviews_per_hotel'].items():
        print(f"  {metric}: {value:.2f}")
    
    print("\nReviews per Author:")
    for metric, value in stats['reviews_per_author'].items():
        print(f"  {metric}: {value:.2f}")
    
    print("\nTop 10 Hotels by Review Count:")
    for hotel, count in stats['top_hotels'].items():
        print(f"  {hotel}: {count:,} reviews")
    
    print("\nTop 10 Authors by Review Count:")
    for author, count in stats['top_authors'].items():
        print(f"  {author}: {count:,} reviews")
    
    # Save statistics
    analyzer.save_statistics(stats, args.output_file)

if __name__ == '__main__':
    main()

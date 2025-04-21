"""
Data loading module specifically designed for HotelRec dataset.
Each line in the dataset is a JSON object containing hotel reviews with ratings and metadata.
"""

import json
from pathlib import Path
from typing import List, Dict, Union, Optional, Iterator
import os

import pandas as pd
import numpy as np
from tqdm import tqdm

from .preprocessor import DataPreprocessor


class DataLoader:
    """
    Specialized loader for HotelRec dataset that handles its specific JSON structure.
    
    The dataset contains reviews with the following structure:
    {
        "hotel_url": str,  # Unique identifier for the hotel
        "author": str,     # Username of the reviewer
        "date": str,       # ISO format date
        "rating": float,   # Overall rating (1-5)
        "title": str,      # Review title
        "text": str,       # Review text
        "property_dict": {  # Sub-ratings
            "sleep quality": float,
            "value": float,
            "rooms": float,
            "service": float,
            "cleanliness": float,
            "location": float
        }
    }
    """
    
    # Define expected columns and their types
    EXPECTED_COLUMNS = {
        'hotel_url': str,
        'author': str,
        'date': 'datetime64[ns]',
        'rating': float,
        'title': str,
        'text': str
    }
    
    # Sub-ratings in property_dict
    PROPERTY_RATINGS = [
        'sleep quality',
        'value',
        'rooms',
        'service',
        'cleanliness',
        'location'
    ]
    
    def __init__(self, data_path: Union[str, Path], chunk_size: int = 100000):
        """
        Initialize the DataLoader.
        
        Args:
            data_path (Union[str, Path]): Path to the data directory
            chunk_size (int, optional): Number of reviews per chunk. Defaults to 100000.
        """
        self.data_path = Path(data_path)
        self.chunk_size = chunk_size
        self.preprocessor = DataPreprocessor()  # Initialize the preprocessor
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_path}")
    
    def process_large_file(self, 
                          input_file: str,
                          chunk_size: Optional[int] = None,
                          include_tfidf: bool = False) -> None:
        """
        Process the large HotelRec.txt file into smaller parquet files.
        
        Args:
            input_file (str): Name of the input file (e.g., 'HotelRec.txt')
            chunk_size (Optional[int]): Override default chunk size
            include_tfidf (bool): Whether to include TF-IDF features
        """
        input_path = self.data_path / input_file
        # Change output path to be in data/processed
        output_path = self.data_path.parent / "processed"
        chunk_size = chunk_size or self.chunk_size

        # Validate input file exists
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Create output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)

        # Get total file size for progress bar
        total_size = os.path.getsize(input_path)
        
        print("\nProcessing HotelRec dataset:")
        print(f"Input file: {input_path}")
        print(f"Output directory: {output_path}")
        print(f"Chunk size: {chunk_size:,} reviews")
        
        chunk_number = 0
        reviews = []
        stats = {
            'processed': 0,
            'invalid': 0,
            'total_size': 0
        }
        
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Reading") as pbar:
            with open(input_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        # Parse and validate each review
                        review = json.loads(line.strip())
                        self._validate_review(review)
                        reviews.append(review)
                        stats['processed'] += 1
                        
                        # Update progress bar
                        pbar.update(len(line.encode('utf-8')))
                        
                        # Process chunk when it reaches the desired size
                        if len(reviews) >= chunk_size:
                            self._process_and_save_chunk(
                                reviews, 
                                output_path, 
                                chunk_number,
                                include_tfidf
                            )
                            chunk_number += 1
                            reviews = []
                            
                    except (json.JSONDecodeError, ValueError) as e:
                        stats['invalid'] += 1
                        if stats['invalid'] <= 5:  # Show only first 5 errors
                            print(f"\nError in line {line_num}: {str(e)}")
                        continue
                
                # Process any remaining reviews
                if reviews:
                    self._process_and_save_chunk(
                        reviews, 
                        output_path, 
                        chunk_number,
                        include_tfidf
                    )
        
        # Get preprocessing statistics
        if chunk_number > 0:
            sample_df = pd.read_parquet(output_path / f"reviews_chunk_0000.parquet")
            preproc_stats = self.preprocessor.get_preprocessing_stats(sample_df)
            
            print("\nPreprocessing Statistics:")
            print(f"- Number of reviews processed: {stats['processed']:,}")
            print(f"- Number of unique users: {preproc_stats['num_users']:,}")
            print(f"- Number of unique hotels: {preproc_stats['num_hotels']:,}")
            print(f"- Invalid reviews skipped: {stats['invalid']:,}")
            print(f"- Number of chunks created: {chunk_number + 1}")
            
            # Print feature statistics
            print("\nFeature Statistics:")
            for feat, feat_stats in preproc_stats['feature_stats'].items():
                print(f"- {feat}:")
                print(f"  Mean: {feat_stats['mean']:.2f}")
                print(f"  Std: {feat_stats['std']:.2f}")
                if feat_stats['missing'] > 0:
                    print(f"  Missing values: {feat_stats['missing']}")

            # Get sentiment statistics
            print("\nSentiment Analysis Results:")
            print(f"Average title sentiment: {preproc_stats['sentiment_analysis']['title_sentiment']['mean_compound']:.3f}")
            print(f"Sentiment-rating agreement: {preproc_stats['sentiment_analysis']['sentiment_rating_agreement']:.1%}")

    def _validate_review(self, review: Dict) -> None:
        """Validate review structure using preprocessor's validation"""
        self.preprocessor._validate_review_structure(review)

    def _process_and_save_chunk(self, 
                              reviews: List[Dict], 
                              output_path: Path, 
                              chunk_number: int,
                              include_tfidf: bool) -> None:
        """
        Process and save a chunk of reviews using the preprocessor.
        
        Args:
            reviews (List[Dict]): List of review dictionaries
            output_path (Path): Directory to save the parquet file
            chunk_number (int): Current chunk number
            include_tfidf (bool): Whether to include TF-IDF features
        """
        # Convert reviews to DataFrame
        df = pd.DataFrame(reviews)
        
        # Use preprocessor to process the chunk
        processed_df = self.preprocessor.process_reviews(df, include_tfidf=include_tfidf)
        
        # Save the processed chunk
        filename = f"reviews_chunk_{chunk_number:04d}.parquet"
        output_file = output_path / filename
        processed_df.to_parquet(output_file, index=False)
        
        chunk_stats = {
            'size_mb': processed_df.memory_usage(deep=True).sum() / 1024**2,
            'num_reviews': len(processed_df)
        }
        
        print(f"\nProcessed and saved chunk {chunk_number}:")
        print(f"- File: {filename}")
        print(f"- Reviews: {chunk_stats['num_reviews']:,}")
        print(f"- Memory usage: {chunk_stats['size_mb']:.2f} MB")

    def load_parquet_chunks(self, parquet_dir: str = "processed") -> Iterator[pd.DataFrame]:
        """
        Load processed parquet chunks one at a time.
        
        Args:
            parquet_dir (str): Directory containing parquet files
            
        Yields:
            pd.DataFrame: Each chunk of the processed data
        """
        parquet_path = self.data_path.parent / parquet_dir
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet directory not found: {parquet_path}")
        
        parquet_files = sorted(parquet_path.glob("*.parquet"))
        
        for parquet_file in tqdm(parquet_files, desc="Loading chunks"):
            yield pd.read_parquet(parquet_file)

    def get_chunk_info(self, parquet_dir: str = "processed") -> Dict:
        """
        Get detailed information about the processed chunks.
        
        Args:
            parquet_dir (str): Directory containing parquet files
            
        Returns:
            Dict: Information about the chunks including number of files,
                  total size, columns, and data types
        """
        parquet_path = self.data_path.parent / parquet_dir
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet directory not found: {parquet_path}")
        
        parquet_files = list(parquet_path.glob("*.parquet"))
        total_size = sum(f.stat().st_size for f in parquet_files)
        
        # Get sample of data for column info
        if parquet_files:
            sample_df = pd.read_parquet(parquet_files[0])
            columns = list(sample_df.columns)
            dtypes = sample_df.dtypes.to_dict()
        else:
            columns = []
            dtypes = {}
        
        return {
            "num_chunks": len(parquet_files),
            "total_size_gb": total_size / (1024**3),
            "chunk_files": [f.name for f in parquet_files],
            "columns": columns,
            "dtypes": {str(k): str(v) for k, v in dtypes.items()}
        }

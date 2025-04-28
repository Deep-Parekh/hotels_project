"""
Enhanced data loader for HotelRec dataset with chunked processing.
Handles large text files efficiently with minimal memory usage.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from .preprocessor import DataPreprocessor
from .dataset import ChunkedHotelReviewDataset, ChunkedCFMatrixDataset

class DataLoader:
    def __init__(
        self, 
        data_dir: Union[str, Path], 
        chunk_size: int = 100_000,
        processed_dir: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Initialize DataLoader with configurable directories and chunk size.
        
        Args:
            data_dir (Union[str, Path]): Directory containing raw data
            chunk_size (int): Number of reviews per chunk
            processed_dir (Optional[Union[str, Path]]): Directory for processed files
        """
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.processed_dir = Path(processed_dir) if processed_dir else self.data_dir.parent / "processed"
        self.preprocessor = DataPreprocessor()
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def process_large_file(
        self,
        input_file: str = "HotelRec.txt",
        chunk_size: Optional[int] = None,
        include_tfidf: bool = False,
        resume_from: Optional[int] = None
    ) -> None:
        """
        Process large text file in chunks.
        
        Args:
            input_file (str): Input file name (default: HotelRec.txt)
            chunk_size (Optional[int]): Override default chunk size
            include_tfidf (bool): Whether to include TF-IDF features
            resume_from (Optional[int]): Chunk index to resume from
        """
        in_path = self.data_dir / input_file
        n_per_shard = chunk_size or self.chunk_size
        
        if not in_path.exists():
            raise FileNotFoundError(f"Input file not found: {in_path}")
        
        reviews: List[Dict] = []
        shard_idx = resume_from if resume_from is not None else 0
        n_bad = 0
        file_size = os.path.getsize(in_path)
        
        print("\n" + "="*50)
        print(f"Starting to process: {in_path}")
        print(f"Output directory: {self.processed_dir}")
        print(f"Chunk size: {n_per_shard:,} reviews")
        if resume_from is not None:
            print(f"Resuming from chunk: {resume_from}")
        print("="*50 + "\n")
        
        with tqdm(total=file_size, unit='B', unit_scale=True) as pbar:
            for chunk_data in self._chunk_iterator(in_path, pbar):
                if shard_idx < (resume_from or 0):
                    shard_idx += 1
                    continue
                
                processed_reviews, chunk_bad = self._process_chunk(chunk_data)
                n_bad += chunk_bad
                
                if processed_reviews:
                    self._save_chunk(processed_reviews, shard_idx, include_tfidf)
                    shard_idx += 1
        
        print("\n" + "="*50)
        print(f"Processing completed!")
        print(f"Total shards created: {shard_idx+1}")
        print(f"Total malformed lines skipped: {n_bad}")
        print("="*50 + "\n")

    def _chunk_iterator(self, file_path: Path, pbar: tqdm) -> Iterator[List[str]]:
        """
        Iterate over file in chunks to manage memory usage.
        """
        chunk: List[str] = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                pbar.update(len(line.encode('utf-8')))
                chunk.append(line)
                
                if len(chunk) >= self.chunk_size:
                    yield chunk
                    chunk = []
            
            if chunk:  # Don't forget the last partial chunk
                yield chunk

    def _process_chunk(self, lines: List[str]) -> tuple[List[Dict], int]:
        """
        Process a chunk of lines into review dictionaries.
        
        Returns:
            tuple: (processed_reviews, number_of_bad_lines)
        """
        processed = []
        n_bad = 0
        
        for line in lines:
            try:
                review = json.loads(line)
                self.preprocessor._validate_review_structure(review)
                processed.append(review)
            except (json.JSONDecodeError, ValueError):
                n_bad += 1
                continue
        
        return processed, n_bad

    def _save_chunk(
        self,
        reviews: List[Dict],
        chunk_idx: int,
        include_tfidf: bool
    ) -> None:
        """
        Save processed reviews to a JSONL file with metadata.
        """
        df_raw = pd.DataFrame(reviews)
        df_proc = self.preprocessor.process_reviews(df_raw, include_tfidf=include_tfidf)

        # Remove all unnecessary fields before saving
        drop_cols = [
            'text', 'title', 'processed_text', 'processed_title', 'property_dict',
            'hotel_url', 'author'
        ]
        # Drop columns from the DataFrame
        df_proc = df_proc.drop(columns=drop_cols)

        # Save as JSONL
        jsonl_path = self.processed_dir / f"reviews_chunk_{chunk_idx:04d}.jsonl"
        df_proc.to_json(jsonl_path, orient='records', lines=True)
        
        print(f"Chunk {chunk_idx:04d}: Processed {len(df_proc):,} reviews → {jsonl_path.name}")
        
        # Save chunk metadata
        metadata = {
            'chunk_index': chunk_idx,
            'n_reviews': len(df_proc),
            'file_size': jsonl_path.stat().st_size,
            'columns': list(df_proc.columns),
            'dtypes': df_proc.dtypes.astype(str).to_dict(),
            'timestamp': datetime.now().isoformat()
        }
        
        meta_path = self.processed_dir / f"reviews_chunk_{chunk_idx:04d}.meta.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def get_processing_status(self) -> Dict:
        """
        Get current processing status and statistics.
        """
        chunks = sorted(self.processed_dir.glob("reviews_chunk_*.jsonl"))
        total_size = sum(f.stat().st_size for f in chunks)
        
        status = {
            'n_chunks': len(chunks),
            'total_size_gb': total_size / (1024 ** 3),
            'last_chunk': len(chunks) - 1 if chunks else None,
            'can_resume_from': len(chunks),
            'processed_dir': str(self.processed_dir)
        }
        
        print("\nProcessing Status:")
        print(f"Number of chunks: {status['n_chunks']}")
        print(f"Total size: {status['total_size_gb']:.2f} GB")
        print(f"Last chunk: {status['last_chunk']}")
        print(f"Can resume from: {status['can_resume_from']}")
        print(f"Processed directory: {status['processed_dir']}\n")
        
        return status

    def load_jsonl_chunks(self, processed_dir: Union[str, Path]) -> Iterator[pd.DataFrame]:
        """
        Generator that yields one processed shard at a time.
        """
        p_dir = Path(processed_dir)
        for file in sorted(p_dir.glob("reviews_chunk_*.jsonl")):
            yield pd.read_json(file, lines=True)

    def get_chunk_info(self, processed_dir: Union[str, Path]) -> Dict:
        p_dir = Path(processed_dir)
        files = sorted(p_dir.glob("reviews_chunk_*.jsonl"))
        total = sum(f.stat().st_size for f in files)
        first = pd.read_json(files[0], lines=True) if files else pd.DataFrame()
        return dict(
            num_chunks=len(files),
            total_size_gb=total / (1024 ** 3),
            dtypes=first.dtypes.astype(str).to_dict(),
            columns=list(first.columns),
        )

    def save_encodings(self):
        self.preprocessor.save_encodings(
            hotel_path='data/processed/hotel_url_to_id.pkl',
            author_path='data/processed/author_to_id.pkl'
        )

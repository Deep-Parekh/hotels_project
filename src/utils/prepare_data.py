"""
Data preparation script that handles the entire pipeline from raw data to processed files.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.data import DataLoader
from src.utils.data_utils import count_lines, validate_directories, save_chunk_metadata

def prepare_data(
    raw_file: str = "HotelRec.txt",
    chunk_size: int = 1_000_000,
    force_clean: bool = False,
    include_tfidf: bool = False,
    resume: bool = False
) -> None:
    """
    Prepare the HotelRec dataset for model training.
    
    Args:
        raw_file (str): Name of the raw data file
        chunk_size (int): Number of reviews per chunk
        force_clean (bool): Whether to clean processed directory before starting
        include_tfidf (bool): Whether to include TF-IDF features
        resume (bool): Whether to resume from last processed chunk
    """
    # Setup paths
    data_dir = project_root / "data"
    raw_dir = data_dir / "raw" / "Full_HotelRec"    
    processed_dir = data_dir / "processed"
    
    print("\n" + "="*60)
    print("Starting Data Preparation")
    print("="*60)
    print(f"Configuration:")
    print(f"- Raw file: {raw_file}")
    print(f"- Chunk size: {chunk_size:,}")
    print(f"- Include TF-IDF: {include_tfidf}")
    print(f"- Force clean: {force_clean}")
    print(f"- Resume: {resume}")
    print("="*60 + "\n")
    
    # Validate directories
    validate_directories(
        project_root=project_root,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        force_clean=force_clean and not resume
    )
    
    # Check if raw file exists
    raw_file_path = raw_dir / raw_file
    if not raw_file_path.exists():
        print(f"\nError: Raw data file not found: {raw_file_path}")
        print("Please place the HotelRec.txt file in the data/raw directory.")
        sys.exit(1)
    
    try:
        # Count total lines if not resuming
        if not resume:
            total_lines = count_lines(raw_file_path)
            print(f"Total reviews to process: {total_lines:,}")
        
        # Initialize loader
        loader = DataLoader(
            data_dir=raw_dir,
            chunk_size=chunk_size,
            processed_dir=processed_dir
        )
        
        # Get resume point if requested
        start_chunk = 0
        if resume:
            status = loader.get_processing_status()
            if status['can_resume_from'] > 0:
                start_chunk = status['can_resume_from']
                print(f"\nResuming from chunk {start_chunk}")
        
        # Process the raw file
        print("\nProcessing data file...")
        loader.process_large_file(
            input_file=raw_file,
            chunk_size=chunk_size,
            include_tfidf=include_tfidf,
            resume_from=start_chunk if resume else None
        )

        # Save the encodings after all chunks are processed
        loader.save_encodings()

        # Get final processing status
        status = loader.get_processing_status()
        
        print("\n" + "="*60)
        print("Data Preparation Complete")
        print("="*60)
        print(f"Processed data information:")
        print(f"- Number of chunks: {status['n_chunks']}")
        print(f"- Total size: {status['total_size_gb']:.2f} GB")
        print(f"\nProcessed data is available in: {processed_dir}")
        print("="*60)
        
    except Exception as e:
        print(f"\nError during data preparation: {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Prepare HotelRec dataset for training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--raw-file",
        default="HotelRec.txt",
        help="Name of the raw data file"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help="Number of reviews per chunk"
    )
    
    parser.add_argument(
        "--force-clean",
        action="store_true",
        help="Clean processed directory before starting"
    )
    
    parser.add_argument(
        "--include-tfidf",
        action="store_true",
        help="Include TF-IDF features in processed data"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last processed chunk"
    )
    
    args = parser.parse_args()
    
    prepare_data(
        raw_file=args.raw_file,
        chunk_size=args.chunk_size,
        force_clean=args.force_clean,
        include_tfidf=args.include_tfidf,
        resume=args.resume
    )

if __name__ == "__main__":
    main()

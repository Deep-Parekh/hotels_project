"""
Data preparation script that handles the entire pipeline from raw data to processed files.
"""

import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.data import DataLoader

def prepare_data(raw_file: str = "HotelRec.txt", chunk_size: int = 100000):
    """
    Prepare the HotelRec dataset for model training.
    
    Args:
        raw_file (str): Name of the raw data file
        chunk_size (int): Number of reviews per chunk
    """
    # Setup paths
    data_dir = project_root / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    
    print(f"Setting up data processing pipeline:")
    print(f"- Project root: {project_root}")
    print(f"- Raw data dir: {raw_dir}")
    print(f"- Processed data dir: {processed_dir}")
    
    # Ensure directories exist
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if raw file exists
    raw_file_path = raw_dir / raw_file
    if not raw_file_path.exists():
        print(f"\nError: Raw data file not found: {raw_file_path}")
        print("Please place the HotelRec.txt file in the data/raw directory.")
        sys.exit(1)
    
    try:
        # Initialize loader and preprocessor
        loader = DataLoader(raw_dir)
        
        # Process the raw file into chunks
        print("\nProcessing raw data file...")
        loader.process_large_file(
            input_file=raw_file,
            chunk_size=chunk_size
        )
        
        # Get information about the processed chunks
        chunk_info = loader.get_chunk_info("processed")
        print("\nProcessed data information:")
        print(f"Number of chunks: {chunk_info['num_chunks']}")
        print(f"Total size: {chunk_info['total_size_gb']:.2f} GB")
        print("\nColumns:")
        for col, dtype in chunk_info['dtypes'].items():
            print(f"- {col}: {dtype}")
        
        print("\nData preparation completed successfully!")
        print(f"Processed data is available in: {processed_dir}")
        
    except Exception as e:
        print(f"\nError during data preparation: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    prepare_data()

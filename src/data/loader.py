"""
Data loading module for hotel reviews dataset.
Provides functionality to load and validate hotel review data from various formats.
"""

import json
from pathlib import Path
from typing import List, Dict, Union, Optional

import pandas as pd
from tqdm import tqdm

class DataLoader:
    """
    A class to handle loading and basic validation of hotel review data.
    
    Attributes:
        data_path (Path): Path to the directory containing data files
        chunk_size (int): Number of lines to read at once for large files
    """
    
    def __init__(self, data_path: Union[str, Path], chunk_size: int = 10000):
        """
        Initialize the DataLoader.
        
        Args:
            data_path (Union[str, Path]): Path to the data directory
            chunk_size (int, optional): Number of lines to read at once. Defaults to 10000.
        """
        self.data_path = Path(data_path)
        self.chunk_size = chunk_size
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_path}")
    
    def load_data(self, file_name: str, validate: bool = True) -> pd.DataFrame:
        """
        Load hotel reviews data from a JSON file.
        
        Args:
            file_name (str): Name of the file to load
            validate (bool, optional): Whether to validate the data. Defaults to True.
            
        Returns:
            pd.DataFrame: Loaded and validated data
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the data validation fails
        """
        file_path = self.data_path / file_name
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read the JSON file line by line to handle large files
        data: List[Dict] = []
        total_lines = sum(1 for _ in open(file_path, 'r', encoding='utf-8'))
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=total_lines, desc="Loading reviews"):
                try:
                    review = json.loads(line)
                    data.append(review)
                except json.JSONDecodeError:
                    continue
        
        df = pd.DataFrame(data)
        
        if validate:
            self._validate_data(df)
        
        return df
    
    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate the loaded data for required columns and data types.
        
        Args:
            df (pd.DataFrame): DataFrame to validate
            
        Raises:
            ValueError: If validation fails
        """
        required_columns = [
            'hotel_url', 'author', 'date', 'rating',
            'title', 'text', 'property_dict'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Validate data types
        if not pd.api.types.is_float_dtype(df['rating']):
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    
    def save_processed_data(self, df: pd.DataFrame, output_file: str, 
                          format: str = 'parquet') -> None:
        """
        Save processed data to a file.
        
        Args:
            df (pd.DataFrame): DataFrame to save
            output_file (str): Output file name
            format (str, optional): Output format ('parquet' or 'csv'). Defaults to 'parquet'.
        """
        output_path = self.data_path / output_file
        
        if format == 'parquet':
            df.to_parquet(output_path, index=False)
        elif format == 'csv':
            df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

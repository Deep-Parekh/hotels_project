"""
Enhanced preprocessor for HotelRec dataset with improved chunked processing support.
"""

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Union
import pickle

class DataPreprocessor:
    """
    A class to preprocess hotel review data, including text preprocessing, feature extraction, and sentiment analysis.
    Prepares data for both content-based and collaborative filtering models.
    
    Features:
    - Text preprocessing (tokenization, stopwords, lemmatization)
    - Sentiment analysis for titles and reviews
    - Rating normalization
    - Feature extraction from property_dict
    - Text length features
    """
    
    def __init__(self, custom_stop_words: Optional[List[str]] = None):
        """Initialize preprocessor with optional custom stopwords."""
        print("\nInitializing DataPreprocessor...")
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
            nltk.data.find('sentiment/vader_lexicon')
        except LookupError:
            print("Downloading required NLTK data...")
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
            nltk.download('vader_lexicon')
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        if custom_stop_words:
            print(f"Adding {len(custom_stop_words)} custom stop words")
            self.stop_words.update(custom_stop_words)
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=100,
            min_df=5,
            max_df=0.95
        )
        
        # Define expected columns
        self.expected_columns = [
            'hotel_url', 'author', 'date', 'rating',
            'title', 'text', 'property_dict'
        ]
        
        # Define property ratings
        self.property_ratings = [
            'sleep quality', 'value', 'rooms',
            'service', 'cleanliness', 'location'
        ]

        # Initialize statistics tracking
        self.stats = {
            'processed_chunks': 0,
            'total_reviews': 0,
            'error_count': 0
        }
        
        self.hotel_url_to_id = {}
        self.author_to_id = {}
        self.next_hotel_id = 0
        self.next_author_id = 0
        
        print("DataPreprocessor initialization complete\n")

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text data by tokenizing, removing stopwords, and lemmatizing.
        
        Args:
            text (str): Input text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        if not isinstance(text, str):
            return ""
        
        # Tokenize and convert to lowercase
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and token.isalnum()
        ]
        
        return ' '.join(tokens)

    def get_sentiment_scores(self, text: str) -> Dict[str, float]:
        """
        Get sentiment scores for a piece of text.
        
        Args:
            text (str): Input text
            
        Returns:
            Dict[str, float]: Dictionary containing sentiment scores
        """
        if not isinstance(text, str) or not text.strip():
            return {
                'compound': 0.0,
                'pos': 0.0,
                'neu': 0.0,
                'neg': 0.0
            }
        
        return self.sentiment_analyzer.polarity_scores(text)

    def encode_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        # Encode hotel_url
        hotel_ids = []
        for url in df['hotel_url']:
            if url not in self.hotel_url_to_id:
                self.hotel_url_to_id[url] = self.next_hotel_id
                self.next_hotel_id += 1
            hotel_ids.append(self.hotel_url_to_id[url])
        df['hotel_id'] = hotel_ids

        # Encode author
        author_ids = []
        for author in df['author']:
            if author not in self.author_to_id:
                self.author_to_id[author] = self.next_author_id
                self.next_author_id += 1
            author_ids.append(self.author_to_id[author])
        df['author_id'] = author_ids

        return df

    def save_encodings(self, hotel_path='hotel_url_to_id.pkl', author_path='author_to_id.pkl'):
        with open(hotel_path, 'wb') as f:
            pickle.dump(self.hotel_url_to_id, f)
        with open(author_path, 'wb') as f:
            pickle.dump(self.author_to_id, f)

    def load_encodings(self, hotel_path='hotel_url_to_id.pkl', author_path='author_to_id.pkl'):
        with open(hotel_path, 'rb') as f:
            self.hotel_url_to_id = pickle.load(f)
        with open(author_path, 'rb') as f:
            self.author_to_id = pickle.load(f)
        self.next_hotel_id = max(self.hotel_url_to_id.values(), default=-1) + 1
        self.next_author_id = max(self.author_to_id.values(), default=-1) + 1

    def process_reviews(self, df: pd.DataFrame, include_tfidf: bool = False) -> pd.DataFrame:
        """
        Process a chunk of reviews with statistics tracking.
        
        Args:
            df (pd.DataFrame): Input DataFrame with reviews
            include_tfidf (bool): Whether to include TF-IDF features
            
        Returns:
            pd.DataFrame: Processed DataFrame
        """
        print(f"\nProcessing chunk with {len(df)} reviews...")

        # Strip whitespace from text fields
        text_fields = ['author', 'hotel_url', 'title', 'text']
        for col in text_fields:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        # Filter out reviews with invalid authors or hotel_url
        before = len(df)
        df = df[~df['author'].isin(["/undefined", "null"])]
        df = df[df['author'].notnull()]
        df = df[df['author'] != ""]
        df = df[df['hotel_url'] != ""]
        after = len(df)
        print(f"Filtered out {before - after} reviews with invalid/empty authors or hotel URLs.")

        # Encode hotel_url and author
        df = self.encode_ids(df)

        try:
            # Process text fields
            print("Processing text fields...")
            tqdm.pandas(desc="Text preprocessing")
            df['processed_text'] = df['text'].fillna('').progress_apply(self.preprocess_text)
            df['processed_title'] = df['title'].fillna('').progress_apply(self.preprocess_text)
            
            # Add text length features
            print("Calculating text lengths...")
            df['text_length'] = df['text'].fillna('').str.len()
            df['title_length'] = df['title'].fillna('').str.len()
            
            # Convert date to datetime
            print("Converting dates...")
            df['date'] = pd.to_datetime(df['date'])
            
            # Extract and normalize ratings
            print("Processing ratings...")
            # Extract all property ratings from property_dict
            property_dicts = df['property_dict'].apply(lambda x: x if isinstance(x, dict) else {})
            all_ratings = set()
            for prop_dict in property_dicts:
                all_ratings.update(prop_dict.keys())
            
            # Process each rating found in property_dict
            for rating_name in all_ratings:
                col_name = f'rating_{rating_name.replace(" ", "_")}'
                df[col_name] = df['property_dict'].apply(
                    lambda x: x.get(rating_name, np.nan) if isinstance(x, dict) else np.nan
                )
            
            # Fill missing ratings with mean
            rating_cols = [col for col in df.columns if col.startswith('rating_')]
            for col in rating_cols:
                df[col] = df[col].fillna(df[col].mean())
            
            # Sentiment analysis for text and title
            print("Performing sentiment analysis...")
            df['text_sentiment'] = df['processed_text'].fillna('').apply(self.get_sentiment_scores)
            df['title_sentiment'] = df['processed_title'].fillna('').apply(self.get_sentiment_scores)

            # Expand sentiment dicts into separate columns
            df['text_sentiment_compound'] = df['text_sentiment'].apply(lambda x: x['compound'])
            df['text_sentiment_positive'] = df['text_sentiment'].apply(lambda x: x['pos'])
            df['text_sentiment_neutral']  = df['text_sentiment'].apply(lambda x: x['neu'])
            df['text_sentiment_negative'] = df['text_sentiment'].apply(lambda x: x['neg'])

            df['title_sentiment_compound'] = df['title_sentiment'].apply(lambda x: x['compound'])
            df['title_sentiment_positive'] = df['title_sentiment'].apply(lambda x: x['pos'])
            df['title_sentiment_neutral']  = df['title_sentiment'].apply(lambda x: x['neu'])
            df['title_sentiment_negative'] = df['title_sentiment'].apply(lambda x: x['neg'])

            # Optionally drop the intermediate dict columns
            df = df.drop(columns=['text_sentiment', 'title_sentiment'])
            
            # TF-IDF from both processed_text and processed_title
            if include_tfidf:
                print("Generating TF-IDF features...")
                combined_text = df['processed_text'].fillna('') + " " + df['processed_title'].fillna('')
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(combined_text)
                tfidf_df = pd.DataFrame(
                    tfidf_matrix.toarray(),
                    columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
                )
                df = pd.concat([df, tfidf_df], axis=1)
            
            # Update statistics
            self.stats['processed_chunks'] += 1
            self.stats['total_reviews'] += len(df)
            
            print(f"Chunk processing complete: {len(df)} reviews processed\n")
            return df
            
        except Exception as e:
            self.stats['error_count'] += 1
            print(f"\nError processing chunk: {str(e)}")
            raise

    def _validate_review_structure(self, review: Dict) -> None:
        """
        Validate the structure of a review dictionary.
        
        Args:
            review (Dict): Review dictionary to validate
            
        Raises:
            ValueError: If required fields are missing or invalid
        """
        required_fields = {
            'hotel_url': str,
            'author': str,
            'date': str,
            'rating': (int, float),
            'title': str,
            'text': str
        }
        
        for field, expected_type in required_fields.items():
            if field not in review:
                raise ValueError(f"Missing required field: {field}")
            
            if not isinstance(review[field], expected_type):
                raise ValueError(f"Invalid type for {field}: expected {expected_type}, got {type(review[field])}")

    def get_stats(self) -> Dict:
        """
        Get preprocessing statistics.
        """
        stats = {
            **self.stats,
            'average_reviews_per_chunk': (
                self.stats['total_reviews'] / self.stats['processed_chunks']
                if self.stats['processed_chunks'] > 0 else 0
            )
        }
        
        print("\nPreprocessing Statistics:")
        print(f"Total chunks processed: {stats['processed_chunks']}")
        print(f"Total reviews processed: {stats['total_reviews']:,}")
        print(f"Average reviews per chunk: {stats['average_reviews_per_chunk']:,.1f}")
        print(f"Total errors encountered: {stats['error_count']}\n")
        
        return stats

    def save_processed_data(self, 
                          df: pd.DataFrame, 
                          output_path: Union[str, Path],
                          chunk_size: Optional[int] = None) -> None:
        """
        Save processed data to parquet files.
        
        Args:
            df (pd.DataFrame): Processed DataFrame to save
            output_path (Union[str, Path]): Directory to save the files
            chunk_size (Optional[int]): Number of rows per chunk
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if chunk_size:
            # Save in chunks
            num_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size else 0)
            for i in tqdm(range(num_chunks), desc="Saving chunks"):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(df))
                chunk_df = df.iloc[start_idx:end_idx]
                
                chunk_file = output_path / f"processed_chunk_{i:04d}.parquet"
                chunk_df.to_parquet(chunk_file, index=False)
        else:
            # Save as a single file
            output_file = output_path / "processed_data.parquet"
            df.to_parquet(output_file, index=False)

    def get_feature_columns(self) -> List[str]:
        """
        Get list of feature columns used for model training.
        
        Returns:
            List[str]: List of feature column names
        """
        base_features = [
            'text_length', 'title_length',
            'title_sentiment_compound', 'title_sentiment_positive',
            'title_sentiment_neutral', 'title_sentiment_negative',
            'text_sentiment_compound', 'text_sentiment_positive',
            'text_sentiment_neutral', 'text_sentiment_negative',
            'sentiment_rating_agreement'
        ]
        
        rating_features = [
            f'rating_{rating.replace(" ", "_")}' 
            for rating in self.property_ratings
        ]
        
        return base_features + rating_features

    def get_preprocessing_stats(self, df: pd.DataFrame) -> Dict:
        """
        Get statistics about the preprocessed data.
        
        Args:
            df (pd.DataFrame): Processed DataFrame
            
        Returns:
            Dict: Dictionary containing preprocessing statistics
        """
        stats = {
            'num_reviews': len(df),
            'num_users': df['author'].nunique(),
            'num_hotels': df['hotel_url'].nunique(),
            'rating_stats': df['rating'].describe().to_dict(),
            'date_range': {
                'start': df['date'].min(),
                'end': df['date'].max()
            },
            'feature_stats': {
                col: {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'missing': df[col].isnull().sum()
                }
                for col in self.get_feature_columns()
                if col in df.columns
            }
        }
        
        # Add sentiment statistics
        sentiment_stats = {
            'title_sentiment': {
                'mean_compound': df['title_sentiment_compound'].mean(),
                'positive_ratio': (df['title_sentiment_compound'] > 0).mean(),
                'negative_ratio': (df['title_sentiment_compound'] < 0).mean(),
                'neutral_ratio': (df['title_sentiment_compound'] == 0).mean()
            },
            'text_sentiment': {
                'mean_compound': df['text_sentiment_compound'].mean(),
                'positive_ratio': (df['text_sentiment_compound'] > 0).mean(),
                'negative_ratio': (df['text_sentiment_compound'] < 0).mean(),
                'neutral_ratio': (df['text_sentiment_compound'] == 0).mean()
            },
            'sentiment_rating_agreement': df['sentiment_rating_agreement'].mean()
        }
        
        stats['sentiment_analysis'] = sentiment_stats
        return stats


"""
Data preprocessing module for HotelRec dataset.
Provides functionality for text preprocessing, feature extraction, and sentiment analysis.
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


class DataPreprocessor:
    """
    A class to preprocess hotel review data, including text processing and feature extraction.
    Prepares data for both content-based and collaborative filtering models.
    
    Features:
    - Text preprocessing (tokenization, stopwords, lemmatization)
    - Sentiment analysis for titles and reviews
    - Rating normalization
    - Feature extraction from property_dict
    - Text length features
    """
    
    def __init__(self, custom_stop_words: Optional[List[str]] = None):
        """
        Initialize the preprocessor.
        
        Args:
            custom_stop_words (Optional[List[str]]): Additional stop words to use
        """
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
            nltk.data.find('sentiment/vader_lexicon')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
            nltk.download('vader_lexicon')
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        if custom_stop_words:
            self.stop_words.update(custom_stop_words)
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
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

    def process_reviews(self, df: pd.DataFrame, include_tfidf: bool = False) -> pd.DataFrame:
        """
        Process the reviews dataset for model training.
        
        Args:
            df (pd.DataFrame): Input DataFrame with reviews
            include_tfidf (bool): Whether to include TF-IDF features
            
        Returns:
            pd.DataFrame: Processed DataFrame ready for model training
        """
        print("Processing reviews...")
        processed_df = df.copy()
        
        # Validate required columns
        missing_cols = [col for col in self.expected_columns if col not in processed_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Process text fields
        print("Processing text fields...")
        tqdm.pandas(desc="Text preprocessing")
        processed_df['processed_text'] = processed_df['text'].fillna('').progress_apply(self.preprocess_text)
        processed_df['processed_title'] = processed_df['title'].fillna('').progress_apply(self.preprocess_text)
        
        # Add text length features
        processed_df['text_length'] = processed_df['text'].fillna('').str.len()
        processed_df['title_length'] = processed_df['title'].fillna('').str.len()
        
        # Add sentiment analysis
        print("Analyzing sentiment...")
        tqdm.pandas(desc="Sentiment analysis")
        
        # Title sentiment
        title_sentiments = processed_df['title'].fillna('').progress_apply(self.get_sentiment_scores)
        processed_df['title_sentiment_compound'] = title_sentiments.apply(lambda x: x['compound'])
        processed_df['title_sentiment_positive'] = title_sentiments.apply(lambda x: x['pos'])
        processed_df['title_sentiment_neutral'] = title_sentiments.apply(lambda x: x['neu'])
        processed_df['title_sentiment_negative'] = title_sentiments.apply(lambda x: x['neg'])
        
        # Review text sentiment
        text_sentiments = processed_df['text'].fillna('').progress_apply(self.get_sentiment_scores)
        processed_df['text_sentiment_compound'] = text_sentiments.apply(lambda x: x['compound'])
        processed_df['text_sentiment_positive'] = text_sentiments.apply(lambda x: x['pos'])
        processed_df['text_sentiment_neutral'] = text_sentiments.apply(lambda x: x['neu'])
        processed_df['text_sentiment_negative'] = text_sentiments.apply(lambda x: x['neg'])
        
        # Add sentiment agreement features
        processed_df['sentiment_rating_agreement'] = (
            (processed_df['title_sentiment_compound'] > 0) & (processed_df['rating'] > 3) |
            (processed_df['title_sentiment_compound'] < 0) & (processed_df['rating'] < 3)
        ).astype(int)
        
        # Convert date to datetime
        processed_df['date'] = pd.to_datetime(processed_df['date'])
        
        # Extract and normalize ratings from property_dict
        print("Extracting property ratings...")
        for rating_name in self.property_ratings:
            col_name = f'rating_{rating_name.replace(" ", "_")}'
            processed_df[col_name] = processed_df['property_dict'].apply(
                lambda x: x.get(rating_name, np.nan) if isinstance(x, dict) else np.nan
            )
        
        # Fill missing ratings with mean
        rating_cols = [col for col in processed_df.columns if col.startswith('rating_')]
        for col in rating_cols:
            processed_df[col] = processed_df[col].fillna(processed_df[col].mean())
        
        # Add TF-IDF features if requested
        if include_tfidf:
            print("Generating TF-IDF features...")
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed_df['processed_text'])
            tfidf_df = pd.DataFrame(
                tfidf_matrix.toarray(),
                columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
            )
            processed_df = pd.concat([processed_df, tfidf_df], axis=1)
        
        return processed_df

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


"""
Enhanced preprocessor for HotelRec dataset with improved chunked processing support.
"""

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Union
import pickle
import spacy
import re
import os
import en_core_web_sm

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
    - Aspect-based sentiment analysis
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
            max_features=1000,
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
        
        # Define aspects and their related terms
        self.aspects = {
            'room': ['room', 'bed', 'bathroom', 'shower', 'toilet', 'furniture', 'view'],
            'service': ['service', 'staff', 'reception', 'concierge', 'housekeeping', 'employee'],
            'food': ['breakfast', 'restaurant', 'food', 'meal', 'dining', 'cuisine', 'menu'],
            'location': ['location', 'area', 'neighborhood', 'transport', 'distance', 'access'],
            'cleanliness': ['clean', 'hygiene', 'maintenance', 'tidy', 'spotless', 'dirt'],
            'value': ['price', 'value', 'worth', 'cost', 'expensive', 'cheap', 'affordable']
        }
        
        # Load spaCy model for dependency parsing
        try:
            self.nlp = en_core_web_sm.load()
        except:
            print("Downloading spaCy model...")
            os.system("python -m spacy download en_core_web_sm")
            import en_core_web_sm
            self.nlp = en_core_web_sm.load()
        
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

    def extract_aspect_phrases(self, text: str) -> Dict[str, List[str]]:
        """
        Extract phrases related to each aspect from the text.
        
        Args:
            text (str): Input text
            
        Returns:
            Dict[str, List[str]]: Dictionary mapping aspects to relevant phrases
        """
        if not isinstance(text, str) or not text.strip():
            return {aspect: [] for aspect in self.aspects}
        
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Initialize aspect phrases dictionary
        aspect_phrases = {aspect: [] for aspect in self.aspects}
        
        # Process each sentence
        for sent in doc.sents:
            # Find aspect terms in sentence
            for aspect, terms in self.aspects.items():
                for token in sent:
                    if token.lemma_.lower() in terms:
                        # Extract the phrase around the aspect term
                        phrase = self._extract_relevant_phrase(token)
                        if phrase:
                            aspect_phrases[aspect].append(phrase)
        
        return aspect_phrases

    def _extract_relevant_phrase(self, token) -> str:
        """
        Extract a relevant phrase around an aspect term using dependency parsing.
        
        Args:
            token: spaCy token representing an aspect term
            
        Returns:
            str: Extracted phrase
        """
        # Get the main verb or adjective related to the aspect
        relevant_tokens = []
        
        # Add the aspect term itself
        relevant_tokens.append(token)
        
        # Add adjectives modifying the aspect
        for child in token.children:
            if child.dep_ in ['amod', 'advmod'] or child.pos_ in ['ADJ', 'ADV']:
                relevant_tokens.append(child)
        
        # Add the governing verb and its modifiers
        if token.head.pos_ == 'VERB':
            relevant_tokens.append(token.head)
            for child in token.head.children:
                if child.dep_ in ['advmod', 'neg'] or child.pos_ == 'ADV':
                    relevant_tokens.append(child)
        
        # Sort tokens by their position in the text
        relevant_tokens.sort(key=lambda x: x.i)
        
        # Join the tokens to form a phrase
        return ' '.join(token.text for token in relevant_tokens)

    def get_aspect_sentiments(self, text: str) -> Dict[str, Dict[str, float]]:
        """
        Get sentiment scores for each aspect mentioned in the text.
        
        Args:
            text (str): Input text
            
        Returns:
            Dict[str, Dict[str, float]]: Dictionary mapping aspects to their sentiment scores
        """
        # Extract phrases for each aspect
        aspect_phrases = self.extract_aspect_phrases(text)
        
        # Calculate sentiment for each aspect
        aspect_sentiments = {}
        for aspect, phrases in aspect_phrases.items():
            if phrases:
                # Combine phrases for the aspect
                aspect_text = ' '.join(phrases)
                # Get sentiment scores
                sentiment = self.get_sentiment_scores(aspect_text)
                aspect_sentiments[aspect] = sentiment
            else:
                # No phrases found for this aspect
                aspect_sentiments[aspect] = {
                    'compound': 0.0,
                    'pos': 0.0,
                    'neu': 0.0,
                    'neg': 0.0
                }
        
        return aspect_sentiments

    def process_reviews(self, df: pd.DataFrame, include_tfidf: bool = False) -> pd.DataFrame:
        """
        Process a chunk of reviews with statistics tracking.
        
        Args:
            df (pd.DataFrame): Input DataFrame with reviews
            include_tfidf (bool): Whether to include TF-IDF features
            
        Returns:
            pd.DataFrame: Processed DataFrame with aspect-based sentiment features
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
            
            # Perform aspect-based sentiment analysis
            print("Performing aspect-based sentiment analysis...")
            tqdm.pandas(desc="Analyzing aspects")
            aspect_sentiments = df['text'].fillna('').progress_apply(self.get_aspect_sentiments)
            
            # Add aspect sentiment features
            for aspect in self.aspects:
                for sentiment_type in ['compound', 'pos', 'neu', 'neg']:
                    col_name = f'aspect_{aspect}_{sentiment_type}'
                    df[col_name] = aspect_sentiments.apply(
                        lambda x: x[aspect][sentiment_type]
                    )
            
            # Overall sentiment analysis for text and title
            print("Performing overall sentiment analysis...")
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
            print(f"Error processing reviews: {str(e)}")
            raise

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

    def _validate_review_structure(self, review: Dict) -> None:
        """Validate the structure of a review dictionary."""
        missing = [col for col in self.expected_columns if col not in review]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def get_stats(self) -> Dict:
        """Get preprocessing statistics."""
        return self.stats

    def get_feature_columns(self) -> List[str]:
        """Get list of feature columns."""
        return [
            'hotel_id', 'author_id', 'rating',
            'text_length', 'title_length',
            'text_sentiment_compound', 'text_sentiment_positive',
            'text_sentiment_neutral', 'text_sentiment_negative',
            'title_sentiment_compound', 'title_sentiment_positive',
            'title_sentiment_neutral', 'title_sentiment_negative'
        ] + [
            f'aspect_{aspect}_{sentiment_type}'
            for aspect in self.aspects
            for sentiment_type in ['compound', 'pos', 'neu', 'neg']
        ]

    def get_preprocessing_stats(self, df: pd.DataFrame) -> Dict:
        """Get statistics about the preprocessed data."""
        return {
            'n_reviews': len(df),
            'n_hotels': df['hotel_id'].nunique(),
            'n_authors': df['author_id'].nunique(),
            'rating_stats': df['rating'].describe().to_dict(),
            'text_length_stats': df['text_length'].describe().to_dict(),
            'title_length_stats': df['title_length'].describe().to_dict()
        }


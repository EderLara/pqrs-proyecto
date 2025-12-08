# src/data/preprocessor.py
"""
Text preprocessing and cleaning module for PQRS descriptions
Handles normalization, cleaning, and feature extraction from raw text
"""

import re
import unicodedata
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Text preprocessing and cleaning utility for PQRS descriptions.
    
    Performs operations like:
    - Text normalization
    - Removal of special characters
    - Lowercasing
    - Stop word removal
    - Lemmatization (optional)
    
    Attributes:
        min_length: Minimum text length after cleaning
        max_length: Maximum text length to process
    """
    
    # Spanish stop words
    STOP_WORDS_ES = {
        "el", "la", "de", "que", "y", "a", "en", "un", "ser", "se",
        "no", "haber", "por", "con", "su", "para", "es", "se", "fue",
        "este", "ese", "aquello", "el", "la", "lo", "de", "del", "al",
        "los", "las", "una", "uno", "unos", "unas", "hay", "ha", "han"
    }
    
    def __init__(self, min_length: int = 20, max_length: int = 5000):
        """
        Initialize TextPreprocessor.
        
        Args:
            min_length: Minimum text length after preprocessing
            max_length: Maximum text length to process
        """
        self.min_length = min_length
        self.max_length = max_length
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Performs:
        1. Remove null values and convert to string
        2. Remove URLs and emails
        3. Remove special characters and numbers
        4. Normalize accents
        5. Convert to lowercase
        6. Remove extra whitespace
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
            
        Example:
            >>> preprocessor = TextPreprocessor()
            >>> cleaned = preprocessor.clean_text("FALTA PRESENCIA DEL INGENIERO!!! :-(")
            >>> print(cleaned)
            "falta presencia del ingeniero"
        """
        if not isinstance(text, str):
            text = str(text)
        
        if len(text.strip()) == 0:
            return ""
        
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-záéíóúàèìòùäëïöüA-ZÁÉÍÓÚÀÈÌÒÙÄËÏÖ\s]', '', text)
        
        # Normalize unicode characters (remove accents)
        text = self._remove_accents(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @staticmethod
    def _remove_accents(text: str) -> str:
        """
        Remove accents from text.
        
        Args:
            text: Text with potential accents
            
        Returns:
            Text without accents
        """
        text = unicodedata.normalize('NFKD', text)
        return ''.join([c for c in text if not unicodedata.combining(c)])
    
    def remove_stopwords(self, text: str) -> str:
        """
        Remove Spanish stop words from text.
        
        Args:
            text: Cleaned text
            
        Returns:
            Text without stop words
            
        Example:
            >>> cleaned = "el ingeniero del municipio realiza la inspeccion"
            >>> without_stops = preprocessor.remove_stopwords(cleaned)
            >>> print(without_stops)
            "ingeniero municipio realiza inspeccion"
        """
        words = text.split()
        filtered = [w for w in words if w not in self.STOP_WORDS_ES]
        return ' '.join(filtered)
    
    def validate_text_length(self, text: str) -> bool:
        """
        Validate that text meets length requirements.
        
        Args:
            text: Text to validate
            
        Returns:
            True if text length is valid
        """
        if len(text.strip()) < self.min_length:
            return False
        if len(text) > self.max_length:
            return False
        return True
    
    def preprocess(self, text: str, remove_stops: bool = False) -> Optional[str]:
        """
        Complete preprocessing pipeline for a single text.
        
        Args:
            text: Raw text to preprocess
            remove_stops: Whether to remove stop words
            
        Returns:
            Preprocessed text or None if invalid
            
        Example:
            >>> raw_text = "FALTA PRESENCIA DEL INGENIERO!!! Urgente!!!"
            >>> processed = preprocessor.preprocess(raw_text)
            >>> print(processed)
            "falta presencia ingeniero urgente"
        """
        try:
            # Clean text
            cleaned = self.clean_text(text)
            
            # Remove stop words if requested
            if remove_stops:
                cleaned = self.remove_stopwords(cleaned)
            
            # Validate length
            if not self.validate_text_length(cleaned):
                return None
            
            return cleaned
            
        except Exception as e:
            logger.warning(f"Error preprocessing text: {str(e)}")
            return None
    
    def batch_preprocess(self, texts: List[str], remove_stops: bool = False) -> Tuple[List[str], List[int]]:
        """
        Preprocess multiple texts.
        
        Args:
            texts: List of texts to preprocess
            remove_stops: Whether to remove stop words
            
        Returns:
            Tuple of (processed_texts, invalid_indices)
            
        Example:
            >>> texts = ["FALTA PRESENCIA!!!", "Muy bien", ""]
            >>> processed, invalid = preprocessor.batch_preprocess(texts)
            >>> print(f"Valid: {len(processed)}, Invalid: {len(invalid)}")
            "Valid: 1, Invalid: 2"
        """
        processed = []
        invalid_indices = []
        
        for idx, text in enumerate(texts):
            result = self.preprocess(text, remove_stops=remove_stops)
            
            if result is None:
                invalid_indices.append(idx)
            else:
                processed.append(result)
        
        logger.info(f"Processed {len(processed)}/{len(texts)} texts successfully")
        
        return processed, invalid_indices


class DataCleaner:
    """
    Clean and prepare DataFrame for modeling.
    
    Handles:
    - Removal of duplicate records
    - Filling missing values
    - Filtering based on criteria
    - Feature extraction
    """
    
    def __init__(self):
        """Initialize DataCleaner"""
        self.preprocessor = TextPreprocessor()
    
    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply cleaning operations to entire DataFrame.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
            
        Operations:
        1. Remove duplicates
        2. Handle missing values
        3. Clean text fields
        4. Validate data integrity
        """
        df = df.copy()
        
        # Remove exact duplicates
        initial_count = len(df)
        df = df.drop_duplicates(subset=['DESCRIPCION DEL HECHO'])
        logger.info(f"Removed {initial_count - len(df)} duplicate records")
        
        # Drop records with missing critical fields
        df = df.dropna(subset=['DESCRIPCION DEL HECHO', 'ENTIDAD RESPONSABLE', 'TIPOS DE HECHO'])
        
        # Clean text field
        df['DESCRIPCION_LIMPIA'] = df['DESCRIPCION DEL HECHO'].apply(
            lambda x: self.preprocessor.preprocess(x)
        )
        
        # Remove rows where cleaning resulted in invalid text
        df = df[df['DESCRIPCION_LIMPIA'].notna()]
        
        # Strip whitespace from categorical columns
        df['ENTIDAD RESPONSABLE'] = df['ENTIDAD RESPONSABLE'].str.strip()
        df['TIPOS DE HECHO'] = df['TIPOS DE HECHO'].str.strip()
        df['ESTADO'] = df['ESTADO'].str.strip()
        
        logger.info(f"✓ Cleaned DataFrame: {len(df)} valid records")
        
        return df
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract additional features from raw text.
        
        Features:
        - text_length: Length of cleaned text
        - is_critical: Contains critical keywords
        - has_risk: Contains risk-related keywords
        - has_safety: Contains safety-related keywords
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            DataFrame with extracted features
        """
        df = df.copy()
        
        # Text length features
        df['text_length'] = df['DESCRIPCION_LIMPIA'].str.len()
        df['word_count'] = df['DESCRIPCION_LIMPIA'].str.split().str.len()
        
        # Keyword-based features
        critical_keywords = ['riesgo', 'peligro', 'urgente', 'crítico', 'emergencia']
        df['is_critical'] = df['DESCRIPCION_LIMPIA'].str.contains(
            '|'.join(critical_keywords), case=False, na=False
        ).astype(int)
        
        safety_keywords = ['accidente', 'volcamiento', 'derrumbe', 'colapso', 'caida']
        df['has_safety_issue'] = df['DESCRIPCION_LIMPIA'].str.contains(
            '|'.join(safety_keywords), case=False, na=False
        ).astype(int)
        
        damage_keywords = ['daño', 'falla', 'ruptura', 'rotura', 'fisura']
        df['has_damage'] = df['DESCRIPCION_LIMPIA'].str.contains(
            '|'.join(damage_keywords), case=False, na=False
        ).astype(int)
        
        incomplete_keywords = ['falta', 'ausencia', 'incompleto', 'pendiente']
        df['is_incomplete'] = df['DESCRIPCION_LIMPIA'].str.contains(
            '|'.join(incomplete_keywords), case=False, na=False
        ).astype(int)
        
        logger.info("✓ Extracted features from text")
        
        return df

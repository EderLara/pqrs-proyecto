# sentiment_analyzer.py
"""
Módulo de análisis de sentimientos para PQRS.

Proporciona análisis de sentimientos con soporte para español,
identificación de emociones y scoring de confianza.
"""

import re
from typing import Dict, Tuple
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Diccionarios de palabras clave para emociones en español
EMOTION_KEYWORDS = {
    'negativo': ['problema', 'peligro', 'huecos', 'daño', 'mal', 'roto', 'pésimo', 
                 'terrible', 'horrible', 'inaceptable', 'molesto', 'enojado', 'furioso',
                 'insatisfecho', 'preocupado', 'angustiado'],
    'positivo': ['excelente', 'bien', 'mejorado', 'funcionando', 'satisfecho', 'feliz',
                 'alegre', 'optimista', 'confianza', 'seguro'],
    'neutral': ['información', 'datos', 'detalles', 'informe', 'consulta', 'pregunta']
}

class SentimentAnalyzer:
    """Analizador de sentimientos para textos en español."""
    
    def __init__(self):
        """Inicializa el analizador de sentimientos."""
        self.vader = SentimentIntensityAnalyzer()
    
    def analyze_sentiment(self, text: str) -> Dict:
        """
        Analiza el sentimiento de un texto.
        
        Args:
            text (str): Texto a analizar
            
        Returns:
            Dict con:
                - sentiment_score: float (-1.0 a 1.0)
                - sentiment_label: str (muy_negativo, negativo, neutral, positivo, muy_positivo)
                - confidence: float (0-1)
                - emotions: list de emociones detectadas
                - emoji: str
                - color: str (para Streamlit)
        """
        if not text or not isinstance(text, str):
            return self._get_neutral_sentiment()
        
        # Limpiar texto
        clean_text = text.lower().strip()
        
        # Obtener score de TextBlob
        blob = TextBlob(clean_text)
        textblob_score = blob.sentiment.polarity  # -1.0 a 1.0
        
        # Obtener score de VADER (mejor para redes sociales)
        vader_scores = self.vader.polarity_scores(clean_text)
        vader_score = vader_scores['compound']  # -1.0 a 1.0
        
        # Promediar scores
        combined_score = (textblob_score + vader_score) / 2
        
        # Detectar emociones
        emotions = self._detect_emotions(clean_text)
        
        # Generar resultado
        result = {
            'sentiment_score': round(combined_score, 2),
            'sentiment_label': self._get_sentiment_label(combined_score),
            'confidence': round(min(abs(combined_score) + 0.2, 1.0), 2),  # Calibrar confianza
            'emotions': emotions,
            'emoji': self._get_emoji(combined_score),
            'color': self._get_color(combined_score),
            'textblob_score': round(textblob_score, 2),
            'vader_score': round(vader_score, 2)
        }
        
        return result
    
    def _detect_emotions(self, text: str) -> list:
        """Detecta emociones basándose en palabras clave."""
        emotions = []
        
        for emotion, keywords in EMOTION_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    if emotion == 'negativo':
                        emotions.append('Insatisfacción')
                        emotions.append('Preocupación')
                    elif emotion == 'positivo':
                        emotions.append('Satisfacción')
                        emotions.append('Confianza')
                    elif emotion == 'neutral':
                        emotions.append('Inquietud')
                    break
        
        return list(set(emotions))[:3] if emotions else ['Indeterminado']
    
    def _get_sentiment_label(self, score: float) -> str:
        """Convierte score a etiqueta."""
        if score <= -0.6:
            return 'Muy Negativo'
        elif score <= -0.2:
            return 'Negativo'
        elif score <= 0.2:
            return 'Neutral'
        elif score <= 0.6:
            return 'Positivo'
        else:
            return 'Muy Positivo'
    
    def _get_emoji(self, score: float) -> str:
        """Retorna emoji según sentimiento."""
        if score <= -0.6:
            return '😠'
        elif score <= -0.2:
            return '😞'
        elif score <= 0.2:
            return '😐'
        elif score <= 0.6:
            return '🙂'
        else:
            return '😄'
    
    def _get_color(self, score: float) -> str:
        """Retorna color para Streamlit según sentimiento."""
        if score <= -0.6:
            return '#d62828'  # Rojo oscuro
        elif score <= -0.2:
            return '#f77f00'  # Rojo claro
        elif score <= 0.2:
            return '#ffd60a'  # Amarillo
        elif score <= 0.6:
            return '#90e0ef'  # Verde claro
        else:
            return '#06a77d'  # Verde oscuro
    
    def _get_neutral_sentiment(self) -> Dict:
        """Retorna sentimiento neutral por defecto."""
        return {
            'sentiment_score': 0.0,
            'sentiment_label': 'Neutral',
            'confidence': 0.0,
            'emotions': ['Indeterminado'],
            'emoji': '😐',
            'color': '#ffd60a',
            'textblob_score': 0.0,
            'vader_score': 0.0
        }
    
    def get_sentiment_distribution(self, texts: list) -> Dict:
        """
        Analiza distribución de sentimientos en múltiples textos.
        
        Args:
            texts (list): Lista de textos
            
        Returns:
            Dict con estadísticas de distribución
        """
        if not texts:
            return {}
        
        sentiments = [self.analyze_sentiment(text)['sentiment_score'] for text in texts]
        
        return {
            'media': round(sum(sentiments) / len(sentiments), 2),
            'mediana': round(sorted(sentiments)[len(sentiments)//2], 2),
            'minimo': round(min(sentiments), 2),
            'maximo': round(max(sentiments), 2),
            'total_negativos': len([s for s in sentiments if s < -0.2]),
            'total_neutrales': len([s for s in sentiments if -0.2 <= s <= 0.2]),
            'total_positivos': len([s for s in sentiments if s > 0.2])
        }

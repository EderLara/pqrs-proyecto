# MÓDULO 1: src/models/entity_classifier.py

"""
Clasificador de Entidades Responsables para PQRS.

Modelos soportados:
- Logistic Regression (baseline)
- SVM
- Random Forest
- Gradient Boosting
"""

import logging
import pickle
from typing import Dict, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import json

logger = logging.getLogger(__name__)

class EntityClassifier:
    """
    Clasificador multiclase para entidades responsables.
    
    Atributos:
        model: Modelo entrenado
        model_type: Tipo de modelo ('logistic', 'svm', 'rf', 'gb')
        classes_: Clases únicas
        feature_names_: Nombres de features
    
    Ejemplo:
        >>> clf = EntityClassifier(model_type='logistic')
        >>> clf.train(X_train, y_train)
        >>> predictions = clf.predict(X_test)
        >>> metrics = clf.evaluate(X_test, y_test)
    """
    
    def __init__(self, model_type: str = 'logistic', **kwargs):
        """
        Inicializa el clasificador.
        
        Args:
            model_type: 'logistic', 'svm', 'rf', 'gb'
            **kwargs: Parámetros adicionales para el modelo
        """
        self.model_type = model_type
        self.model = None
        self.classes_ = None
        self.feature_names_ = None
        self.training_history_ = {}
        
        logger.info(f"Initializing EntityClassifier with {model_type}")
        self._create_model(**kwargs)
    
    def _create_model(self, **kwargs) -> None:
        """Crea la instancia del modelo."""
        if self.model_type == 'logistic':
            self.model = LogisticRegression(
                max_iter=500,
                random_state=42,
                multi_class='multinomial',
                **kwargs
            )
        elif self.model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                probability=True,
                random_state=42,
                **kwargs
            )
        elif self.model_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                **kwargs
            )
        elif self.model_type == 'gb':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                random_state=42,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def train(self, X_train, y_train: pd.Series) -> Dict[str, float]:
        """
        Entrena el clasificador.
        
        Args:
            X_train: Features (puede ser dense o sparse)
            y_train: Labels
        
        Returns:
            Dict con métricas de entrenamiento
        """
        logger.info(f"Training {self.model_type} on {X_train.shape[0]} samples")
        
        # Convertir a array denso si es necesario
        if hasattr(X_train, 'toarray'):
            X_train = X_train.toarray()
        
        # Entrenar
        self.model.fit(X_train, y_train)
        self.classes_ = self.model.classes_
        
        # Métricas
        y_pred = self.model.predict(X_train)
        metrics = {
            'accuracy': accuracy_score(y_train, y_pred),
            'precision': precision_score(y_train, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_train, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_train, y_pred, average='weighted', zero_division=0)
        }
        
        self.training_history_['train'] = metrics
        
        logger.info(f"✓ Training complete:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.3f}")
        logger.info(f"  F1-Score: {metrics['f1']:.3f}")
        
        return metrics
    
    def evaluate(self, X_test, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evalúa el modelo en test set.
        
        Args:
            X_test: Features
            y_test: Labels
        
        Returns:
            Dict con métricas, matriz de confusión y reporte
        """
        # Convertir si es necesario
        if hasattr(X_test, 'toarray'):
            X_test = X_test.toarray()
        
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        self.training_history_['test'] = metrics
        
        logger.info(f"✓ Evaluation complete:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.3f}")
        logger.info(f"  F1-Score: {metrics['f1']:.3f}")
        
        return {
            'metrics': metrics,
            'confusion_matrix': cm,
            'report': report,
            'y_pred': y_pred
        }
    
    def predict(self, X, return_proba: bool = False):
        """
        Realiza predicción.
        
        Args:
            X: Features
            return_proba: Si True, retorna probabilidades
        
        Returns:
            Predicciones (y opcionalmente probabilidades)
        """
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        predictions = self.model.predict(X)
        
        if return_proba:
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(X)
                return predictions, proba
            else:
                logger.warning("Model doesn't support probability estimates")
                return predictions, None
        
        return predictions
    
    def save(self, path: str) -> None:
        """Guarda el modelo."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"✓ Model saved to {path}")
    
    @staticmethod
    def load(path: str) -> 'EntityClassifier':
        """Carga un modelo guardado."""
        with open(path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"✓ Model loaded from {path}")
        return model


# MÓDULO 2: src/models/issue_classifier.py

"""
Clasificador de Tipos de Hechos para PQRS.

Características especiales:
- Manejo de desbalanceo de clases con class_weight
- Soporte para SMOTE en entrenamiento
- Mecanismo de umbral ajustable
"""

import logging
import pickle
from typing import Dict, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)

class IssueClassifier:
    """
    Clasificador multiclase para tipos de hechos.
    
    Maneja clases desbalanceadas usando:
    - class_weight='balanced'
    - SMOTE opcional
    - Ajuste de umbral
    
    Atributos:
        model: Random Forest entrenado
        smote_applied: Si SMOTE se aplicó
        threshold: Umbral de confianza mínima
    """
    
    def __init__(self, use_smote: bool = True, threshold: float = 0.3):
        """
        Inicializa el clasificador.
        
        Args:
            use_smote: Aplicar SMOTE en entrenamiento
            threshold: Confianza mínima para predicción
        """
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',  # Maneja desbalanceo
            max_depth=15
        )
        self.use_smote = use_smote
        self.smote_applied = False
        self.threshold = threshold
        self.classes_ = None
        self.training_history_ = {}
        
        logger.info("Initializing IssueClassifier with RandomForest + Balanced classes")
    
    def train(self, X_train, y_train: pd.Series) -> Dict[str, float]:
        """
        Entrena con manejo de desbalanceo.
        
        Args:
            X_train: Features
            y_train: Labels
        
        Returns:
            Métricas de entrenamiento
        """
        logger.info(f"Training IssueClassifier on {X_train.shape[0]} samples")
        
        # Convertir a denso
        if hasattr(X_train, 'toarray'):
            X_train = X_train.toarray()
        
        # SMOTE si es necesario
        if self.use_smote:
            logger.info("Applying SMOTE for class balancing...")
            try:
                smote = SMOTE(
                    random_state=42,
                    k_neighbors=min(3, (y_train.value_counts().min() - 1))
                )
                X_train, y_train = smote.fit_resample(X_train, y_train)
                self.smote_applied = True
                logger.info(f"  ✓ SMOTE applied: {len(y_train)} samples")
            except Exception as e:
                logger.warning(f"  ⚠️ SMOTE failed: {e}. Continuing without SMOTE.")
                self.smote_applied = False
        
        # Entrenar
        self.model.fit(X_train, y_train)
        self.classes_ = self.model.classes_
        
        # Métricas
        y_pred = self.model.predict(X_train)
        metrics = {
            'accuracy': accuracy_score(y_train, y_pred),
            'precision': precision_score(y_train, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_train, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_train, y_pred, average='weighted', zero_division=0)
        }
        
        self.training_history_['train'] = metrics
        
        logger.info(f"✓ Training complete (SMOTE: {self.smote_applied}):")
        logger.info(f"  Accuracy: {metrics['accuracy']:.3f}")
        logger.info(f"  F1-Score: {metrics['f1']:.3f}")
        
        return metrics
    
    def evaluate(self, X_test, y_test: pd.Series) -> Dict[str, Any]:
        """Evalúa el modelo."""
        if hasattr(X_test, 'toarray'):
            X_test = X_test.toarray()
        
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        self.training_history_['test'] = metrics
        
        logger.info(f"✓ Evaluation complete:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.3f}")
        logger.info(f"  F1-Score: {metrics['f1']:.3f}")
        
        return {
            'metrics': metrics,
            'confusion_matrix': cm,
            'report': report,
            'y_pred': y_pred
        }
    
    def predict(self, X, return_proba: bool = False):
        """Realiza predicción."""
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        predictions = self.model.predict(X)
        
        if return_proba:
            proba = self.model.predict_proba(X)
            confidence = np.max(proba, axis=1)
            
            # Aplicar umbral
            uncertain = confidence < self.threshold
            predictions[uncertain] = 'UNCERTAIN'
            
            return predictions, proba, confidence
        
        return predictions
    
    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """Retorna los features más importantes."""
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[-top_n:]
        
        return pd.DataFrame({
            'feature_index': indices,
            'importance': importances[indices]
        })
    
    def save(self, path: str) -> None:
        """Guarda el modelo."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"✓ Model saved to {path}")
    
    @staticmethod
    def load(path: str) -> 'IssueClassifier':
        """Carga un modelo."""
        with open(path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"✓ Model loaded from {path}")
        return model


# MÓDULO 3: src/models/sentiment_analyzer.py

"""
Analizador de Sentimientos para PQRS.

Implementa análisis de sentimientos usando:
- Diccionario de palabras clave personalizado
- TextBlob para análisis base
- Scoring ponderado por dominio
"""

import logging
from typing import Dict, Tuple, Optional
import re
from textblob import TextBlob
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Analizador de sentimientos específico para PQRS.
    
    Levels:
    - VERY_NEGATIVE: -1.0 a -0.5
    - NEGATIVE: -0.5 a 0.0
    - NEUTRAL: -0.1 a 0.1
    - POSITIVE: 0.0 a 1.0
    """
    
    # Diccionario de palabras clave para dominio vial
    DOMAIN_KEYWORDS = {
        'critical': [
            'riesgo', 'peligro', 'derrumbe', 'colapso', 'volcamiento',
            'critico', 'urgente', 'grave', 'peligrosa', 'accidente'
        ],
        'negative': [
            'falta', 'ausencia', 'daño', 'defecto', 'problema', 'error',
            'incumplimiento', 'retraso', 'incorrecto', 'deficiente',
            'inseguro', 'deteriorado', 'roto', 'tapado', 'bloqueado'
        ],
        'positive': [
            'excelente', 'bueno', 'adecuado', 'correcto', 'completo',
            'mejorado', 'restaurado', 'beneficio', 'favorable'
        ]
    }
    
    def __init__(self):
        """Inicializa el analizador."""
        logger.info("Initializing SentimentAnalyzer with domain dictionary")
        self._prepare_keywords()
    
    def _prepare_keywords(self) -> None:
        """Prepara diccionario de palabras clave."""
        self.critical_words = set(self.DOMAIN_KEYWORDS['critical'])
        self.negative_words = set(self.DOMAIN_KEYWORDS['negative'])
        self.positive_words = set(self.DOMAIN_KEYWORDS['positive'])
        logger.info(f"  Loaded {len(self.critical_words)} critical words")
        logger.info(f"  Loaded {len(self.negative_words)} negative words")
        logger.info(f"  Loaded {len(self.positive_words)} positive words")
    
    def _preprocess_text(self, text: str) -> str:
        """Limpia y normaliza texto."""
        text = str(text).lower()
        text = re.sub(r'[^a-záéíóú\s]', '', text)
        return text.strip()
    
    def analyze(self, text: str) -> Dict[str, any]:
        """
        Analiza sentimiento de un texto.
        
        Args:
            text: Texto a analizar
        
        Returns:
            Dict con:
            - polarity: -1 a 1
            - subjectivity: 0 a 1
            - level: VERY_NEGATIVE, NEGATIVE, NEUTRAL, POSITIVE
            - keywords_found: palabras clave detectadas
            - confidence: confianza del análisis
        """
        clean_text = self._preprocess_text(text)
        
        # Análisis base con TextBlob
        blob = TextBlob(clean_text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Detección de palabras clave
        words = set(clean_text.split())
        
        critical_count = len(words & self.critical_words)
        negative_count = len(words & self.negative_words)
        positive_count = len(words & self.positive_words)
        
        # Ajustar polaridad con palabras clave
        keyword_polarity = 0
        if critical_count > 0:
            keyword_polarity -= 0.5 * min(critical_count / 3, 1.0)
        if negative_count > 0:
            keyword_polarity -= 0.3 * min(negative_count / 5, 1.0)
        if positive_count > 0:
            keyword_polarity += 0.3 * min(positive_count / 3, 1.0)
        
        # Polaridad final (combinada)
        final_polarity = np.clip(polarity * 0.6 + keyword_polarity * 0.4, -1, 1)
        
        # Determinar nivel
        if final_polarity <= -0.5:
            level = 'VERY_NEGATIVE'
        elif final_polarity <= 0.0:
            level = 'NEGATIVE'
        elif final_polarity <= 0.1:
            level = 'NEUTRAL'
        else:
            level = 'POSITIVE'
        
        # Confianza
        confidence = 0.7 + (critical_count + negative_count) * 0.05
        confidence = min(confidence, 0.95)
        
        return {
            'text': text[:100],
            'polarity': final_polarity,
            'subjectivity': subjectivity,
            'level': level,
            'critical_words': list(words & self.critical_words),
            'negative_words': list(words & self.negative_words),
            'positive_words': list(words & self.positive_words),
            'confidence': confidence
        }
    
    def analyze_batch(self, texts: list) -> pd.DataFrame:
        """
        Analiza múltiples textos.
        
        Args:
            texts: Lista de textos
        
        Returns:
            DataFrame con resultados
        """
        results = [self.analyze(text) for text in texts]
        return pd.DataFrame(results)


# MÓDULO 4: src/models/severity_scorer.py

"""
Calculador de Severidad/Importancia para PQRS.

Combina múltiples señales en un score 0-10.
"""

import logging
from typing import Dict, List
import numpy as np
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

class SeverityScorer:
    """
    Calcula score de severidad (0-10) basado en múltiples factores.
    
    Factores considerados:
    - Sentimiento (30%)
    - Palabras clave críticas (25%)
    - Estado del reclamo (20%)
    - Tiempo sin resolver (15%)
    - Impacto geográfico (10%)
    
    Categorías:
    - RED (8-10): Urgente
    - YELLOW (5-7): Importante
    - GREEN (0-4): Rutinario
    """
    
    CRITICAL_KEYWORDS = [
        'riesgo', 'peligro', 'derrumbe', 'colapso', 'volcamiento',
        'accidente', 'urgente', 'grave', 'critico', 'emergencia'
    ]
    
    def __init__(self, weights: Optional[Dict] = None):
        """
        Inicializa el calculador.
        
        Args:
            weights: Dict con pesos de cada factor
        """
        self.weights = weights or {
            'sentiment': 0.30,
            'keywords': 0.25,
            'status': 0.20,
            'time': 0.15,
            'impact': 0.10
        }
        
        # Validar suma de pesos
        if not np.isclose(sum(self.weights.values()), 1.0):
            raise ValueError("Weights must sum to 1.0")
        
        logger.info("Initializing SeverityScorer")
        logger.info(f"  Weights: {self.weights}")
    
    def _score_sentiment(self, polarity: float) -> float:
        """Convierte polaridad (-1 a 1) a score (0-10)."""
        # Negativo → alta severidad
        # Positivo → baja severidad
        if polarity <= -0.5:
            return 10.0
        elif polarity <= 0.0:
            return 7.0 - (polarity * 14)  # -0.5 a 0 → 10 a 7
        elif polarity <= 0.5:
            return 5.0 - (polarity * 10)  # 0 a 0.5 → 5 a 0
        else:
            return 0.0
    
    def _score_keywords(self, critical_count: int, text_length: int) -> float:
        """Calcula score basado en palabras críticas."""
        if text_length == 0:
            return 0.0
        
        keyword_density = critical_count / max(text_length.split(), 1)
        
        if keyword_density > 0.1:
            return 10.0
        elif keyword_density > 0.05:
            return 7.0
        elif keyword_density > 0.01:
            return 4.0
        else:
            return 0.0
    
    def _score_status(self, status: str) -> float:
        """Convierte estado a score."""
        status_lower = str(status).lower()
        
        if 'resuelto' in status_lower or 'cerrado' in status_lower:
            return 1.0  # Bajo: ya resuelto
        elif 'tramite' in status_lower or 'proceso' in status_lower:
            return 6.0  # Medio: en proceso
        else:
            return 9.0  # Alto: abierto/sin resolver
    
    def _score_time_elapsed(self, days_elapsed: int) -> float:
        """Convierte días transcurridos a score."""
        if days_elapsed <= 7:
            return 3.0
        elif days_elapsed <= 30:
            return 6.0
        elif days_elapsed <= 90:
            return 8.0
        else:
            return 10.0  # Más de 90 días sin resolver
    
    def calculate(self,
                  polarity: float,
                  critical_keywords: int,
                  text_length: int,
                  status: str,
                  days_elapsed: int,
                  impact_community: str = 'local') -> Dict[str, any]:
        """
        Calcula score final de severidad.
        
        Args:
            polarity: Sentimiento (-1 a 1)
            critical_keywords: Cantidad de palabras clave críticas
            text_length: Longitud del texto en palabras
            status: Estado del reclamo
            days_elapsed: Días desde la radicación
            impact_community: 'local', 'regional', 'national'
        
        Returns:
            Dict con score, nivel y detalles
        """
        # Componentes de score
        sentiment_score = self._score_sentiment(polarity)
        keywords_score = self._score_keywords(critical_keywords, text_length)
        status_score = self._score_status(status)
        time_score = self._score_time_elapsed(days_elapsed)
        
        # Impact score
        impact_scores = {
            'local': 2.0,
            'regional': 5.0,
            'national': 8.0
        }
        impact_score = impact_scores.get(impact_community.lower(), 2.0)
        
        # Score final ponderado
        final_score = (
            sentiment_score * self.weights['sentiment'] +
            keywords_score * self.weights['keywords'] +
            status_score * self.weights['status'] +
            time_score * self.weights['time'] +
            impact_score * self.weights['impact']
        )
        
        # Categoría
        if final_score >= 8:
            level = 'RED'
            urgency = 'URGENTE'
        elif final_score >= 5:
            level = 'YELLOW'
            urgency = 'IMPORTANTE'
        else:
            level = 'GREEN'
            urgency = 'RUTINARIO'
        
        return {
            'final_score': round(final_score, 2),
            'level': level,
            'urgency': urgency,
            'components': {
                'sentiment': round(sentiment_score, 2),
                'keywords': round(keywords_score, 2),
                'status': round(status_score, 2),
                'time': round(time_score, 2),
                'impact': round(impact_score, 2)
            }
        }
    
    def batch_calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula severidad para múltiples registros."""
        results = []
        
        for _, row in df.iterrows():
            score_result = self.calculate(
                polarity=row.get('sentiment_polarity', 0),
                critical_keywords=row.get('critical_keywords', 0),
                text_length=len(str(row.get('text', '')).split()),
                status=row.get('status', 'unknown'),
                days_elapsed=row.get('days_elapsed', 0),
                impact_community=row.get('impact', 'local')
            )
            results.append(score_result)
        
        return pd.DataFrame(results)

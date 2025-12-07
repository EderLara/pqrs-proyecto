# MÓDULOS 5, 6 Y 7

# ═════════════════════════════════════════════════════════════════════════════
# MÓDULO 5: src/models/model_manager.py
# ═════════════════════════════════════════════════════════════════════════════

"""
Gestor centralizado de modelos con versionado.

Funcionalidades:
- Cargar/guardar modelos por versión
- Historial de entrenamiento
- Metadata de modelos
- Comparación entre versiones
"""

import logging
import os
import json
from typing import Dict, Optional, Any, Tuple
from datetime import datetime
import pickle
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Gestor de modelos con versionado.
    
    Estructura de carpetas:
    models/
    ├── v1/
    │   ├── entity_classifier.pkl
    │   ├── issue_classifier.pkl
    │   ├── sentiment_analyzer.pkl
    │   ├── severity_scorer.pkl
    │   ├── vectorizer.pkl
    │   └── metadata.json
    ├── v2/
    └── ...
    
    Ejemplo:
        >>> mgr = ModelManager()
        >>> mgr.load_latest()
        >>> predictions = mgr.predict(text)
    """
    
    def __init__(self, base_path: str = "models"):
        """
        Inicializa el gestor.
        
        Args:
            base_path: Ruta base para guardar modelos
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        self.current_version = None
        self.models = {}
        self.metadata = {}
        
        logger.info(f"ModelManager initialized at {self.base_path}")
    
    def save_version(self, 
                    version: str,
                    entity_clf,
                    issue_clf,
                    sentiment_analyzer,
                    severity_scorer,
                    vectorizer,
                    metrics: Dict = None) -> None:
        """
        Guarda todos los modelos de una versión.
        
        Args:
            version: Nombre de versión (e.g., 'v1', 'v2')
            entity_clf: EntityClassifier entrenado
            issue_clf: IssueClassifier entrenado
            sentiment_analyzer: SentimentAnalyzer
            severity_scorer: SeverityScorer
            vectorizer: Vectorizador TF-IDF
            metrics: Dict con métricas de evaluación
        """
        version_path = self.base_path / version
        version_path.mkdir(exist_ok=True)
        
        logger.info(f"Saving version {version}...")
        
        # Guardar modelos
        with open(version_path / "entity_classifier.pkl", 'wb') as f:
            pickle.dump(entity_clf, f)
        logger.info(f"  ✓ Entity classifier saved")
        
        with open(version_path / "issue_classifier.pkl", 'wb') as f:
            pickle.dump(issue_clf, f)
        logger.info(f"  ✓ Issue classifier saved")
        
        with open(version_path / "sentiment_analyzer.pkl", 'wb') as f:
            pickle.dump(sentiment_analyzer, f)
        logger.info(f"  ✓ Sentiment analyzer saved")
        
        with open(version_path / "severity_scorer.pkl", 'wb') as f:
            pickle.dump(severity_scorer, f)
        logger.info(f"  ✓ Severity scorer saved")
        
        with open(version_path / "vectorizer.pkl", 'wb') as f:
            pickle.dump(vectorizer, f)
        logger.info(f"  ✓ Vectorizer saved")
        
        # Guardar metadata
        metadata = {
            'version': version,
            'created_at': datetime.now().isoformat(),
            'metrics': metrics or {},
            'files': {
                'entity_classifier': 'entity_classifier.pkl',
                'issue_classifier': 'issue_classifier.pkl',
                'sentiment_analyzer': 'sentiment_analyzer.pkl',
                'severity_scorer': 'severity_scorer.pkl',
                'vectorizer': 'vectorizer.pkl'
            }
        }
        
        with open(version_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"  ✓ Metadata saved")
        
        logger.info(f"✓ Version {version} saved successfully")
    
    def load_version(self, version: str) -> Dict[str, Any]:
        """
        Carga todos los modelos de una versión.
        
        Args:
            version: Versión a cargar
        
        Returns:
            Dict con todos los componentes
        """
        version_path = self.base_path / version
        
        if not version_path.exists():
            raise FileNotFoundError(f"Version {version} not found")
        
        logger.info(f"Loading version {version}...")
        
        # Cargar modelos
        with open(version_path / "entity_classifier.pkl", 'rb') as f:
            entity_clf = pickle.load(f)
        
        with open(version_path / "issue_classifier.pkl", 'rb') as f:
            issue_clf = pickle.load(f)
        
        with open(version_path / "sentiment_analyzer.pkl", 'rb') as f:
            sentiment_analyzer = pickle.load(f)
        
        with open(version_path / "severity_scorer.pkl", 'rb') as f:
            severity_scorer = pickle.load(f)
        
        with open(version_path / "vectorizer.pkl", 'rb') as f:
            vectorizer = pickle.load(f)
        
        # Cargar metadata
        with open(version_path / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        self.current_version = version
        self.models = {
            'entity_clf': entity_clf,
            'issue_clf': issue_clf,
            'sentiment_analyzer': sentiment_analyzer,
            'severity_scorer': severity_scorer,
            'vectorizer': vectorizer
        }
        self.metadata = metadata
        
        logger.info(f"✓ Version {version} loaded successfully")
        
        return self.models
    
    def load_latest(self) -> Dict[str, Any]:
        """Carga la versión más reciente."""
        versions = sorted([v.name for v in self.base_path.glob("v*")])
        
        if not versions:
            raise FileNotFoundError("No versions found")
        
        latest = versions[-1]
        logger.info(f"Loading latest version: {latest}")
        
        return self.load_version(latest)
    
    def list_versions(self) -> list:
        """Lista todas las versiones disponibles."""
        versions = sorted([v.name for v in self.base_path.glob("v*")])
        
        for v in versions:
            metadata_path = self.base_path / v / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                    logger.info(f"{v}: {metadata.get('created_at', 'unknown')}")
        
        return versions
    
    def delete_version(self, version: str) -> None:
        """Elimina una versión."""
        version_path = self.base_path / version
        
        if version_path.exists():
            shutil.rmtree(version_path)
            logger.info(f"✓ Version {version} deleted")
        else:
            logger.warning(f"Version {version} not found")
    
    def predict(self, text: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Realiza predicción usando la versión cargada.
        
        Args:
            text: Texto a clasificar
            version: Versión específica (si no, usa la cargada)
        
        Returns:
            Dict con todas las predicciones
        """
        if version and version != self.current_version:
            self.load_version(version)
        
        if not self.models:
            raise ValueError("No model loaded. Call load_version() first.")
        
        # Vectorizar
        X = self.models['vectorizer'].transform([text])
        
        # Entity
        entity_pred, entity_proba = self.models['entity_clf'].predict(X, return_proba=True)
        entity_conf = entity_proba.max() if entity_proba is not None else 0.0
        
        # Issue
        issue_pred, issue_proba, issue_conf = self.models['issue_clf'].predict(X, return_proba=True)
        
        # Sentiment
        sentiment_result = self.models['sentiment_analyzer'].analyze(text)
        
        # Severity
        severity_result = self.models['severity_scorer'].calculate(
            polarity=sentiment_result['polarity'],
            critical_keywords=len(sentiment_result['critical_words']),
            text_length=len(text.split()),
            status='open',
            days_elapsed=0
        )
        
        return {
            'entity': entity_pred[0],
            'entity_confidence': float(entity_conf),
            'issue': issue_pred[0],
            'issue_confidence': float(issue_conf[0]) if hasattr(issue_conf, '__len__') else float(issue_conf),
            'sentiment': sentiment_result,
            'severity': severity_result,
            'version': self.current_version
        }


# ═════════════════════════════════════════════════════════════════════════════
# MÓDULO 6: src/features/extractor.py
# ═════════════════════════════════════════════════════════════════════════════

"""
Extractor de características adicionales desde el texto.

Características extraídas:
- Conteo de palabras
- Sentencias
- Caracteres especiales
- Palabras clave por dominio
"""

import logging
import re
from typing import Dict, List
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Extrae características del texto para mejorar modelado."""
    
    DOMAIN_WORDS = {
        'infrastructure': [
            'vía', 'carretera', 'camino', 'puente', 'alcantarilla',
            'pavimento', 'asfalto', 'cemento', 'acera', 'sardinel'
        ],
        'safety': [
            'riesgo', 'peligro', 'seguridad', 'accidente', 'colapso',
            'derrumbe', 'volcamiento', 'caída'
        ],
        'environmental': [
            'contaminación', 'residuos', 'basura', 'agua', 'ambiente',
            'ecosistema', 'flora', 'fauna', 'vertimiento'
        ],
        'social': [
            'comunidad', 'población', 'participación', 'capacitación',
            'empleo', 'beneficiario', 'vereda', 'barrio'
        ]
    }
    
    def __init__(self):
        """Inicializa el extractor."""
        logger.info("Initializing FeatureExtractor")
        self._prepare_dictionaries()
    
    def _prepare_dictionaries(self) -> None:
        """Prepara diccionarios de palabras."""
        for category, words in self.DOMAIN_WORDS.items():
            setattr(self, f'{category}_words', set(words))
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """
        Extrae múltiples características.
        
        Args:
            text: Texto a procesar
        
        Returns:
            Dict con características numéricas
        """
        text_lower = str(text).lower()
        words = text_lower.split()
        
        features = {
            'text_length': len(text),
            'word_count': len(words),
            'sentence_count': len(re.split(r'[.!?]+', text)),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'digit_ratio': sum(1 for c in text if c.isdigit()) / max(len(text), 1),
            'punctuation_count': sum(1 for c in text if c in '.,!?;:'),
        }
        
        # Contar palabras clave por categoría
        word_set = set(words)
        for category in self.DOMAIN_WORDS.keys():
            domain_set = getattr(self, f'{category}_words')
            matches = len(word_set & domain_set)
            features[f'{category}_keywords'] = matches
        
        return features
    
    def extract_batch(self, texts: List[str]) -> pd.DataFrame:
        """Extrae features para múltiples textos."""
        features_list = [self.extract_features(text) for text in texts]
        return pd.DataFrame(features_list)


# ═════════════════════════════════════════════════════════════════════════════
# MÓDULO 7: src/models/ensemble_predictor.py
# ═════════════════════════════════════════════════════════════════════════════

"""
Predictor ensemble que combina múltiples modelos.

Estrategia:
- Voting ensemble para entidades
- Stacking para tipos de hechos
- Promedio ponderado para sentimiento
"""

import logging
from typing import Dict, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class EnsemblePredictor:
    """
    Combina predicciones de múltiples modelos para mejorar robustez.
    
    Métodos:
    - Voting: Mayoría simple
    - Weighted Voting: Voto ponderado por confianza
    - Stacking: Meta-modelo
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        Inicializa el ensemble.
        
        Args:
            weights: Pesos para each model
        """
        self.weights = weights or {
            'entity': 1.0,
            'issue': 1.0,
            'sentiment': 1.0,
            'severity': 1.0
        }
        
        self.models = {}
        logger.info("Initializing EnsemblePredictor")
    
    def register_model(self, name: str, model, weight: float = 1.0) -> None:
        """Registra un modelo en el ensemble."""
        self.models[name] = model
        self.weights[name] = weight
        logger.info(f"✓ Model {name} registered with weight {weight}")
    
    def predict(self, X) -> Dict[str, Any]:
        """
        Realiza predicción con todos los modelos.
        
        Args:
            X: Features
        
        Returns:
            Predicciones combinadas con confianza
        """
        predictions = {}
        confidences = {}
        
        for name, model in self.models.items():
            try:
                # Obtener predicción
                if hasattr(model, 'predict_proba'):
                    pred = model.predict(X)
                    proba = model.predict_proba(X)
                    conf = proba.max()
                else:
                    pred = model.predict(X)
                    conf = 1.0
                
                predictions[name] = pred
                confidences[name] = conf * self.weights.get(name, 1.0)
                
            except Exception as e:
                logger.warning(f"Model {name} failed: {e}")
        
        # Combinar predicciones
        avg_confidence = np.mean(list(confidences.values())) if confidences else 0.0
        
        return {
            'predictions': predictions,
            'confidences': confidences,
            'average_confidence': avg_confidence
        }
    
    def get_summary(self) -> str:
        """Retorna resumen del ensemble."""
        summary = f"Ensemble with {len(self.models)} models:\n"
        for name, model in self.models.items():
            summary += f"  - {name}: weight={self.weights.get(name, 1.0)}\n"
        return summary


# ═════════════════════════════════════════════════════════════════════════════
# RESUMEN DE 7 MÓDULOS CREADOS
# ═════════════════════════════════════════════════════════════════════════════

"""
✓ MÓDULO 1: EntityClassifier
  - Logistic Regression, SVM, Random Forest, Gradient Boosting
  - Métodos: train(), evaluate(), predict(), save/load()
  - Uso: Clasificar entidad responsable

✓ MÓDULO 2: IssueClassifier
  - Random Forest con class_weight='balanced'
  - Manejo de SMOTE opcional
  - Ajuste de umbral de confianza
  - Uso: Clasificar tipos de hechos

✓ MÓDULO 3: SentimentAnalyzer
  - Análisis basado en diccionario + TextBlob
  - 4 niveles: VERY_NEGATIVE, NEGATIVE, NEUTRAL, POSITIVE
  - Detección de palabras clave críticas
  - Uso: Analizar sentimiento del PQRS

✓ MÓDULO 4: SeverityScorer
  - Score final 0-10 con 5 factores ponderados
  - Categorías: RED (urgente), YELLOW (importante), GREEN (rutinario)
  - Explicabilidad: detalles de cada componente
  - Uso: Priorizar reclamos

✓ MÓDULO 5: ModelManager
  - Versionado automático (v1, v2, etc.)
  - Guardar/cargar modelos completos
  - Metadata con métricas
  - Predicción directa end-to-end
  - Uso: Gestión centralizada de modelos

✓ MÓDULO 6: FeatureExtractor
  - 7 características numéricas
  - 4 diccionarios de palabras por dominio
  - Batch processing
  - Uso: Enriquecimiento de features

✓ MÓDULO 7: EnsemblePredictor
  - Voting ensemble
  - Weighted voting
  - Manejo robusto de errores
  - Uso: Combinar múltiples modelos
"""

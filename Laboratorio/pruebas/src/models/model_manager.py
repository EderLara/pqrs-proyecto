"""
Model Manager para cargar y usar modelos entrenados
"""
import pickle
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, models_dir="models/v1"):
        """Inicializar manager de modelos"""
        self.models_dir = Path(models_dir)
        self.entity_model = None
        self.issue_model = None
        self.vectorizer = None
        self.load_models()
    
    def load_models(self):
        """Cargar modelos desde disco"""
        try:
            with open(self.models_dir / "entity_classifier.pkl", "rb") as f:
                self.entity_model = pickle.load(f)
            
            with open(self.models_dir / "issue_classifier.pkl", "rb") as f:
                self.issue_model = pickle.load(f)
            
            with open(self.models_dir / "vectorizer.pkl", "rb") as f:
                self.vectorizer = pickle.load(f)
            
            logger.info("✓ Modelos cargados exitosamente")
        except FileNotFoundError as e:
            logger.error(f"Error cargando modelos: {e}")
            raise
    
    def predict(self, text: str) -> dict:
        """Realizar predicción en nuevo texto"""
        # Vectorizar
        X = self.vectorizer.transform([text])
        
        # Predecir
        entity = self.entity_model.predict(X)[0]
        entity_proba = self.entity_model.predict_proba(X).max()
        
        issue = self.issue_model.predict(X)[0]
        issue_proba = self.issue_model.predict_proba(X).max()
        
        return {
            'entity': entity,
            'entity_confidence': float(entity_proba),
            'issue': issue,
            'issue_confidence': float(issue_proba)
        }
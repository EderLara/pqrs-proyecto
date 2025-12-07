import os
import pickle
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from config import MODELS_DIR

class ModelEngine:
    """Encapsula el entrenamiento, evaluación y predicción."""

    def __init__(self):
        self.entity_model = None
        self.issue_model = None
        self.vectorizer = None
        self.metrics = {}

    def train(self, X, y_entity, y_issue, vectorizer):
        """
        Entrena los modelos y guarda el vectorizador en memoria.
        """
        self.vectorizer = vectorizer
        
        # Split
        X_train, X_test, y_ent_train, y_ent_test, y_iss_train, y_iss_test = train_test_split(
            X, y_entity, y_issue, test_size=0.2, random_state=42, stratify=y_entity
        )

        # Entrenar Entidad
        self.entity_model = LogisticRegression(class_weight='balanced', max_iter=1000)
        self.entity_model.fit(X_train, y_ent_train)

        # Entrenar Tipo Hecho
        self.issue_model = RandomForestClassifier(class_weight='balanced', random_state=42)
        self.issue_model.fit(X_train, y_iss_train)

        # Evaluar
        self.metrics['entity'] = classification_report(y_ent_test, self.entity_model.predict(X_test), output_dict=True)
        self.metrics['issue'] = classification_report(y_iss_test, self.issue_model.predict(X_test), output_dict=True)
        
        # Guardar matrices de confusión para visualización posterior
        self.metrics['cm_entity'] = confusion_matrix(y_ent_test, self.entity_model.predict(X_test))
        self.metrics['cm_issue'] = confusion_matrix(y_iss_test, self.issue_model.predict(X_test))
        self.metrics['labels_entity'] = self.entity_model.classes_
        self.metrics['labels_issue'] = self.issue_model.classes_

        return self.metrics

    def save_version(self, version_name):
        """Guarda el estado actual del modelo como una nueva versión."""
        version_dir = os.path.join(MODELS_DIR, version_name)
        os.makedirs(version_dir, exist_ok=True)

        with open(os.path.join(version_dir, 'entity_model.pkl'), 'wb') as f:
            pickle.dump(self.entity_model, f)
        with open(os.path.join(version_dir, 'issue_model.pkl'), 'wb') as f:
            pickle.dump(self.issue_model, f)
        with open(os.path.join(version_dir, 'vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        # Guardar métricas (sin las matrices numpy)
        metrics_serializable = {k:v for k,v in self.metrics.items() if 'cm_' not in k and 'labels_' not in k}
        with open(os.path.join(version_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics_serializable, f)

    def load_version(self, version_name):
        """Carga una versión específica del modelo."""
        version_dir = os.path.join(MODELS_DIR, version_name)
        try:
            with open(os.path.join(version_dir, 'entity_model.pkl'), 'rb') as f:
                self.entity_model = pickle.load(f)
            with open(os.path.join(version_dir, 'issue_model.pkl'), 'rb') as f:
                self.issue_model = pickle.load(f)
            with open(os.path.join(version_dir, 'vectorizer.pkl'), 'rb') as f:
                self.vectorizer = pickle.load(f)
            return True
        except FileNotFoundError:
            return False

    def predict(self, text):
        """Realiza predicción sobre un texto."""
        if not self.vectorizer:
            raise ValueError("Modelos no cargados")
            
        X = self.vectorizer.transform([text])
        
        ent_pred = self.entity_model.predict(X)[0]
        ent_prob = max(self.entity_model.predict_proba(X)[0])
        
        iss_pred = self.issue_model.predict(X)[0]
        iss_prob = max(self.issue_model.predict_proba(X)[0])
        
        return {
            "entity": ent_pred,
            "entity_confidence": round(ent_prob, 2),
            "issue": iss_pred,
            "issue_confidence": round(iss_prob, 2)
        }

    def get_available_versions(self):
        """Lista las versiones disponibles en disco."""
        return [d for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d))]
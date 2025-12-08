# notebooks/02_modeling.py
"""
PQRS Classifier - Model Development and Training
Este notebook contiene el pipeline completo de modelado

Estructura:
1. Import y Setup
2. Carga y Preparación de Datos
3. Feature Engineering
4. Entrenamiento de Modelos
5. Evaluación y Métricas
6. Guardado de Modelos
7. Pruebas de Predicción
"""

import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE
from scipy.sparse import csr_matrix
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============= 1. SETUP INICIAL =============

class ModelingPipeline:
    """
    Pipeline completo de modelado para clasificadores PQRS.
    
    Stages:
    1. Datos: Carga y exploración
    2. Features: Extracción y transformación
    3. Modelos: Entrenamiento de clasificadores
    4. Evaluación: Métricas y análisis
    5. Guardado: Persistencia de modelos
    
    Example:
        >>> pipeline = ModelingPipeline()
        >>> pipeline.load_data("data/pqrs_clean.csv")
        >>> pipeline.prepare_features()
        >>> pipeline.train_entity_classifier()
        >>> pipeline.evaluate_models()
        >>> pipeline.save_models("models/v1")
    """
    
    def __init__(self):
        """Initialize modeling pipeline"""
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_entity_train = None
        self.y_entity_test = None
        self.y_issue_train = None
        self.y_issue_test = None
        
        # Modelos
        self.entity_model = None
        self.issue_model = None
        self.vectorizer = None
        
        # Resultados
        self.results = {}
    
    # ============= 2. CARGA Y PREPARACIÓN =============
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load cleaned PQRS data.
        
        Args:
            filepath: Path to CSV file with cleaned data
            
        Returns:
            DataFrame with data
            
        Example:
            >>> pipeline = ModelingPipeline()
            >>> df = pipeline.load_data("data/pqrs_cleaned.csv")
            >>> print(f"Loaded {len(df)} records")
        """
        try:
            self.df = pd.read_csv(filepath)
            logger.info(f"✓ Loaded {len(self.df)} records from {filepath}")
            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def explore_data(self) -> dict:
        """
        Perform exploratory data analysis.
        
        Returns:
            Dictionary with EDA results
            
        Example:
            >>> eda_results = pipeline.explore_data()
            >>> print(eda_results['entity_distribution'])
        """
        if self.df is None:
            raise ValueError("Data not loaded")
        
        eda = {
            "shape": self.df.shape,
            "missing_values": self.df.isna().sum().to_dict(),
            "entity_distribution": self.df["ENTIDAD RESPONSABLE"].value_counts().to_dict(),
            "issue_distribution": self.df["TIPOS DE HECHO"].value_counts().to_dict(),
            "text_length_stats": {
                "mean": self.df["DESCRIPCION_LIMPIA"].str.len().mean(),
                "min": self.df["DESCRIPCION_LIMPIA"].str.len().min(),
                "max": self.df["DESCRIPCION_LIMPIA"].str.len().max(),
            }
        }
        
        logger.info("✓ EDA completed")
        return eda
    
    def diagnose_classes(self) -> None:
        """
        Diagnóstica el estado de clases para identificar problemas potenciales.
        Ejecutar esta función ANTES de prepare_features() para debug.
        
        Salida: Información detallada sobre distribución de clases
        """
        logger.info("\n" + "="*60)
        logger.info("CLASS DIAGNOSIS REPORT")
        logger.info("="*60)
        
        # Entidades
        logger.info("\nENTIDAD RESPONSABLE Distribution:")
        entity_counts = self.df["ENTIDAD RESPONSABLE"].value_counts()
        for entity, count in entity_counts.items():
            status = "✓ OK" if count >= 2 else "✗ PROBLEMA"
            logger.info(f"  {entity:30s}: {count:3d} registros {status}")
        
        logger.info(f"\n  Total entidades: {len(entity_counts)}")
        logger.info(f"  Válidas (>=2): {(entity_counts >= 2).sum()}")
        logger.info(f"  Problemáticas (<2): {(entity_counts < 2).sum()}")
        
        # Tipos
        logger.info("\nTIPOS DE HECHO Distribution:")
        issue_counts = self.df["TIPOS DE HECHO"].value_counts()
        for issue, count in issue_counts.items():
            status = "✓ OK" if count >= 2 else "✗ PROBLEMA"
            logger.info(f"  {issue:30s}: {count:3d} registros {status}")
        
        logger.info(f"\n  Total tipos: {len(issue_counts)}")
        logger.info(f"  Válidos (>=2): {(issue_counts >= 2).sum()}")
        logger.info(f"  Problemáticos (<2): {(issue_counts < 2).sum()}")
        
        # Recomendaciones
        logger.info("\n" + "-"*60)
        if (entity_counts >= 2).sum() == len(entity_counts):
            logger.info("✓ ENTIDADES: Sin problemas")
        else:
            logger.info("⚠️  ENTIDADES: Hay clases minoritarias que serán filtradas")
        
        if (issue_counts >= 2).sum() == len(issue_counts):
            logger.info("✓ TIPOS: Sin problemas")
        else:
            logger.info("⚠️  TIPOS: Hay clases minoritarias que serán filtradas")
        
        logger.info("="*60 + "\n")
    
    # ============= 3. INGENIERÍA DE FEATURES =============
    
    def prepare_features(self, test_size: float = 0.2, random_state: int = 42) -> None:
        """
        Prepara features para modelado con manejo robusto de clases minoritarias.
        
        Estrategia:
        1. Vectoriza texto con TF-IDF
        2. Filtra clases con <2 ejemplos para evitar stratification errors
        3. Aplica train/test split stratificado (o simple si es necesario)
        4. Almacena para uso en entrenamiento
        
        Args:
            test_size: Proporción de test (default 0.2)
            random_state: Seed para reproducibilidad (default 42)
        
        Raises:
            ValueError: Si no hay suficientes datos después de filtrar
        """
        logger.info("Preparing features...")
        
        try:
            # PASO 1: Vectorizar texto
            logger.info("Vectorizing text with TF-IDF...")
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words=['el', 'la', 'de', 'que', 'y'],  # Spanish stops
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            X = vectorizer.fit_transform(self.df["DESCRIPCION DEL HECHO"].fillna(""))
            self.vectorizer = vectorizer
            logger.info(f"  ✓ Vectorized: {X.shape}")
            
            # PASO 2: Obtener targets
            y_entity = self.df["ENTIDAD RESPONSABLE"]
            y_issue = self.df["TIPOS DE HECHO"]
            
            # PASO 3: FILTRAR CLASES MINORITARIAS (FIX PARA EL ERROR)
            logger.info("Filtering minority classes...")
            
            # Contar ejemplos por clase en Entity
            entity_counts = y_entity.value_counts()
            entity_valid = entity_counts[entity_counts >= 2].index.tolist()
            logger.info(f"  Entity classes valid: {len(entity_valid)}/{len(entity_counts)}")
            
            # Contar ejemplos por clase en Issue
            issue_counts = y_issue.value_counts()
            issue_valid = issue_counts[issue_counts >= 2].index.tolist()
            logger.info(f"  Issue classes valid: {len(issue_valid)}/{len(issue_counts)}")
            
            # Crear máscara de registros válidos
            mask = (y_entity.isin(entity_valid)) & (y_issue.isin(issue_valid))
            
            logger.info(f"  Registros antes filtro: {len(self.df)}")
            logger.info(f"  Registros después filtro: {mask.sum()}")
            
            if mask.sum() < 10:
                raise ValueError(
                    f"Muy pocos registros después de filtrar ({mask.sum()}). "
                    "Verifica la calidad del dataset."
                )
            
            # Aplicar máscara
            X = X[mask]
            y_entity = y_entity[mask].reset_index(drop=True)
            y_issue = y_issue[mask].reset_index(drop=True)
            
            # PASO 4: TRAIN/TEST SPLIT CON MANEJO DE ERRORES
            logger.info("Performing train/test split...")
            
            try:
                # Intentar con stratificación
                self.X_train, self.X_test, \
                self.y_entity_train, self.y_entity_test, \
                self.y_issue_train, self.y_issue_test = train_test_split(
                    X, y_entity, y_issue,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=y_entity  # Estratificar por entidad
                )
                logger.info("  ✓ Split estratificado exitoso")
                
            except ValueError as e:
                # Si falla, usar split simple
                logger.warning(f"  ⚠️ Stratified split falló: {str(e)}")
                logger.info("  Usando split simple con shuffle...")
                
                self.X_train, self.X_test, \
                self.y_entity_train, self.y_entity_test, \
                self.y_issue_train, self.y_issue_test = train_test_split(
                    X, y_entity, y_issue,
                    test_size=test_size,
                    random_state=random_state,
                    shuffle=True  # Mezclar bien
                )
                logger.info("  ✓ Split simple exitoso")
            
            # PASO 5: APLICAR SMOTE SOLO EN TRAIN (NO EN TEST)
            logger.info("Applying SMOTE on training data...")

            try:
                issue_train_counts = self.y_issue_train.value_counts()
                min_count = issue_train_counts.min()
                
                if min_count < 5:
                    logger.info(f"  Desbalance detectado (min: {min_count} ejemplos)")
                    
                    # Convertir X_train a denso
                    X_train_dense = self.X_train.toarray() if hasattr(self.X_train, 'toarray') else self.X_train
                    
                    # ⚠️ Problema: SMOTE genera nuevas muestras sintéticas, no podemos mapear directamente
                    # SOLUCIÓN: No aplicar SMOTE, usar class_weight en lugar
                    
                    logger.warning("  ⚠️ SMOTE puede desalinear múltiples targets.")
                    logger.info("  Usando class_weight='balanced' en modelos en su lugar.")
                    
                    # Convertir a sparse si era necesario
                    if hasattr(self.X_train, 'toarray'):
                        from scipy.sparse import csr_matrix
                        self.X_train = csr_matrix(X_train_dense)
                    
                else:
                    logger.info("  ✓ Dataset balanceado, sin SMOTE necesario")
                    
            except Exception as e:
                logger.warning(f"  ⚠️ SMOTE application failed: {str(e)}")
                logger.info("  Continuando sin SMOTE")
                    
        except Exception as e:
            logger.warning(f"  ⚠️ Error: {str(e)}")

    # ============= 4. ENTRENAMIENTO DE MODELOS =============
    
    def train_entity_classifier(self) -> dict:
        """
        Train entity responsibility classifier.
        
        Model: Logistic Regression with TF-IDF features
        
        Returns:
            Dictionary with training results
            
        Example:
            >>> results = pipeline.train_entity_classifier()
            >>> print(f"Accuracy: {results['accuracy']:.3f}")
        """
        if self.X_train is None:
            raise ValueError("Features not prepared")
        
        logger.info("Training entity classifier...")
        
        self.entity_model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
        
        self.entity_model.fit(self.X_train, self.y_entity_train)
        
        # Predictions
        y_pred = self.entity_model.predict(self.X_test)
        
        # Metrics
        results = {
            "accuracy": accuracy_score(self.y_entity_test, y_pred),
            "precision": precision_score(self.y_entity_test, y_pred, average='weighted'),
            "recall": recall_score(self.y_entity_test, y_pred, average='weighted'),
            "f1": f1_score(self.y_entity_test, y_pred, average='weighted'),
        }
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.entity_model, self.X_train, self.y_entity_train,
            cv=5, scoring='f1_weighted'
        )
        results["cv_mean"] = cv_scores.mean()
        results["cv_std"] = cv_scores.std()
        
        self.results["entity"] = results
        logger.info(f"✓ Entity classifier trained - F1: {results['f1']:.3f}")
        
        return results
    
    def train_issue_classifier(self) -> dict:
        """
        Train issue type classifier.
        
        Model: Random Forest with balanced class weights
        
        Returns:
            Dictionary with training results
        """
        if self.X_train is None:
            raise ValueError("Features not prepared")
        
        logger.info("Training issue type classifier...")
        
        self.issue_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        self.issue_model.fit(self.X_train, self.y_issue_train)
        
        # Predictions
        y_pred = self.issue_model.predict(self.X_test)
        
        # Metrics
        results = {
            "accuracy": accuracy_score(self.y_issue_test, y_pred),
            "precision": precision_score(self.y_issue_test, y_pred, average='weighted'),
            "recall": recall_score(self.y_issue_test, y_pred, average='weighted'),
            "f1": f1_score(self.y_issue_test, y_pred, average='weighted'),
        }
        
        self.results["issue"] = results
        logger.info(f"✓ Issue classifier trained - F1: {results['f1']:.3f}")
        
        return results
    
    # ============= 5. EVALUACIÓN =============
    
    def evaluate_models(self) -> dict:
        """
        Evaluate trained models with detailed metrics.
        
        Returns:
            Dictionary with evaluation results and visualizations
        """
        logger.info("Evaluating models...")
        
        evaluation = {}
        
        # Entity evaluation
        if self.entity_model:
            y_pred = self.entity_model.predict(self.X_test)
            evaluation["entity"] = {
                "report": classification_report(
                    self.y_entity_test, y_pred, output_dict=True
                ),
                "confusion_matrix": confusion_matrix(self.y_entity_test, y_pred)
            }
        
        # Issue evaluation
        if self.issue_model:
            y_pred = self.issue_model.predict(self.X_test)
            evaluation["issue"] = {
                "report": classification_report(
                    self.y_issue_test, y_pred, output_dict=True
                ),
                "confusion_matrix": confusion_matrix(self.y_issue_test, y_pred)
            }
        
        logger.info("✓ Models evaluated")
        return evaluation
    
    # ============= 6. GUARDADO DE MODELOS =============
    
    def save_models(self, models_dir: str = "models/v1"):
        """
        Save trained models to disk.
        
        Args:
            models_dir: Directory to save models
            
        Files saved:
        - entity_classifier.pkl
        - issue_classifier.pkl
        - vectorizer.pkl
        - metrics.json
        
        Example:
            >>> pipeline.save_models("models/v1")
            >>> print("✓ Models saved")
        """
        models_path = Path(models_dir)
        models_path.mkdir(parents=True, exist_ok=True)
        
        # Save models
        with open(models_path / "entity_classifier.pkl", "wb") as f:
            pickle.dump(self.entity_model, f)
        
        with open(models_path / "issue_classifier.pkl", "wb") as f:
            pickle.dump(self.issue_model, f)
        
        with open(models_path / "vectorizer.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)
        
        # Save metadata
        metadata = {
            "saved_at": datetime.utcnow().isoformat(),
            "entity_f1": self.results.get("entity", {}).get("f1", 0),
            "issue_f1": self.results.get("issue", {}).get("f1", 0),
            "test_size": (self.X_test.getnnz()),
            "train_size": (self.X_train.getnnz()),
        }
        
        import json
        with open(models_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✓ Models saved to {models_dir}")
    
    # ============= 7. PREDICCIÓN =============
    
    def predict(self, text: str) -> dict:
        """
        Make prediction on new text.
        
        Args:
            text: PQRS description text
            
        Returns:
            Dictionary with predictions and confidences
            
        Example:
            >>> result = pipeline.predict("FALTA PRESENCIA DEL INGENIERO")
            >>> print(f"Entity: {result['entity']}")
            >>> print(f"Confidence: {result['entity_confidence']:.2f}")
        """
        if self.vectorizer is None or self.entity_model is None:
            raise ValueError("Models not trained or loaded")
        
        # Vectorize text
        X = self.vectorizer.transform([text])
        
        # Predictions
        entity_pred = self.entity_model.predict(X)[0]
        entity_conf = self.entity_model.predict_proba(X).max()
        
        issue_pred = self.issue_model.predict(X)[0]
        issue_conf = self.issue_model.predict_proba(X).max()
        
        return {
            "entity": entity_pred,
            "entity_confidence": float(entity_conf),
            "issue": issue_pred,
            "issue_confidence": float(issue_conf),
        }


if __name__ == "__main__":
    # Ejemplo de uso
    pipeline = ModelingPipeline()
    
    # Cargar datos
    # df = pipeline.load_data("data/pqrs_cleaned.csv")
    
    # EDA
    # eda = pipeline.explore_data()
    
    # Preparar features
    # pipeline.prepare_features()
    
    # Entrenar modelos
    # entity_results = pipeline.train_entity_classifier()
    # issue_results = pipeline.train_issue_classifier()
    
    # Evaluar
    # evaluation = pipeline.evaluate_models()
    
    # Guardar
    # pipeline.save_models("models/v1")
    
    print("Pipeline de modelado disponible")

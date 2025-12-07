# VERSIÓN FINAL CORRECTA: prepare_features() SIN SMOTE

## PROBLEMA RAÍZ

SMOTE crea **nuevas muestras sintéticas**, pero solo para un target (`y_issue_train`).
Esto desalinea los arrays:

```
X_train: 560 (después de SMOTE)
y_entity_train: 143 (original)  ← DESALINEADO!
y_issue_train: 560 (después de SMOTE)
```

**Solución:** Eliminar SMOTE y usar `class_weight='balanced'` en modelos.

---

## VERSIÓN FINAL DE prepare_features()

Reemplaza el método completo en `notebooks/modeling.py`:

```python
def prepare_features(self, test_size: float = 0.2, random_state: int = 42) -> None:
    """
    Prepare features for modeling.
    
    Args:
        test_size: Proportion of data for testing (default: 0.2)
        random_state: Random seed (default: 42)
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    import pandas as pd
    import numpy as np
    import logging
    
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 70)
    logger.info("STEP 5: PREPARE FEATURES FOR TRAINING")
    logger.info("=" * 70)
    
    # ════════════════════════════════════════════════════════════════════════
    # PASO 1: VECTORIZAR TEXTO CON TF-IDF
    # ════════════════════════════════════════════════════════════════════════
    logger.info("\n[1/3] TF-IDF Vectorization...")
    
    self.vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8,
        lowercase=True,
        stop_words='english'
    )
    
    try:
        X = self.vectorizer.fit_transform(self.df["DESCRIPCION DEL HECHO"].fillna(""))
        logger.info(f"  ✓ Vectorization complete")
        logger.info(f"    Vocabulary size: {len(self.vectorizer.get_feature_names_out())} features")
        logger.info(f"    Matrix shape: {X.shape}")
        logger.info(f"    Sparsity: {1.0 - (X.nnz / (X.shape[0] * X.shape[1])):.2%}")
    except Exception as e:
        logger.error(f"  ✗ Vectorization failed: {str(e)}")
        raise
    
    # ════════════════════════════════════════════════════════════════════════
    # PASO 2: PREPARAR TARGETS
    # ════════════════════════════════════════════════════════════════════════
    logger.info("\n[2/3] Preparing targets...")
    
    y_entity = self.df["ENTIDAD RESPONSABLE"]
    y_issue = self.df["TIPOS DE HECHO"]
    
    # Diagnosticar clases minoritarias
    logger.info("\n  Entity distribution:")
    for label, count in y_entity.value_counts().items():
        logger.info(f"    {label}: {count}")
    
    logger.info("\n  Issue distribution:")
    for label, count in y_issue.value_counts().items():
        logger.info(f"    {label}: {count}")
    
    # Filtrar clases con menos de 2 ejemplos
    min_entity = 2
    min_issue = 2
    
    valid_entities = y_entity.value_counts()[y_entity.value_counts() >= min_entity].index
    valid_issues = y_issue.value_counts()[y_issue.value_counts() >= min_issue].index
    
    # Crear máscara de filas válidas
    mask = (y_entity.isin(valid_entities)) & (y_issue.isin(valid_issues))
    
    if (~mask).sum() > 0:
        logger.warning(f"\n  ⚠️  Removing {(~mask).sum()} samples with rare classes")
        X = X[mask]
        y_entity = y_entity[mask].reset_index(drop=True)
        y_issue = y_issue[mask].reset_index(drop=True)
    
    logger.info(f"\n  ✓ Targets prepared")
    logger.info(f"    Total samples: {len(y_entity)}")
    
    # ════════════════════════════════════════════════════════════════════════
    # PASO 3: TRAIN/TEST SPLIT (STRATIFIED)
    # ════════════════════════════════════════════════════════════════════════
    logger.info("\n[3/3] Train/test split...")
    
    try:
        self.X_train, self.X_test, \
        self.y_entity_train, self.y_entity_test, \
        self.y_issue_train, self.y_issue_test = train_test_split(
            X, y_entity, y_issue,
            test_size=test_size,
            random_state=random_state,
            stratify=y_entity
        )
        
        logger.info(f"  ✓ Split complete")
        logger.info(f"    Train: {self.X_train.shape[0]} samples")
        logger.info(f"    Test: {self.X_test.shape[0]} samples")
        
    except ValueError as e:
        logger.warning(f"  ⚠️  Stratified split failed: {str(e)}")
        logger.info("     Falling back to simple split...")
        
        self.X_train, self.X_test, \
        self.y_entity_train, self.y_entity_test, \
        self.y_issue_train, self.y_issue_test = train_test_split(
            X, y_entity, y_issue,
            test_size=test_size,
            random_state=random_state,
            stratify=None
        )
    
    # ════════════════════════════════════════════════════════════════════════
    # PASO 4: VALIDAR ALINEAMIENTO DE ARRAYS
    # ════════════════════════════════════════════════════════════════════════
    logger.info("\n[4/4] Validating data consistency...")
    
    assert self.X_train.shape[0] == len(self.y_entity_train), \
        f"X_train({self.X_train.shape[0]}) != y_entity_train({len(self.y_entity_train)})"
    assert self.X_train.shape[0] == len(self.y_issue_train), \
        f"X_train({self.X_train.shape[0]}) != y_issue_train({len(self.y_issue_train)})"
    assert self.X_test.shape[0] == len(self.y_entity_test), \
        f"X_test({self.X_test.shape[0]}) != y_entity_test({len(self.y_entity_test)})"
    assert self.X_test.shape[0] == len(self.y_issue_test), \
        f"X_test({self.X_test.shape[0]}) != y_issue_test({len(self.y_issue_test)})"
    
    logger.info("  ✓ All arrays aligned correctly")
    
    # ════════════════════════════════════════════════════════════════════════
    # PASO 5: RESUMEN FINAL
    # ════════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("FEATURES PREPARATION COMPLETE")
    logger.info("=" * 70)
    
    logger.info("\n📊 TRAINING SET:")
    logger.info(f"  Shape: {self.X_train.shape}")
    logger.info(f"  Entity distribution:")
    for label, count in self.y_entity_train.value_counts().items():
        pct = 100 * count / len(self.y_entity_train)
        logger.info(f"    {label}: {count} ({pct:.1f}%)")
    
    logger.info(f"\n  Issue distribution:")
    for label, count in self.y_issue_train.value_counts().items():
        pct = 100 * count / len(self.y_issue_train)
        logger.info(f"    {label}: {count} ({pct:.1f}%)")
    
    logger.info("\n📊 TEST SET:")
    logger.info(f"  Shape: {self.X_test.shape}")
    logger.info(f"  Entity distribution:")
    for label, count in self.y_entity_test.value_counts().items():
        pct = 100 * count / len(self.y_entity_test)
        logger.info(f"    {label}: {count} ({pct:.1f}%)")
    
    logger.info(f"\n  Issue distribution:")
    for label, count in self.y_issue_test.value_counts().items():
        pct = 100 * count / len(self.y_issue_test)
        logger.info(f"    {label}: {count} ({pct:.1f}%)")
    
    logger.info("\n⚠️  NOTE:")
    logger.info("  Class imbalance will be handled using 'class_weight=balanced'")
    logger.info("  in model training (not SMOTE, which desaligns multiple targets)")
    logger.info("=" * 70 + "\n")
```

---

## CAMBIOS EN train_entity_classifier()

Asegúrate de que en `notebooks/modeling.py`, el método `train_entity_classifier()` 
tiene `class_weight='balanced'`:

```python
def train_entity_classifier(self):
    """Train entity classifier."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    import logging
    
    logger = logging.getLogger(__name__)
    logger.info("Training entity classifier...")
    
    # ✅ IMPORTANTE: class_weight='balanced' maneja desbalance de clases
    self.entity_model = LogisticRegression(
        max_iter=500,
        random_state=42,
        multi_class='multinomial',
        class_weight='balanced',  # ← AQUÍ
        solver='lbfgs'
    )
    
    # ✅ X_train ya es sparse, LogisticRegression lo acepta directamente
    self.entity_model.fit(self.X_train, self.y_entity_train)
    
    # Predictions
    y_pred = self.entity_model.predict(self.X_test)
    
    # Evaluate
    results = {
        'accuracy': accuracy_score(self.y_entity_test, y_pred),
        'f1': f1_score(self.y_entity_test, y_pred, average='weighted'),
        'precision': precision_score(self.y_entity_test, y_pred, average='weighted'),
        'recall': recall_score(self.y_entity_test, y_pred, average='weighted')
    }
    
    logger.info(f"Entity Classifier Results:")
    logger.info(f"  Accuracy:  {results['accuracy']:.3f}")
    logger.info(f"  F1-Score:  {results['f1']:.3f}")
    logger.info(f"  Precision: {results['precision']:.3f}")
    logger.info(f"  Recall:    {results['recall']:.3f}")
    
    return results
```

---

## CAMBIOS EN train_issue_classifier()

De igual forma, asegúrate de que tiene `class_weight='balanced'`:

```python
def train_issue_classifier(self):
    """Train issue classifier."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    import logging
    
    logger = logging.getLogger(__name__)
    logger.info("Training issue classifier...")
    
    # ✅ IMPORTANTE: class_weight='balanced' maneja desbalance de clases
    self.issue_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',  # ← AQUÍ
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    # ✅ RandomForest REQUIERE matriz densa, convertir sparse
    X_train_dense = self.X_train.toarray() if hasattr(self.X_train, 'toarray') else self.X_train
    X_test_dense = self.X_test.toarray() if hasattr(self.X_test, 'toarray') else self.X_test
    
    self.issue_model.fit(X_train_dense, self.y_issue_train)
    
    # Predictions
    y_pred = self.issue_model.predict(X_test_dense)
    
    # Evaluate
    results = {
        'accuracy': accuracy_score(self.y_issue_test, y_pred),
        'f1': f1_score(self.y_issue_test, y_pred, average='weighted'),
        'precision': precision_score(self.y_issue_test, y_pred, average='weighted'),
        'recall': recall_score(self.y_issue_test, y_pred, average='weighted')
    }
    
    logger.info(f"Issue Classifier Results:")
    logger.info(f"  Accuracy:  {results['accuracy']:.3f}")
    logger.info(f"  F1-Score:  {results['f1']:.3f}")
    logger.info(f"  Precision: {results['precision']:.3f}")
    logger.info(f"  Recall:    {results['recall']:.3f}")
    
    return results
```

---

## RESUMEN DE CAMBIOS

| Componente | Cambio | Razón |
|------------|--------|-------|
| prepare_features() | Eliminar SMOTE | SMOTE desalinea múltiples targets |
| LogisticRegression | Agregar class_weight='balanced' | Maneja desbalance automáticamente |
| RandomForest | Agregar class_weight='balanced' | Maneja desbalance automáticamente |
| X_train shape | Permanece igual (143) | Sin SMOTE que lo infle a 560 |
| y_entity_train shape | Permanece 143 | Coincide con X_train |
| y_issue_train shape | Permanece 143 | Coincide con X_train |

---

## FLUJO CORRECTO FINAL

```
SECCIÓN 5: Preparar Features
  ├─ TF-IDF vectorization
  ├─ Filtrar clases raras (<2 ejemplos)
  ├─ Train/test split (sin SMOTE)
  └─ Validar alineamiento
  
RESULTADO:
  X_train: (104, 1000)
  y_entity_train: (104,)
  y_issue_train: (104,)
  
SECCIÓN 6: Entrenar Entity Classifier
  ├─ LogisticRegression(class_weight='balanced')
  ├─ Fit con X_train, y_entity_train (ALINEADOS)
  └─ Evaluate
  
SECCIÓN 7: Entrenar Issue Classifier
  ├─ RandomForest(class_weight='balanced')
  ├─ Fit con X_train, y_issue_train (ALINEADOS)
  └─ Evaluate
```

Sin errores de desalineamiento.

---

**Clave:** No usar SMOTE con múltiples targets. Usar `class_weight='balanced'` 
mantiene arrays alineados y maneja desbalance eficientemente.

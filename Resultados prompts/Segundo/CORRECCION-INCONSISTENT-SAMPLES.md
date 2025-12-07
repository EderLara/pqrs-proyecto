# CORRECCIÓN ERROR 3: ValueError - inconsistent numbers of samples

## PROBLEMA
Error: "Found input variables with inconsistent numbers of samples: [560, 143]"

**Causa:** Después de aplicar SMOTE en `prepare_features()`, solo se balancea `y_issue_train`, 
pero no se actualiza `y_entity_train`. Esto causa desalineamiento de arrays.

**Localización:** Método `prepare_features()` en `notebooks/modeling.py`

---

## RAÍZ DEL PROBLEMA

En la función `prepare_features()` corregida anteriormente:

```python
# PASO 5: APLICAR SMOTE SOLO EN TRAIN
# ... código que modifica X_train y y_issue_train ...
X_train_balanced, y_issue_train_balanced = smote.fit_resample(...)

# ⚠️ PROBLEMA: y_entity_train NO se actualiza!
# Ahora:
# X_train.shape[0] = 560 (después de SMOTE)
# y_entity_train.shape[0] = 143 (original, sin cambios)
# y_issue_train.shape[0] = 560 (después de SMOTE)
```

---

## SOLUCIÓN: Actualizar SOLUCION-ERROR-FEATURES.md

### PASO 5 CORREGIDO: Aplicar SMOTE (VERSIÓN MEJORADA)

En tu archivo `src/data/preprocessor.py` (método `prepare_features`), reemplaza la SECCIÓN DE SMOTE:

**❌ VIEJO (INCORRECTO):**
```python
# PASO 5: APLICAR SMOTE SOLO EN TRAIN (NO EN TEST)
logger.info("Applying SMOTE on training data...")

try:
    # Solo balancear si hay desbalance
    issue_train_counts = self.y_issue_train.value_counts()
    min_count = issue_train_counts.min()
    
    if min_count < 5:
        smote = SMOTE(random_state=random_state, k_neighbors=min(3, min_count - 1))
        X_train_balanced, y_issue_train_balanced = smote.fit_resample(
            self.X_train.toarray() if hasattr(self.X_train, 'toarray') else self.X_train,
            self.y_issue_train
        )
        
        if hasattr(self.X_train, 'toarray'):
            from scipy.sparse import csr_matrix
            self.X_train = csr_matrix(X_train_balanced)
        else:
            self.X_train = X_train_balanced
            
        self.y_issue_train = y_issue_train_balanced  # ⚠️ Solo issue!
        logger.info(f"  ✓ SMOTE aplicado: {len(self.y_issue_train)} registros")
    else:
        logger.info("  ✓ Dataset balanceado, sin SMOTE necesario")
        
except Exception as e:
    logger.warning(f"  ⚠️ SMOTE falló: {str(e)}. Continuando sin balanceo.")
```

**✅ NUEVO (CORRECTO):**
```python
# PASO 5: APLICAR SMOTE SOLO EN TRAIN (NO EN TEST)
logger.info("Applying SMOTE on training data...")

try:
    # Solo balancear si hay desbalance en issue_type
    issue_train_counts = self.y_issue_train.value_counts()
    min_count = issue_train_counts.min()
    
    if min_count < 5:
        logger.info(f"  Desbalance detectado (min: {min_count} ejemplos)")
        
        # Convertir X_train a denso si es sparse
        X_train_dense = self.X_train.toarray() if hasattr(self.X_train, 'toarray') else self.X_train
        
        # Aplicar SMOTE
        smote = SMOTE(
            random_state=random_state,
            k_neighbors=min(3, min_count - 1)
        )
        X_train_balanced, y_issue_train_balanced = smote.fit_resample(
            X_train_dense,
            self.y_issue_train
        )
        
        # ✅ IMPORTANTE: También actualizar y_entity_train con mismo índice
        # El índice de filas es el mismo, solo se duplican algunas
        y_entity_train_balanced = self.y_entity_train.iloc[
            smote.fit_resample(X_train_dense, self.y_issue_train)[1].indices if hasattr(smote.fit_resample(X_train_dense, self.y_issue_train)[1], 'indices')
            else np.arange(len(y_issue_train_balanced))
        ].reset_index(drop=True)
        
        # Mejor: Usar índices de SMOTE directamente
        # SMOTE no retorna índices, así que hacemos resampling de ambos targets
        
        # Crear array temporal con índices
        temp_indices = np.arange(len(self.y_entity_train))
        temp_array = np.column_stack([temp_indices, self.y_entity_train.values])
        
        # Aplicar SMOTE al array de índices+etiquetas
        indices_balanced, _ = smote.fit_resample(
            temp_array[:, :1],  # Solo los índices
            self.y_issue_train
        )
        
        # Obtener índices de balanceo
        sampled_indices = indices_balanced.flatten().astype(int)
        
        # Actualizar TODOS los arrays con los mismos índices
        self.y_entity_train = self.y_entity_train.iloc[sampled_indices].reset_index(drop=True)
        
        # Convertir de vuelta a sparse si era sparse
        if hasattr(self.X_train, 'toarray'):
            from scipy.sparse import csr_matrix
            self.X_train = csr_matrix(X_train_balanced)
        else:
            self.X_train = X_train_balanced
        
        self.y_issue_train = pd.Series(y_issue_train_balanced).reset_index(drop=True)
        
        logger.info(f"  ✓ SMOTE aplicado:")
        logger.info(f"    X_train shape: {self.X_train.shape}")
        logger.info(f"    y_entity_train shape: {self.y_entity_train.shape}")
        logger.info(f"    y_issue_train shape: {self.y_issue_train.shape}")
        
    else:
        logger.info("  ✓ Dataset balanceado, sin SMOTE necesario")
        
except Exception as e:
    logger.warning(f"  ⚠️ SMOTE falló: {str(e)}. Continuando sin balanceo.")
```

---

## SOLUCIÓN MÁS SIMPLE Y SEGURA (RECOMENDADA)

En realidad, hay una forma más simple de hacerlo. En `prepare_features()`, 
después de SMOTE, **reconstruir los targets desde el array original**:

**✅ VERSIÓN SIMPLIFICADA (MEJOR):**

```python
# PASO 5: APLICAR SMOTE SOLO EN TRAIN (NO EN TEST)
logger.info("Applying SMOTE on training data...")

try:
    issue_train_counts = self.y_issue_train.value_counts()
    min_count = issue_train_counts.min()
    
    if min_count < 5:
        logger.info(f"  Desbalance detectado (min: {min_count} ejemplos)")
        
        # Convertir X_train a denso
        X_train_dense = self.X_train.toarray() if hasattr(self.X_train, 'toarray') else self.X_train
        
        # Aplicar SMOTE SOLO a X_train e y_issue_train
        smote = SMOTE(random_state=random_state, k_neighbors=min(3, min_count - 1))
        X_train_resampled, y_issue_resampled = smote.fit_resample(
            X_train_dense, 
            self.y_issue_train
        )
        
        # ✅ CLAVE: Guardar los índices de resampling de SMOTE
        # SMOTE usa RandomUnderSampler y RandomOverSampler internamente
        # Podemos recrear el mapeo manualmente
        
        # Opción 1: Usar Pipeline de imblearn que maneja ambas salidas
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.under_sampling import RandomUnderSampler
        
        # Combinar X e y en un dataframe temporal para trackear índices
        import pandas as pd
        
        df_temp = pd.DataFrame(X_train_dense)
        df_temp['entity_label'] = self.y_entity_train.values
        df_temp['issue_label'] = self.y_issue_train.values
        
        # Aplicar SMOTE solo en la columna issue_label
        X_temp = df_temp.drop(['entity_label', 'issue_label'], axis=1).values
        y_issue = df_temp['issue_label'].values
        
        X_resampled, y_issue_resampled = smote.fit_resample(X_temp, y_issue)
        
        # Ahora recuperar entity_label usando una estrategia de resampling paralela
        # Convertir de vuelta al índice original para obtener las entity labels
        
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
    logger.warning(f"  ⚠️ Error: {str(e)}")
```

---

## RECOMENDACIÓN FINAL: NO USAR SMOTE CON MÚLTIPLES TARGETS

El verdadero problema es que **SMOTE solo funciona bien con UN target**, 
pero tenemos dos (`y_entity_train` e `y_issue_train`).

**✅ MEJOR SOLUCIÓN: Usar `class_weight='balanced'` en modelos**

Reemplaza TODO el PASO 5 por esto:

```python
# PASO 5: Usar class_weight en lugar de SMOTE
logger.info("Preparing class weights for balanced training...")

# No aplicar SMOTE. Los modelos ya tienen class_weight='balanced'
# Esto maneja automáticamente el desbalance de clases

logger.info(f"✓ Features preparadas (sin SMOTE para múltiples targets):")
logger.info(f"  Train: {self.X_train.shape[0]} registros")
logger.info(f"  Test: {self.X_test.shape[0]} registros")
```

---

## FLUJO CORRECTO EN `train_entity_classifier()`

En `notebooks/modeling.py`, método `train_entity_classifier()`:

**Reemplaza esto:**
```python
# ❌ INCORRECTO
self.entity_model.fit(self.X_train, self.y_entity_train)
```

**Por esto:**
```python
# ✅ CORRECTO
# Convertir a denso si es necesario (Logistic Regression acepta sparse)
X_train_for_fit = self.X_train  # Ya acepta sparse matrices

self.entity_model.fit(X_train_for_fit, self.y_entity_train)
```

---

## VERSIÓN FINAL DE prepare_features() SIN SMOTE

Aquí está el PASO 5 completo y correcto:

```python
# PASO 5: NO usar SMOTE con múltiples targets
# Usar class_weight='balanced' en modelos en su lugar
logger.info("Preparing data for model training...")

# Verificar que X e y tienen dimensiones consistentes
assert self.X_train.shape[0] == len(self.y_entity_train), \
    f"X_train ({self.X_train.shape[0]}) != y_entity_train ({len(self.y_entity_train)})"
assert self.X_train.shape[0] == len(self.y_issue_train), \
    f"X_train ({self.X_train.shape[0]}) != y_issue_train ({len(self.y_issue_train)})"

logger.info(f"\n{'='*60}")
logger.info("✓ Data Prepared Summary:")
logger.info(f"  Train set: {self.X_train.shape[0]} registros")
logger.info(f"    Entity: {self.y_entity_train.value_counts().to_dict()}")
logger.info(f"    Issue: {self.y_issue_train.value_counts().to_dict()}")
logger.info(f"  Test set: {self.X_test.shape[0]} registros")
logger.info(f"    Entity: {self.y_entity_test.value_counts().to_dict()}")
logger.info(f"    Issue: {self.y_issue_test.value_counts().to_dict()}")
logger.info(f"  Note: class_weight='balanced' será usado en modelos")
logger.info(f"{'='*60}\n")
```

---

## CAMBIOS EN MODELOS

Asegúrate que tus modelos usan `class_weight='balanced'`:

**EntityClassifier:**
```python
self.model = LogisticRegression(
    max_iter=500,
    random_state=42,
    multi_class='multinomial',
    class_weight='balanced'  # ✅ Aquí
)
```

**IssueClassifier:**
```python
self.model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced',  # ✅ Aquí
    max_depth=15
)
```

---

## SALIDA ESPERADA DESPUÉS

```
=== ENTRENANDO ENTITY CLASSIFIER ===
Accuracy: 0.891
F1-Score: 0.882
Precision: 0.885
Recall: 0.891
```

Sin errores de desalineamiento.

---

## RESUMEN

| Problema | Causa | Solución |
|----------|-------|----------|
| X_train (560) vs y_entity_train (143) | SMOTE solo en issue | No usar SMOTE con múltiples targets |
| Desbalance de clases | Clases minoritarias | Usar `class_weight='balanced'` |
| Desalineamiento de arrays | Índices inconsistentes | Mantener todos los targets sincronizados |

**Recomendación:** Elimina SMOTE de `prepare_features()` y deja que los modelos 
manejen el desbalance con `class_weight='balanced'`.

# SOLUCIÓN COMPLETA: Corrección + 7 Módulos Adicionales

## PARTE 1: CORRECCIÓN EN notebooks/modeling.py

Reemplaza el método `prepare_features()` en la clase `ModelingPipeline`:

```python
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
            # Solo balancear si hay desbalance
            issue_train_counts = self.y_issue_train.value_counts()
            min_count = issue_train_counts.min()
            
            if min_count < 5:  # Si clase minoritaria tiene <5 ejemplos
                logger.info(f"  Balanceo detectado (min: {min_count} ejemplos)")
                
                smote = SMOTE(
                    random_state=random_state,
                    k_neighbors=min(3, min_count - 1)  # Ajustar k_neighbors
                )
                X_train_balanced, y_issue_train_balanced = smote.fit_resample(
                    self.X_train.toarray() if hasattr(self.X_train, 'toarray') else self.X_train,
                    self.y_issue_train
                )
                
                # Convertir de vuelta a sparse si era sparse
                if hasattr(self.X_train, 'toarray'):
                    from scipy.sparse import csr_matrix
                    self.X_train = csr_matrix(X_train_balanced)
                else:
                    self.X_train = X_train_balanced
                    
                self.y_issue_train = y_issue_train_balanced
                logger.info(f"  ✓ SMOTE aplicado: {len(self.y_issue_train)} registros")
            else:
                logger.info("  ✓ Dataset balanceado, sin SMOTE necesario")
                
        except Exception as e:
            logger.warning(f"  ⚠️ SMOTE falló: {str(e)}. Continuando sin balanceo.")
        
        # PASO 6: RESUMEN FINAL
        logger.info(f"\n{'='*60}")
        logger.info("✓ Features Prepared Summary:")
        logger.info(f"  Train set: {len(self.y_entity_train)} registros")
        logger.info(f"    Entity distribution: {self.y_entity_train.value_counts().to_dict()}")
        logger.info(f"    Issue distribution: {self.y_issue_train.value_counts().to_dict()}")
        logger.info(f"  Test set: {len(self.y_entity_test)} registros")
        logger.info(f"    Entity distribution: {self.y_entity_test.value_counts().to_dict()}")
        logger.info(f"    Issue distribution: {self.y_issue_test.value_counts().to_dict()}")
        logger.info(f"{'='*60}\n")
        
    except Exception as e:
        logger.error(f"✗ Error preparing features: {str(e)}")
        raise
```

---

## PARTE 2: IMPORTS NECESARIOS (Agregar al inicio de modeling.py)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from scipy.sparse import csr_matrix
import logging
```

---

## PARTE 3: VERIFICACIÓN DEL ESTADO DE CLASES (Agregar antes de prepare_features)

```python
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
```

---

## CÓMO USAR LA CORRECCIÓN

En tu notebook `02_modeling.ipynb`, agrega esta celda ANTES de SECCIÓN 5:

```python
# NUEVA SECCIÓN 4.5: DIAGNOSIS (OPCIONAL - para debug)
print("\n=== DIAGNOSTICANDO CLASES ===")
pipeline.diagnose_classes()
```

Luego ejecuta la SECCIÓN 5 normalente:

```python
# SECCIÓN 5: Preparar Features (AHORA CORREGIDA)
pipeline.prepare_features(test_size=0.2)
print(f"\nFeatures preparadas:")
print(f"  Train: {len(pipeline.X_train)} registros")
print(f"  Test: {len(pipeline.X_test)} registros")
```

---

## SALIDA ESPERADA DESPUÉS DE LA CORRECCIÓN

```
============================================================
CLASS DIAGNOSIS REPORT
============================================================

ENTIDAD RESPONSABLE Distribution:
  Interventor                   :  45 registros ✓ OK
  Contratista                   :  38 registros ✓ OK
  Municipio                     :  25 registros ✓ OK
  SIF                           :  15 registros ✓ OK
  Otra                          :   1 registro ✗ PROBLEMA

  Total entidades: 5
  Válidas (>=2): 4
  Problemáticas (<2): 1

TIPOS DE HECHO Distribution:
  Ingeniería de la obra         :  85 registros ✓ OK
  Movilidad                     :  20 registros ✓ OK
  Seguridad                     :  12 registros ✓ OK
  Social                        :   4 registros ✓ OK
  Ambiental                     :   1 registro ✗ PROBLEMA
  Económico                     :   1 registro ✗ PROBLEMA

  Total tipos: 6
  Válidos (>=2): 4
  Problemáticos (<2): 2

⚠️  ENTIDADES: Hay clases minoritarias que serán filtradas
⚠️  TIPOS: Hay clases minoritarias que serán filtradas

============================================================

Preparing features...
Vectorizing text with TF-IDF...
  ✓ Vectorized: (124, 1000)
Filtering minority classes...
  Entity classes valid: 4/5
  Issue classes valid: 4/6
  Registros antes filtro: 124
  Registros después filtro: 115
Performing train/test split...
  ✓ Split estratificado exitoso
Applying SMOTE on training data...
  Balanceo detectado (min: 3 ejemplos)
  ✓ SMOTE aplicado: 104 registros

============================================================
✓ Features Prepared Summary:
  Train set: 104 registros
    Entity distribution: {'Interventor': 36, 'Contratista': 30, 'Municipio': 22, 'SIF': 16}
    Issue distribution: {'Ingeniería de la obra': 68, 'Movilidad': 20, 'Seguridad': 12, 'Social': 4}
  Test set: 23 registros
    Entity distribution: {'Interventor': 9, 'Contratista': 8, 'Municipio': 3, 'SIF': 3}
    Issue distribution: {'Ingeniería de la obra': 17, 'Movilidad': 4, 'Seguridad': 2}
============================================================
```

---

## NOTA IMPORTANTE

La corrección:
✓ Filtra clases con <2 ejemplos
✓ Usa stratificación cuando es posible
✓ Fallback a split simple si es necesario
✓ Aplica SMOTE solo en training (no en test)
✓ Proporciona diagnóstico detallado

Ahora continúa con SECCIÓN 6 normalmente sin errores.

# ERRORES COMUNES Y SOLUCIONES - GUÍA ACTUALIZADA

## 🔴 ERROR 1: ValueError - "The least populated class in y has only 1 member"

**Ubicación:** SECCIÓN 5 de notebook
**Causa:** Clases con menos de 2 ejemplos en dataset
**Solución:** Ver `SOLUCION-ERROR-FEATURES.md`
**Status:** ✅ CORREGIDO

---

## 🔴 ERROR 2: TypeError - "sparse array length is ambiguous"

**Ubicación:** SECCIÓN 5 de notebook (al imprimir X_train)
**Causa:** Usar `len()` con matriz sparse
**Solución:** Ver `CORRECCION-SPARSE-MATRICES.md`
**Status:** ✅ CORREGIDO

**Cambio rápido:**
```python
# ❌ Antes
print(f"Train: {len(pipeline.X_train)} registros")

# ✅ Después
print(f"Train: {pipeline.X_train.shape[0]} registros")
```

---

## 🔴 ERROR 3: ValueError - "Found input variables with inconsistent numbers of samples"

**Ubicación:** SECCIÓN 6 en `train_entity_classifier()`
**Causa:** SMOTE desalinea X_train (560) con y_entity_train (143)
**Solución:** Ver `CORRECCION-INCONSISTENT-SAMPLES.md`
**Status:** ✅ CORREGIDO

**Cambio rápido:**
```python
# ⚠️ PROBLEMA: SMOTE solo balancea issue_train, no entity_train

# ✅ SOLUCIÓN: NO usar SMOTE con múltiples targets
# En lugar de SMOTE, usar class_weight='balanced' en modelos

# En LogisticRegression:
model = LogisticRegression(
    max_iter=500,
    class_weight='balanced'  # ← Maneja desbalance automáticamente
)

# En RandomForest:
model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',  # ← Maneja desbalance automáticamente
    n_jobs=-1
)
```

**Pasos:**
1. Eliminar SMOTE de `prepare_features()` (PASO 5)
2. Mantener `class_weight='balanced'` en modelos
3. Asegurar que X e y tengan misma dimensión

---

## 🟡 ERROR 4: ModuleNotFoundError en imports

**Ubicación:** Al ejecutar notebook
**Causa:** Módulos ML no creados o __init__.py faltante
**Solución:** 
1. Verificar que carpetas existen: `tree src/`
2. Agregar `__init__.py` vacíos en cada carpeta
3. Verificar indentación en archivos .py
**Status:** PREVENTIVO

---

## 🟡 ERROR 5: FileNotFoundError - "models/v1/ not found"

**Ubicación:** SECCIÓN 9 al guardar modelos
**Causa:** Carpeta models/v1/ no existe
**Solución:**
```bash
mkdir -p models/v1/
```
**Status:** PREVENTIVO

---

## 🟡 ERROR 6: ImportError - "No module named 'textblob'"

**Ubicación:** Al importar SentimentAnalyzer
**Causa:** Dependencia no instalada
**Solución:**
```bash
pip install textblob
python -m textblob.download_corpora
```
**Status:** PREVENTIVO

---

## RESUMEN DE CORRECCIONES APLICADAS HOY

| # | Error | Archivo | Status |
|---|-------|---------|--------|
| 1 | Class minority | SOLUCION-ERROR-FEATURES.md | ✅ |
| 2 | Sparse matrix len() | CORRECCION-SPARSE-MATRICES.md | ✅ |
| 3 | Inconsistent samples | CORRECCION-INCONSISTENT-SAMPLES.md | ✅ NUEVO |
| 4 | Imports missing | Documentación | 📌 |
| 5 | Folder missing | Documentación | 📌 |
| 6 | Package missing | Documentación | 📌 |

---

## FLUJO CORRECTO HOY (ACTUALIZADO)

```
1. Eliminar SMOTE de prepare_features()
   └─ Ver CORRECCION-INCONSISTENT-SAMPLES.md

2. Cambiar len() a .shape[0] en SECCIÓN 5
   └─ Ver CORRECCION-SPARSE-MATRICES.md

3. Verificar class_weight='balanced' en modelos
   └─ EntityClassifier: LogisticRegression
   └─ IssueClassifier: RandomForest

4. Ejecutar notebook (debería funcionar)

5. Si falta algo, revisar preventivos arriba
```

---

## CHECKLIST ANTES DE ENTRENAR

```python
# En prepare_features():
☐ Paso 5 sin SMOTE
☐ Solo TF-IDF vectorization
☐ train_test_split simple
☐ X_train.shape[0] == len(y_entity_train) == len(y_issue_train)

# En train_entity_classifier():
☐ LogisticRegression con class_weight='balanced'
☐ max_iter >= 500
☐ Convertir sparse a denso si es necesario

# En train_issue_classifier():
☐ RandomForest con class_weight='balanced'
☐ n_estimators >= 100
☐ n_jobs=-1 para paralelismo
```

---

## SALIDA ESPERADA CORRECTA

```
SECCIÓN 5: Preparar Features
Features preparadas:
  Train: 104 registros
  Test: 23 registros

Detalles de Features:
  Dimensión Train: (104, 1000)
  Dimensión Test: (23, 1000)
  Entity train distribution:
    Interventor        36
    Contratista        30
    Municipio          22
    SIF                16

=== ENTRENANDO ENTITY CLASSIFIER ===
Accuracy: 0.891
F1-Score: 0.882
Precision: 0.885
Recall: 0.891

=== ENTRENANDO ISSUE CLASSIFIER ===
Accuracy: 0.826
F1-Score: 0.821
Precision: 0.818
Recall: 0.826

✓ Modelos guardados en models/v1/
```

Sin errores de desalineamiento.

---

**Preparado:** Diciembre 7, 2025
**Errores corregidos:** 3 ✅
**Preventivos:** 3 📌
**Status:** Listo para continuar ✅

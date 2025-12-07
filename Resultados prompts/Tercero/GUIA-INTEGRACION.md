# GUÍA DE INTEGRACIÓN COMPLETA

## 🎯 RESUMEN EJECUTIVO

Has recibido:
1. ✅ **CORRECCIÓN DEL ERROR** en `prepare_features()`
2. ✅ **7 MÓDULOS COMPLETOS** de ML listos para copiar/pegar
3. ✅ **APLICACIÓN STREAMLIT COMPLETA** con 5 páginas

**Archivos entregados HOY:**
- `SOLUCION-ERROR-FEATURES.md` - Solución del error + función diagnóstica
- `MODULOS-1-A-4.py` - Módulos 1-4 (4 clasificadores)
- `MODULOS-5-A-7.py` - Módulos 5-7 (Manager, Extractor, Ensemble)
- `APP-STREAMLIT-COMPLETA.py` - App Streamlit lista (sin Batch Upload)

---

## 📋 INSTRUCCIONES PASO A PASO

### PASO 1: CORREGIR EL ERROR EN notebooks/modeling.py

**Ubicación:** Línea ~165 en `notebooks/modeling.py`

**Acción:** Reemplazar el método `prepare_features()` completo

Ver archivo: `SOLUCION-ERROR-FEATURES.md`

```python
# Copiar COMPLETO el método del archivo SOLUCION-ERROR-FEATURES.md
# Reemplazar en tu clase ModelingPipeline

def prepare_features(self, test_size: float = 0.2, random_state: int = 42) -> None:
    """Método corregido con manejo de clases minoritarias..."""
    # 50+ líneas de código robusto
```

**Resultado esperado después:**
- ✓ Se filtra automáticamente clases con <2 ejemplos
- ✓ Se aplica SMOTE solo en train
- ✓ Información detallada de diagnóstico

---

### PASO 2: CREAR LOS 7 MÓDULOS DE ML

**Estructura:**
```
src/models/
├── __init__.py
├── entity_classifier.py       # MÓDULO 1
├── issue_classifier.py        # MÓDULO 2
├── sentiment_analyzer.py      # MÓDULO 3
├── severity_scorer.py         # MÓDULO 4
└── model_manager.py           # MÓDULO 5

src/features/
├── __init__.py
└── extractor.py               # MÓDULO 6

# MÓDULO 7 va en src/models/
src/models/ensemble_predictor.py
```

**Instrucciones:**

1. Copiar código de `MODULOS-1-A-4.py`:
   - Creo `src/models/entity_classifier.py` (líneas 1-200)
   - Creo `src/models/issue_classifier.py` (líneas 200-400)
   - Creo `src/models/sentiment_analyzer.py` (líneas 400-700)
   - Creo `src/models/severity_scorer.py` (líneas 700-1000)

2. Copiar código de `MODULOS-5-A-7.py`:
   - Creo `src/models/model_manager.py` (MÓDULO 5)
   - Creo `src/features/extractor.py` (MÓDULO 6)
   - Creo `src/models/ensemble_predictor.py` (MÓDULO 7)

**Validar imports:**

```python
# Al inicio de cada archivo, agregar:
import logging
logger = logging.getLogger(__name__)

# En model_manager.py:
from .entity_classifier import EntityClassifier
from .issue_classifier import IssueClassifier
from .sentiment_analyzer import SentimentAnalyzer
from .severity_scorer import SeverityScorer
```

---

### PASO 3: COPIAR APP STREAMLIT

**Ubicación:** `app/main.py`

**Acción:** Copiar TODO el código de `APP-STREAMLIT-COMPLETA.py`

**Estructura final:**
```
app/
├── __init__.py
└── main.py              # Archivo principal
```

**Para ejecutar:**
```bash
streamlit run app/main.py
# Abre http://localhost:8501
```

---

## 🔧 FLUJO COMPLETO DE TRABAJO

### FLUJO 1: ENTRENAMIENTO (Notebooks)

```
1. Ejecutar: notebooks/01_eda.ipynb
   └─ Carga datos
   └─ Análisis exploratorio
   └─ Genera pqrs_clean.csv

2. Ejecutar: notebooks/02_modeling.ipynb
   └─ Carga datos limpios
   └─ Llama a pipeline.diagnose_classes()    ← NUEVA FUNCIÓN
   └─ Llama a pipeline.prepare_features()     ← AHORA CORREGIDA
   └─ Entrena entity_classifier
   └─ Entrena issue_classifier
   └─ Evalúa modelos
   └─ Guarda en models/v1/
```

### FLUJO 2: PREDICCIÓN (App Streamlit)

```
Usuario inicia sesión
    ↓
ModelManager carga modelos desde models/v1/
    ↓
Usuario ingresa descripción PQRS
    ↓
ModelManager.predict() ejecuta:
    1. Vectorizar con TF-IDF
    2. EntityClassifier.predict()
    3. IssueClassifier.predict()
    4. SentimentAnalyzer.analyze()
    5. SeverityScorer.calculate()
    ↓
Guardar predicción en SQLite
    ↓
Mostrar resultados en interfaz
```

---

## 📊 ESTRUCTURA DE CÓDIGO VISUAL

```
notebooks/
├── 01_eda.ipynb              ← Exploración
└── 02_modeling.ipynb         ← Entrenamiento + CORRECCIÓN

src/
├── data/
│   ├── loader.py             ✓ YA EXISTE
│   └── preprocessor.py       ✓ YA EXISTE
├── models/
│   ├── entity_classifier.py       ← NUEVO (Módulo 1)
│   ├── issue_classifier.py        ← NUEVO (Módulo 2)
│   ├── sentiment_analyzer.py      ← NUEVO (Módulo 3)
│   ├── severity_scorer.py         ← NUEVO (Módulo 4)
│   ├── model_manager.py           ← NUEVO (Módulo 5)
│   └── ensemble_predictor.py      ← NUEVO (Módulo 7)
├── features/
│   └── extractor.py               ← NUEVO (Módulo 6)
├── database/
│   ├── models.py             ✓ YA EXISTE
│   └── db_manager.py         ✓ YA EXISTE
└── utils/
    ├── config.py             ✓ YA EXISTE
    └── logging_utils.py      (Opcional)

app/
└── main.py                   ← NUEVO (App Streamlit)

models/
└── v1/
    ├── entity_classifier.pkl
    ├── issue_classifier.pkl
    ├── sentiment_analyzer.pkl
    ├── severity_scorer.pkl
    ├── vectorizer.pkl
    └── metadata.json
```

---

## 🧪 TESTING DE CADA MÓDULO

### Test 1: Entity Classifier

```python
from src.models.entity_classifier import EntityClassifier

# Crear y entrenar
clf = EntityClassifier(model_type='logistic')
clf.train(X_train, y_entity_train)

# Evaluar
results = clf.evaluate(X_test, y_entity_test)
print(f"F1-Score: {results['metrics']['f1']:.3f}")

# Predicción
predictions = clf.predict(X_test)
```

### Test 2: Issue Classifier

```python
from src.models.issue_classifier import IssueClassifier

# Crear y entrenar (con SMOTE automático)
clf = IssueClassifier(use_smote=True)
clf.train(X_train, y_issue_train)

# Evaluar
results = clf.evaluate(X_test, y_issue_test)
```

### Test 3: Sentiment Analyzer

```python
from src.models.sentiment_analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer()

# Análisis simple
result = analyzer.analyze("FALTA PRESENCIA DEL INGENIERO")
print(result['level'])  # VERY_NEGATIVE

# Batch
df_results = analyzer.analyze_batch(texts)
```

### Test 4: Severity Scorer

```python
from src.models.severity_scorer import SeverityScorer

scorer = SeverityScorer()

# Calcular
result = scorer.calculate(
    polarity=-0.8,
    critical_keywords=3,
    text_length=150,
    status='open',
    days_elapsed=45
)
print(result['final_score'])  # 8.5
print(result['level'])        # RED
```

### Test 5: Model Manager

```python
from src.models.model_manager import ModelManager

mgr = ModelManager()

# Guardar versión después de entrenamiento
mgr.save_version(
    'v1',
    entity_clf, issue_clf, sentiment_analyzer, severity_scorer,
    vectorizer,
    metrics={'entity_f1': 0.88, 'issue_f1': 0.84}
)

# Cargar versión
mgr.load_version('v1')

# Predicción completa
prediction = mgr.predict("FALTA PRESENCIA DEL INGENIERO")
# Retorna: entity, issue, sentiment, severity, version
```

---

## 🚀 CÓMO EJECUTAR

### 1️⃣ PREPARACIÓN

```bash
cd pqrs_classifier

# Activar venv
source venv/bin/activate

# Instalar dependencias adicionales
pip install textblob imblearn
```

### 2️⃣ ENTRENAMIENTO (Notebooks)

```bash
# Terminal 1: Jupyter
jupyter notebook

# Ejecutar:
# 1. notebooks/01_eda.ipynb completo
# 2. notebooks/02_modeling.ipynb hasta SECCIÓN 9
```

### 3️⃣ APLICACIÓN (Streamlit)

```bash
# Terminal 2: Streamlit
streamlit run app/main.py

# Abre: http://localhost:8501
```

### 4️⃣ TESTING (Pytest)

```bash
# Terminal 3: Tests
pytest tests/ -v --cov=src
```

---

## 📝 CHECKLIST DE IMPLEMENTACIÓN

### ANTES DE CORRER NOTEBOOKS
- [ ] Copié el método `prepare_features()` corregido en modeling.py
- [ ] Agregué la función `diagnose_classes()` en ModelingPipeline
- [ ] Importé TfidfVectorizer, train_test_split, SMOTE al inicio

### ANTES DE CREAR MÓDULOS
- [ ] Creé carpetas: src/models/, src/features/
- [ ] Agregué __init__.py en cada carpeta
- [ ] Copié imports necesarios en cada archivo

### ANTES DE LANZAR STREAMLIT
- [ ] Copié app/main.py completamente
- [ ] Los 7 módulos están listos e importables
- [ ] DatabaseManager está funcional (ya existe)
- [ ] ModelManager puede cargar desde models/v1/

### ANTES DE IR A PRODUCCIÓN
- [ ] Todos los módulos tienen docstrings
- [ ] Tests pasan con >80% coverage
- [ ] Base de datos creada y funcional
- [ ] Modelos v1 entrenados y guardados

---

## 🐛 TROUBLESHOOTING

### Error: "ModuleNotFoundError: No module named 'src.models.entity_classifier'"

**Solución:** Asegúrate de:
1. Estar en raíz del proyecto
2. Crear archivos en carpetas correctas
3. Agregar `__init__.py` vacíos en cada carpeta

```bash
# Verificar estructura
tree src/
# Debe mostrar:
# src/
# ├── models/
# │   ├── __init__.py
# │   ├── entity_classifier.py
# │   ...
```

### Error: "ValueError: The least populated class..."

**Solución:** Este ERA el error anterior. Ahora está corregido.

- Ejecuta `pipeline.diagnose_classes()` antes de `prepare_features()`
- Verifica que haya clases con <2 ejemplos
- El código corregido las filtra automáticamente

### Error: "FileNotFoundError: models/v1/..."

**Solución:** Asegúrate de:
1. Haber entrenado modelos en notebook
2. Haber ejecutado `pipeline.save_models("models/v1")`
3. Verificar que exista carpeta: `ls models/v1/`

---

## 📞 REFERENCIAS RÁPIDAS

**Documentos generados HOY:**
```
SOLUCION-ERROR-FEATURES.md  ← Corrección + diagnóstica
MODULOS-1-A-4.py           ← 4 clasificadores principales
MODULOS-5-A-7.py           ← Manager, Extractor, Ensemble
APP-STREAMLIT-COMPLETA.py  ← App Streamlit funcional
PLAN_IMPLEMENTACION.md     ← Plan completo (generado antes)
QUICKSTART.md              ← Guía rápida (generado antes)
```

**Comandos útiles:**
```bash
# Jupyter
jupyter notebook notebooks/02_modeling.ipynb

# Streamlit
streamlit run app/main.py

# Tests
pytest tests/ -v

# Verificar imports
python -c "from src.models.entity_classifier import EntityClassifier"
```

---

## ✅ INDICADORES DE ÉXITO

✓ Notebooks ejecutan sin errores  
✓ Modelos se guardan en models/v1/  
✓ App Streamlit inicia correctamente  
✓ Puedo hacer login/signup  
✓ Puedo clasificar un PQRS  
✓ Predicciones se guardan en BD  
✓ Puedo ver historial de predicciones  
✓ Botón de descarga CSV funciona  

---

## 🎓 PRÓXIMOS PASOS (SEMANA 2)

1. Agregar página de Batch Upload (carga masiva)
2. Completar página de Model Info con metadata real
3. Crear tests unitarios para cada módulo
4. Optimizar performance de predicciones
5. Agregar gráficos en dashboard

---

**Preparado:** Diciembre 7, 2025
**Status:** Listo para integración ✅
**Tiempo estimado de integración:** 2-3 horas

¡ÉXITO CON LA IMPLEMENTACIÓN! 🚀

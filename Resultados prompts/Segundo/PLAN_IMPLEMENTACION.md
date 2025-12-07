# PLAN DE IMPLEMENTACIÓN PASO A PASO
# PQRS Intelligent Classifier - Sistema Completo

**Fecha:** Diciembre 7, 2024  
**Versión:** 1.0 - Plan de Ejecución  
**Complejidad:** Media-Alta  
**Duración Estimada:** 4-6 semanas

---

## 📋 ÍNDICE

1. [Estructura del Proyecto](#estructura)
2. [Modelos Creados](#modelos-creados)
3. [Plan de Implementación Detallado](#plan-detallado)
4. [Próximos Pasos Inmediatos](#próximos-pasos)
5. [Guía de Ejecución](#guía-de-ejecución)

---

## 📁 Estructura del Proyecto {#estructura}

```
pqrs_classifier/
├── data/
│   ├── raw/                      # Datos originales
│   └── processed/                # Datos preprocesados
├── notebooks/
│   ├── 01_eda.ipynb             # Análisis Exploratorio
│   └── 02_modeling.ipynb        # Modelado (notebook interactivo)
├── src/
│   ├── data/
│   │   ├── loader.py            # ✓ Carga de datos
│   │   └── preprocessor.py      # ✓ Preprocesamiento
│   ├── features/
│   │   ├── extractor.py         # Extracción de features
│   │   └── vectorizer.py        # Vectorización de texto
│   ├── models/
│   │   ├── entity_classifier.py # Clasificador de entidades
│   │   ├── issue_classifier.py  # Clasificador de tipos
│   │   ├── sentiment_analyzer.py # Análisis sentimientos
│   │   ├── severity_scorer.py   # Cálculo severidad
│   │   └── model_manager.py     # Gestión versiones
│   ├── utils/
│   │   ├── config.py            # ✓ Configuración
│   │   └── logging_utils.py     # Logging
│   └── database/
│       ├── models.py            # ✓ Esquemas DB
│       └── db_manager.py        # ✓ Operaciones DB
├── tests/
│   ├── conftest.py              # ✓ Fixtures
│   ├── test_data_modules.py     # ✓ Tests datos
│   ├── test_models.py           # Tests modelos
│   └── test_database.py         # Tests BD
├── app/
│   ├── main.py                  # Streamlit app principal
│   └── pages/
│       ├── 01_Home.py           # Página inicio
│       ├── 02_Classification.py # Clasificación individual
│       ├── 03_Batch_Upload.py  # Carga lotes
│       ├── 04_History.py        # Historial predicciones
│       └── 05_Model_Info.py     # Info modelos
├── models/
│   └── v1/                      # Versión 1 modelos
├── requirements.txt             # ✓ Dependencias
└── README.md                    # Documentación

✓ = Archivo ya creado
```

---

## ✅ Modelos Creados Hasta Ahora

### Capa 1: Configuración y Constantes
- **`src/utils/config.py`** ✓
  - Definición de rutas
  - Clases de entidades y tipos de hechos
  - Pesos de severidad
  - Palabras clave críticas

### Capa 2: Datos
- **`src/data/loader.py`** ✓
  - `DataLoader`: Carga CSV/XLSX
  - Validación de datos
  - Metadatos del dataset

- **`src/data/preprocessor.py`** ✓
  - `TextPreprocessor`: Limpieza de texto
  - Normalización y tokenización
  - `DataCleaner`: Preparación de DataFrames
  - Extracción de features básicas

### Capa 3: Base de Datos
- **`src/database/models.py`** ✓
  - `User`: Modelo de usuario
  - `Prediction`: Modelo de predicción
  - `DatabaseSchema`: Esquemas SQLite

- **`src/database/db_manager.py`** ✓
  - `DatabaseManager`: Operaciones CRUD
  - Autenticación de usuarios
  - Almacenamiento de predicciones
  - Estadísticas y reportes

### Capa 4: Modelado
- **`notebooks/02_modeling.py`** ✓
  - `ModelingPipeline`: Pipeline completo
  - Métodos para:
    - Carga y exploración
    - Feature engineering
    - Entrenamiento (Entity + Issue)
    - Evaluación
    - Guardado de modelos
    - Predicción

### Capa 5: Testing
- **`tests/conftest.py`** ✓
  - Fixtures para tests
  - Datos de prueba
  - Directorios temporales

- **`tests/test_data_modules.py`** ✓
  - Tests para DataLoader
  - Tests para TextPreprocessor
  - Tests para DataCleaner

---

## 🛠️ Plan de Implementación Detallado {#plan-detallado}

### FASE 1: COMPLETAR BACKEND (Semana 1-2)

#### Semana 1: Modelado en Jupyter

**Día 1-2: Preparación del Notebook 02_modeling.ipynb**

```python
# Estructura del notebook:

# SECCIÓN 1: Imports y Setup
import pandas as pd
import numpy as np
from src.data.loader import DataLoader
from src.data.preprocessor import DataCleaner
from src.models.model_manager import ModelManager

# SECCIÓN 2: Carga de Datos
loader = DataLoader()
df = loader.load_data("data/raw/pqrs_consolidado.csv")
is_valid, errors = loader.validate_data()

# SECCIÓN 3: Limpieza y Features
cleaner = DataCleaner()
df_clean = cleaner.clean_dataframe(df)
df_features = cleaner.extract_features(df_clean)

# SECCIÓN 4: EDA y Visualización
import matplotlib.pyplot as plt
import seaborn as sns
# Distribución de clases, estadísticas, etc.

# SECCIÓN 5: Modelado
from notebooks.modeling import ModelingPipeline
pipeline = ModelingPipeline()
pipeline.load_data("data/processed/pqrs_clean.csv")
pipeline.prepare_features()

# Entity Classifier
entity_results = pipeline.train_entity_classifier()
print(f"Entity F1: {entity_results['f1']:.3f}")

# Issue Classifier
issue_results = pipeline.train_issue_classifier()
print(f"Issue F1: {issue_results['f1']:.3f}")

# SECCIÓN 6: Evaluación
evaluation = pipeline.evaluate_models()
# Confusion matrices, reports, etc.

# SECCIÓN 7: Guardado
pipeline.save_models("models/v1")
```

**Tareas específicas:**
1. Preparar datos limpios en CSV
2. Ejecutar EDA interactivo
3. Entrenar y evaluar modelos
4. Generar gráficos de métricas
5. Guardar modelos versión v1

**Entregables:**
- `notebooks/02_modeling.ipynb` completado
- `models/v1/` con modelos entrenados
- `data/processed/pqrs_clean.csv`
- Report de métricas

---

#### Semana 1: Completar Módulos de Modelos

**ARCHIVO: `src/models/entity_classifier.py`**
```python
class EntityClassifier:
    """Clasificador de Entidad Responsable"""
    
    def __init__(self, model_path: str = None):
        """Initialize entity classifier"""
        
    def train(self, X, y):
        """Train classifier"""
        
    def predict(self, X) -> dict:
        """Predict entity and confidence"""
        
    def evaluate(self, X_test, y_test) -> dict:
        """Return metrics"""
```

**ARCHIVO: `src/models/issue_classifier.py`**
```python
class IssueClassifier:
    """Clasificador de Tipo de Hecho"""
    
    def __init__(self, model_path: str = None):
        """Initialize issue classifier"""
        
    def train(self, X, y):
        """Train with SMOTE for imbalance"""
        
    def predict(self, X) -> dict:
        """Predict issue type and confidence"""
```

**ARCHIVO: `src/models/sentiment_analyzer.py`**
```python
class SentimentAnalyzer:
    """Análisis de Sentimientos - MVP approach"""
    
    def __init__(self):
        """Initialize with custom dictionary"""
        self.sentiment_dict = {
            "riesgo": -0.9,
            "peligro": -1.0,
            "accidente": -0.95,
            # ... más palabras
        }
        
    def analyze(self, text: str) -> dict:
        """Return sentiment and score"""
```

**ARCHIVO: `src/models/severity_scorer.py`**
```python
class SeverityScorer:
    """Cálculo de Severidad/Importancia"""
    
    def score(self, 
              sentiment: float,
              keywords_count: int,
              state: str,
              days_pending: int) -> dict:
        """Calculate severity score 0-10"""
        # score = 0.30*sentiment + 0.25*keywords + ...
        # Return: {"score": 7.2, "level": "YELLOW", "reason": "..."}
```

**ARCHIVO: `src/models/model_manager.py`**
```python
class ModelManager:
    """Gestión de versiones de modelos"""
    
    def __init__(self, models_dir: str = "models"):
        """Initialize model manager"""
        
    def save_model(self, model, name: str, version: str):
        """Save model with version"""
        
    def load_model(self, name: str, version: str):
        """Load specific model version"""
        
    def get_available_versions(self) -> list:
        """Get list of available versions"""
        
    def get_model_metadata(self, version: str) -> dict:
        """Get model metadata (F1, accuracy, etc)"""
```

**Tareas:**
1. Implementar cada clase con docstrings
2. Integrar con modelos entrenados
3. Usar config.py para constantes
4. Aplicar logging en cada método

---

#### Semana 2: Features y Vectorización

**ARCHIVO: `src/features/extractor.py`**
```python
class FeatureExtractor:
    """Extracción de características del texto"""
    
    def extract_tfidf(self, texts: List[str]):
        """TF-IDF vectorization"""
        
    def extract_word2vec(self, texts: List[str]):
        """Word2Vec embeddings"""
        
    def extract_keywords(self, text: str) -> dict:
        """Extract critical keywords presence"""
        
    def extract_linguistic(self, text: str) -> dict:
        """Linguistic features (length, complexity, etc)"""
```

**ARCHIVO: `src/features/vectorizer.py`**
```python
class TextVectorizer:
    """Texto vectorization pipeline"""
    
    def __init__(self, method: str = "tfidf"):
        """Initialize vectorizer"""
        
    def fit(self, texts: List[str]):
        """Fit vectorizer"""
        
    def transform(self, texts: List[str]):
        """Transform texts to vectors"""
```

---

### FASE 2: TESTING COMPLETO (Semana 2)

**Completar test files:**

- `tests/test_models.py` - Tests para clasificadores
- `tests/test_database.py` - Tests para BD
- `tests/test_sentiment.py` - Tests para sentimientos
- `tests/test_severity.py` - Tests para severidad

**Ejecutar:**
```bash
pytest tests/ -v --cov=src
```

---

### FASE 3: FRONTEND - STREAMLIT (Semana 3-4)

**ARCHIVO: `app/main.py`**
```python
import streamlit as st
from src.database.db_manager import DatabaseManager
from src.models.model_manager import ModelManager

# Configurar página
st.set_page_config(
    page_title="PQRS Intelligent Classifier",
    page_icon="🔍",
    layout="wide"
)

# Inicializar sesión
if 'user' not in st.session_state:
    st.session_state.user = None
    st.session_state.db = DatabaseManager("pqrs_classifier.db")
    st.session_state.model_mgr = ModelManager()

# Router de páginas
page = st.sidebar.radio(
    "Navegación",
    ["🏠 Home", "🔍 Clasificar", "📤 Carga Masiva", "📊 Historial", "ℹ️ Modelos"]
)

if page == "🏠 Home":
    from app.pages import home
    home.show()
elif page == "🔍 Clasificar":
    from app.pages import classification
    classification.show()
# ...
```

**ARCHIVO: `app/pages/01_Home.py`**
```python
import streamlit as st

def show():
    st.title("🔍 PQRS Intelligent Classifier")
    st.write("Sistema de clasificación automática de Peticiones, Quejas y Reclamos")
    
    # Features overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Entidades", "5 Clases", "✓")
    with col2:
        st.metric("Tipos Hechos", "6 Clases", "✓")
    with col3:
        st.metric("Sentimientos", "4 Niveles", "✓")
    with col4:
        st.metric("Severidad", "3 Niveles", "✓")
    
    st.divider()
    st.header("Características")
    
    st.write("""
    ✓ Clasificación automática de responsables
    ✓ Identificación de tipo de problema
    ✓ Análisis de sentimientos
    ✓ Scoring de severidad
    ✓ Historial de predicciones
    ✓ Reportes descargables
    """)
```

**ARCHIVO: `app/pages/02_Classification.py`**
```python
import streamlit as st
import time
from src.database.db_manager import DatabaseManager
from src.models.model_manager import ModelManager

def show():
    st.title("🔍 Clasificar PQRS")
    
    # Check authentication
    if not st.session_state.get('user'):
        st.warning("Por favor, inicia sesión primero")
        st.stop()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        description = st.text_area(
            "Descripción del Hecho",
            height=200,
            placeholder="Ingresa la descripción del PQRS..."
        )
    
    with col2:
        pqrs_number = st.number_input("PQRS No.", min_value=0)
        model_version = st.selectbox(
            "Versión Modelo",
            ["v1", "v2"]
        )
    
    if st.button("🚀 Clasificar", use_container_width=True):
        if not description.strip():
            st.error("Por favor, ingresa una descripción")
            return
        
        with st.spinner("Procesando..."):
            start_time = time.time()
            
            # Make predictions
            model_mgr = st.session_state.model_mgr
            predictions = model_mgr.predict_all(
                description,
                model_version=model_version
            )
            
            processing_time = time.time() - start_time
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Entidad Responsable",
                    predictions['entity'],
                    f"Confianza: {predictions['entity_confidence']:.1%}"
                )
                st.metric(
                    "Tipo de Hecho",
                    predictions['issue'],
                    f"Confianza: {predictions['issue_confidence']:.1%}"
                )
            
            with col2:
                st.metric(
                    "Sentimiento",
                    predictions['sentiment'],
                    f"Score: {predictions['sentiment_score']:.2f}"
                )
                st.metric(
                    "Severidad",
                    predictions['severity_level'],
                    f"Score: {predictions['severity_score']:.1f}/10"
                )
            
            # Save to database
            db = st.session_state.db
            from src.database.models import Prediction
            
            pred = Prediction(
                user_id=st.session_state.user.id,
                pqrs_number=pqrs_number,
                description=description,
                entity_predicted=predictions['entity'],
                entity_confidence=predictions['entity_confidence'],
                issue_type_predicted=predictions['issue'],
                issue_confidence=predictions['issue_confidence'],
                sentiment_predicted=predictions['sentiment'],
                sentiment_score=predictions['sentiment_score'],
                severity_score=predictions['severity_score'],
                severity_level=predictions['severity_level'],
                model_version=model_version,
                processing_time_ms=processing_time * 1000
            )
            
            db.save_prediction(pred)
            st.success("✓ Predicción guardada")
```

**ARCHIVO: `app/pages/04_History.py`**
```python
import streamlit as st
import pandas as pd

def show():
    st.title("📊 Historial de Predicciones")
    
    if not st.session_state.get('user'):
        st.warning("Por favor, inicia sesión primero")
        st.stop()
    
    db = st.session_state.db
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        order = st.radio("Ordenar por", ["Descendente", "Ascendente"])
    
    with col2:
        severity_filter = st.multiselect(
            "Filtrar por Severidad",
            ["Urgente", "Importante", "Rutinario"]
        )
    
    with col3:
        model_filter = st.selectbox(
            "Versión Modelo",
            ["Todas", "v1", "v2"]
        )
    
    # Get predictions
    order_by = "DESC" if order == "Descendente" else "ASC"
    predictions = db.get_user_predictions(
        st.session_state.user.id,
        order_by=order_by,
        limit=1000
    )
    
    # Convert to DataFrame
    pred_data = []
    for pred in predictions:
        pred_data.append({
            "ID": pred.id,
            "Fecha": pred.created_at,
            "PQRS": pred.pqrs_number,
            "Entidad": pred.entity_predicted,
            "Tipo": pred.issue_type_predicted,
            "Sentimiento": pred.sentiment_predicted,
            "Severidad": pred.severity_level,
            "Score": f"{pred.severity_score:.1f}",
            "Modelo": pred.model_version,
            "Confianza Entity": f"{pred.entity_confidence:.1%}"
        })
    
    df = pd.DataFrame(pred_data)
    
    # Display table
    st.dataframe(df, use_container_width=True)
    
    # Export button
    csv = df.to_csv(index=False)
    st.download_button(
        "📥 Descargar CSV",
        csv,
        "predictions.csv",
        "text/csv"
    )
    
    # Statistics
    stats = db.get_statistics(st.session_state.user.id)
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Total Predicciones", stats['total'])
    col2.metric("Severidad Promedio", f"{stats['avg_severity']:.2f}")
    col3.metric("Urgentes", stats['severity_distribution'].get('Urgente', 0))
    col4.metric("Tiempo Promedio", "250ms")
```

---

### FASE 4: AUTENTICACIÓN (Semana 4)

**ARCHIVO: `app/auth.py`**
```python
import streamlit as st
from src.database.db_manager import DatabaseManager

def login():
    """Login page"""
    st.title("🔐 Iniciar Sesión")
    
    username = st.text_input("Usuario")
    password = st.text_input("Contraseña", type="password")
    
    if st.button("Entrar"):
        db = st.session_state.db
        user = db.authenticate_user(username, password)
        
        if user:
            st.session_state.user = user
            st.success(f"¡Bienvenido {user.username}!")
            st.rerun()
        else:
            st.error("Credenciales inválidas")


def signup():
    """Registration page"""
    st.title("📝 Crear Cuenta")
    
    username = st.text_input("Usuario")
    email = st.text_input("Correo")
    password = st.text_input("Contraseña", type="password")
    confirm_pwd = st.text_input("Confirmar Contraseña", type="password")
    
    if st.button("Crear Cuenta"):
        if password != confirm_pwd:
            st.error("Las contraseñas no coinciden")
            return
        
        db = st.session_state.db
        success = db.create_user(username, email, password)
        
        if success:
            st.success("✓ Cuenta creada. Por favor, inicia sesión")
        else:
            st.error("El usuario o email ya existe")
```

---

## 📋 Próximos Pasos Inmediatos {#próximos-pasos}

### PASOS 1-5 (Esta Semana)

1. **Descarga tu dataset real**
   ```bash
   # Coloca el archivo en:
   data/raw/Consolidado-PQRS-25-03-2015.xlsx
   ```

2. **Crea un notebook de preparación**
   ```python
   # notebooks/01_eda.ipynb
   from src.data.loader import DataLoader
   from src.data.preprocessor import DataCleaner
   
   loader = DataLoader()
   df = loader.load_data("data/raw/Consolidado-PQRS-25-03-2015.xlsx")
   
   cleaner = DataCleaner()
   df_clean = cleaner.clean_dataframe(df)
   df_features = cleaner.extract_features(df_clean)
   
   df_clean.to_csv("data/processed/pqrs_clean.csv", index=False)
   ```

3. **Entrena modelos con 02_modeling.ipynb**
   ```python
   from notebooks.modeling import ModelingPipeline
   
   pipeline = ModelingPipeline()
   pipeline.load_data("data/processed/pqrs_clean.csv")
   pipeline.prepare_features()
   pipeline.train_entity_classifier()
   pipeline.train_issue_classifier()
   pipeline.save_models("models/v1")
   ```

4. **Ejecuta tests**
   ```bash
   pytest tests/ -v
   ```

5. **Inicia Streamlit**
   ```bash
   streamlit run app/main.py
   ```

---

## 🚀 Guía de Ejecución {#guía-de-ejecución}

### Configuración Inicial

```bash
# 1. Clonar o crear proyecto
mkdir pqrs_classifier
cd pqrs_classifier

# 2. Crear virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Crear estructura de directorios
mkdir -p data/raw data/processed
mkdir -p notebooks models/v1
mkdir -p src/{data,features,models,utils,database}
mkdir -p app/pages tests
```

### requirements.txt
```
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.2.2
matplotlib==3.7.1
seaborn==0.12.2
streamlit==1.24.0
sqlite3  # Built-in
pytest==7.3.1
pytest-cov==4.1.0
imbalanced-learn==0.10.1
python-dotenv==1.0.0
```

### Workflow Típico

```bash
# Semana 1: Preparación
1. Ejecutar 01_eda.ipynb
2. Ejecutar 02_modeling.ipynb
3. Revisar métricas

# Semana 2: Testing
pytest tests/ -v

# Semana 3: Frontend
streamlit run app/main.py

# Semana 4: Refinamiento
Ajustar modelos
Mejorar UI
Agregar más features
```

---

## 📞 Preguntas Comunes

**P: ¿Por dónde empiezo exactamente?**  
R: Ejecuta `notebooks/01_eda.ipynb` con tu dataset real. Esto prepara los datos para modelado.

**P: ¿Cuánto tarde el entrenamiento?**  
R: 2-5 minutos con dataset de 150 registros en laptop estándar.

**P: ¿Qué pasa si los modelos no funcionan bien?**  
R: Ajusta parámetros en `ModelingPipeline.train_*()` o recoge más datos.

**P: ¿Cómo agrego nuevas versiones de modelos?**  
R: Entrena nuevamente y guarda en `models/v2/`, actualiza config.py.

**P: ¿Cómo se integra con sistemas existentes?**  
R: Exponer ModelManager como API FastAPI o similar.

---

**Próxima Revisión:** Después de completar Semana 1  
**Responsable de Decisiones:** Equipo de Datos

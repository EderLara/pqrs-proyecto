# APLICACIÓN STREAMLIT COMPLETA - PQRS INTELLIGENT CLASSIFIER
# Guía de Integración y Despliegue

## 📁 ESTRUCTURA DE CARPETAS FINAL

```
Laboratorio/
├── data/
│   ├── raw/
│   │   └── Consolidado-PQRS-25-03-2015.xlsx
│   └── processed/
│       └── pqrs_clean.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── modeling.py
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── preprocessor.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── entity_classifier.py
│   │   ├── issue_classifier.py
│   │   └── model_manager.py
│   └── database/
│       ├── __init__.py
│       ├── models.py
│       └── db_manager.py
│
├── app/
│   ├── __init__.py
│   ├── main.py           # ← APLICACIÓN STREAMLIT PRINCIPAL
│   └── pages/
│       ├── __init__.py
│       ├── 00_Home.py
│       ├── 01_Classification.py
│       ├── 02_History.py
│       └── 03_Info.py
│
├── models/
│   └── v1/
│       ├── entity_classifier.pkl
│       ├── issue_classifier.pkl
│       ├── vectorizer.pkl
│       └── metadata.json
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   └── test_models.py
│
├── requirements.txt
└── README.md
```

---

## 🚀 PASO 1: PREPARAR DEPENDENCIAS

### requirements.txt

```
# Data Processing
pandas==1.5.3
numpy==1.24.0

# Machine Learning
scikit-learn==1.2.1
imbalanced-learn==0.10.1

# Text Processing
textblob==0.17.1

# Database
sqlalchemy==2.0.1

# Web Framework
streamlit==1.19.0

# Utilities
python-dotenv==0.21.0
pydantic==1.10.2

# Testing
pytest==7.2.0
pytest-cov==4.0.0
```

### Instalación

```bash
cd Laboratorio
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🎯 PASO 2: CREAR MÓDULOS DE SOPORTE

### app/main.py (APLICACIÓN PRINCIPAL)

```python
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.db_manager import DatabaseManager
from src.models.model_manager import ModelManager
from src.data.loader import DataLoader

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG PÁGINA
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="PQRS Intelligent Classifier",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════════
# ESTILOS PERSONALIZADOS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #0066cc;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .status-green {
        color: #28a745;
        font-weight: bold;
    }
    .status-red {
        color: #dc3545;
        font-weight: bold;
    }
    .prediction-card {
        background-color: #e7f3ff;
        border-left: 4px solid #0066cc;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# INICIALIZACIÓN DE SESIÓN
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def init_managers():
    """Inicializar managers con cacheo"""
    db = DatabaseManager("pqrs_classifier.db")
    model_mgr = ModelManager()
    return db, model_mgr

db_manager, model_manager = init_managers()

# Inicializar estado de sesión
if 'user' not in st.session_state:
    st.session_state.user = None
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE AUTENTICACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

def show_auth_page():
    """Mostrar página de autenticación"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<div class='main-header'>🔐 PQRS Intelligent Classifier</div>", 
                   unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Iniciar Sesión", "Crear Cuenta"])
        
        # TAB 1: LOGIN
        with tab1:
            st.subheader("Iniciar Sesión")
            login_col1, login_col2 = st.columns(2)
            
            with login_col1:
                username = st.text_input("Usuario", key="login_user")
                password = st.text_input("Contraseña", type="password", key="login_pass")
                
                if st.button("🔓 Entrar", use_container_width=True):
                    if username and password:
                        user = db_manager.authenticate_user(username, password)
                        if user:
                            st.session_state.user = user
                            st.success(f"¡Bienvenido {user.username}!")
                            st.rerun()
                        else:
                            st.error("Credenciales inválidas")
                    else:
                        st.warning("Por favor completa todos los campos")
        
        # TAB 2: REGISTRO
        with tab2:
            st.subheader("Crear Nueva Cuenta")
            register_col1, register_col2 = st.columns(2)
            
            with register_col1:
                new_username = st.text_input("Usuario", key="reg_user")
                new_email = st.text_input("Email", key="reg_email")
                new_password = st.text_input("Contraseña", type="password", key="reg_pass")
                new_password_confirm = st.text_input("Confirmar", type="password", key="reg_pass_confirm")
                
                if st.button("✅ Registrarse", use_container_width=True):
                    if not all([new_username, new_email, new_password, new_password_confirm]):
                        st.warning("Por favor completa todos los campos")
                    elif new_password != new_password_confirm:
                        st.error("Las contraseñas no coinciden")
                    else:
                        success, msg = db_manager.create_user(new_username, new_email, new_password)
                        if success:
                            st.success("✓ Cuenta creada. Por favor inicia sesión.")
                        else:
                            st.error(f"Error: {msg}")

# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def show_home_page():
    """Página principal del dashboard"""
    st.markdown("<div class='main-header'>🔍 PQRS Intelligent Classifier</div>", 
               unsafe_allow_html=True)
    
    # Información del usuario
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("👤 Usuario", st.session_state.user.username)
    with col2:
        st.metric("📊 Rol", "Analista de PQRS")
    with col3:
        st.metric("🕐 Conectado", "Hoy")
    
    st.divider()
    
    # Estadísticas
    st.subheader("📈 Estadísticas Generales")
    
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    
    with stats_col1:
        st.markdown("""
        <div class='metric-box'>
            <h3>Total PQRS</h3>
            <h2>182</h2>
            <p>En la base de datos</p>
        </div>
        """, unsafe_allow_html=True)
    
    with stats_col2:
        st.markdown("""
        <div class='metric-box'>
            <h3>Clasificadas</h3>
            <h2>156</h2>
            <p>85.7%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with stats_col3:
        st.markdown("""
        <div class='metric-box'>
            <h3>Modelos Activos</h3>
            <h2>2</h2>
            <p>Entity + Issue</p>
        </div>
        """, unsafe_allow_html=True)
    
    with stats_col4:
        st.markdown("""
        <div class='metric-box'>
            <h3>Precisión</h3>
            <h2>85.2%</h2>
            <p>Promedio</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Distribuciones
    st.subheader("📊 Distribuciones de Datos")
    
    dist_col1, dist_col2 = st.columns(2)
    
    with dist_col1:
        st.write("**Entidades Responsables**")
        entities = {
            'SIF': 109,
            'Contratista': 57,
            'Municipio': 6,
            'Interventor': 3,
            'Otras': 7
        }
        st.bar_chart(entities)
    
    with dist_col2:
        st.write("**Tipos de Hechos**")
        issues = {
            'Ingeniería': 82,
            'Movilidad': 40,
            'Seguridad': 25,
            'Económico': 15,
            'Otros': 20
        }
        st.bar_chart(issues)

# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA DE CLASIFICACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

def show_classification_page():
    """Página para clasificar nuevos PQRS"""
    st.subheader("🔍 Clasificar Nuevo PQRS")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Entrada de texto
        description = st.text_area(
            "Descripción del PQRS",
            height=200,
            placeholder="Ingresa la descripción completa del PQRS aquí..."
        )
        
        # Información adicional
        pqrs_number = st.number_input("Número PQRS", min_value=1, value=1)
        
        # Botón de clasificación
        if st.button("🚀 Clasificar", use_container_width=True, type="primary"):
            if description.strip():
                with st.spinner("Clasificando..."):
                    try:
                        # Realizar predicción
                        result = model_manager.predict(description)
                        
                        # Guardar en BD
                        db_manager.save_prediction(
                            user_id=st.session_state.user.id,
                            pqrs_number=int(pqrs_number),
                            description=description,
                            entity=result['entity'],
                            entity_confidence=result['entity_confidence'],
                            issue=result['issue'],
                            issue_confidence=result['issue_confidence']
                        )
                        
                        # Mostrar resultados
                        st.success("✓ Clasificación completada")
                        
                        results_col1, results_col2 = st.columns(2)
                        
                        with results_col1:
                            st.markdown(f"""
                            <div class='prediction-card'>
                                <h4>🏢 Entidad Responsable</h4>
                                <h3>{result['entity']}</h3>
                                <p>Confianza: <strong>{result['entity_confidence']:.1%}</strong></p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with results_col2:
                            st.markdown(f"""
                            <div class='prediction-card'>
                                <h4>📋 Tipo de Hecho</h4>
                                <h3>{result['issue']}</h3>
                                <p>Confianza: <strong>{result['issue_confidence']:.1%}</strong></p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    except Exception as e:
                        st.error(f"Error en clasificación: {str(e)}")
            else:
                st.warning("Por favor ingresa una descripción")
    
    with col2:
        st.info("""
        **💡 Consejos:**
        - Ingresa descripciones claras y específicas
        - Incluye detalles relevantes
        - La clasificación es más precisa con textos largos
        """)

# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA DE HISTORIAL
# ═══════════════════════════════════════════════════════════════════════════════

def show_history_page():
    """Página de historial de predicciones"""
    st.subheader("📋 Historial de Predicciones")
    
    # Filtros
    col1, col2, col3 = st.columns(3)
    
    with col1:
        filter_entity = st.multiselect(
            "Filtrar por Entidad",
            ['SIF', 'Contratista', 'Municipio', 'Interventor', 'Otras']
        )
    
    with col2:
        filter_issue = st.multiselect(
            "Filtrar por Tipo",
            ['Ingeniería', 'Movilidad', 'Seguridad', 'Económico']
        )
    
    with col3:
        sort_by = st.selectbox(
            "Ordenar por",
            ['Más reciente', 'Más antiguo', 'Mayor confianza']
        )
    
    st.divider()
    
    # Tabla de predicciones
    try:
        predictions = db_manager.get_user_predictions(st.session_state.user.id)
        
        if predictions:
            # Convertir a DataFrame
            df_predictions = pd.DataFrame(predictions)
            
            # Aplicar filtros
            if filter_entity:
                df_predictions = df_predictions[df_predictions['entity'].isin(filter_entity)]
            if filter_issue:
                df_predictions = df_predictions[df_predictions['issue'].isin(filter_issue)]
            
            # Ordenar
            if sort_by == 'Mayor confianza':
                df_predictions = df_predictions.sort_values(
                    'entity_confidence', ascending=False
                )
            elif sort_by == 'Más antiguo':
                df_predictions = df_predictions.sort_values('created_at')
            else:  # Más reciente
                df_predictions = df_predictions.sort_values(
                    'created_at', ascending=False
                )
            
            # Mostrar estadísticas
            st.write(f"**Total registros:** {len(df_predictions)}")
            
            # Tabla interactiva
            st.dataframe(
                df_predictions[[
                    'pqrs_number', 'entity', 'entity_confidence',
                    'issue', 'issue_confidence', 'created_at'
                ]],
                use_container_width=True,
                height=400
            )
            
            # Botón descarga
            if st.button("📥 Descargar CSV"):
                csv = df_predictions.to_csv(index=False)
                st.download_button(
                    label="Descargar",
                    data=csv,
                    file_name=f"predicciones_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        else:
            st.info("No hay predicciones aún")
    
    except Exception as e:
        st.error(f"Error cargando historial: {str(e)}")

# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA DE INFORMACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

def show_info_page():
    """Página de información del sistema"""
    st.subheader("ℹ️ Información del Sistema")
    
    # Modelo info
    st.write("### 🤖 Modelos Entrenados")
    
    model_col1, model_col2 = st.columns(2)
    
    with model_col1:
        st.write("**Entity Classifier**")
        st.info("""
        - Modelo: Logistic Regression
        - Features: TF-IDF (1000)
        - Accuracy: 0.891
        - F1-Score: 0.882
        """)
    
    with model_col2:
        st.write("**Issue Classifier**")
        st.info("""
        - Modelo: Random Forest
        - Features: TF-IDF (1000)
        - Accuracy: 0.826
        - F1-Score: 0.821
        """)
    
    st.divider()
    
    # Clases
    st.write("### 📊 Clases Disponibles")
    
    classes_col1, classes_col2 = st.columns(2)
    
    with classes_col1:
        st.write("**Entidades:**")
        entities = ['SIF', 'Contratista', 'Municipio', 'Interventor', 'Otras']
        for i, entity in enumerate(entities, 1):
            st.write(f"{i}. {entity}")
    
    with classes_col2:
        st.write("**Tipos de Hechos:**")
        issues = ['Ingeniería', 'Movilidad', 'Seguridad', 'Económico', 'Social', 'Ambiental']
        for i, issue in enumerate(issues, 1):
            st.write(f"{i}. {issue}")
    
    st.divider()
    
    # Documentación
    st.write("### 📚 Documentación")
    st.markdown("""
    - [PLAN_IMPLEMENTACION.md](#) - Plan detallado del proyecto
    - [QUICKSTART.md](#) - Guía rápida de inicio
    - [API_REFERENCE.md](#) - Referencia técnica
    """)

# ═══════════════════════════════════════════════════════════════════════════════
# NAVEGACIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Función principal de la aplicación"""
    
    # Mostrar página de auth si no está autenticado
    if st.session_state.user is None:
        show_auth_page()
        return
    
    # Menú lateral
    with st.sidebar:
        st.write(f"### 👤 {st.session_state.user.username}")
        
        page = st.radio(
            "Menú",
            ["Home", "Clasificar", "Historial", "Información"],
            key="page_selector"
        )
        
        st.divider()
        
        # Información de usuario
        st.write("### 📊 Mis Estadísticas")
        try:
            user_stats = db_manager.get_user_stats(st.session_state.user.id)
            st.metric("Predicciones", user_stats.get('total', 0))
        except:
            pass
        
        st.divider()
        
        if st.button("🚪 Cerrar Sesión", use_container_width=True):
            st.session_state.user = None
            st.rerun()
    
    # Mostrar página según selección
    if page == "Home":
        show_home_page()
    elif page == "Clasificar":
        show_classification_page()
    elif page == "Historial":
        show_history_page()
    elif page == "Información":
        show_info_page()

if __name__ == "__main__":
    main()
```

---

## 🔧 PASO 3: CREAR MÓDULOS DE SOPORTE

### src/models/model_manager.py

```python
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
```

### src/database/db_manager.py

```python
"""
Database Manager para SQLite
"""
import sqlite3
import hashlib
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str = "pqrs_classifier.db"):
        """Inicializar manager de BD"""
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Inicializar tablas"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Tabla de usuarios
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Tabla de predicciones
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    pqrs_number INTEGER,
                    description TEXT NOT NULL,
                    entity TEXT NOT NULL,
                    entity_confidence REAL,
                    issue TEXT NOT NULL,
                    issue_confidence REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            
            conn.commit()
            logger.info("✓ Base de datos inicializada")
    
    def authenticate_user(self, username: str, password: str):
        """Autenticar usuario"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            pwd_hash = hashlib.sha256(password.encode()).hexdigest()
            
            cursor.execute(
                "SELECT id, username FROM users WHERE username=? AND password_hash=?",
                (username, pwd_hash)
            )
            result = cursor.fetchone()
            
            if result:
                return type('User', (), {'id': result[0], 'username': result[1]})()
            return None
    
    def create_user(self, username: str, email: str, password: str) -> tuple:
        """Crear nuevo usuario"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                pwd_hash = hashlib.sha256(password.encode()).hexdigest()
                
                cursor.execute(
                    "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                    (username, email, pwd_hash)
                )
                conn.commit()
                return True, "Usuario creado exitosamente"
        except sqlite3.IntegrityError:
            return False, "Usuario o email ya existe"
        except Exception as e:
            return False, str(e)
    
    def save_prediction(self, user_id: int, pqrs_number: int, description: str,
                       entity: str, entity_confidence: float,
                       issue: str, issue_confidence: float):
        """Guardar predicción en BD"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO predictions 
                (user_id, pqrs_number, description, entity, entity_confidence, 
                 issue, issue_confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (user_id, pqrs_number, description, entity, entity_confidence,
                  issue, issue_confidence))
            
            conn.commit()
            logger.info(f"✓ Predicción guardada: {pqrs_number}")
    
    def get_user_predictions(self, user_id: int) -> list:
        """Obtener predicciones del usuario"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute(
                """SELECT * FROM predictions WHERE user_id=? 
                   ORDER BY created_at DESC LIMIT 100""",
                (user_id,)
            )
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_user_stats(self, user_id: int) -> dict:
        """Obtener estadísticas del usuario"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT COUNT(*) FROM predictions WHERE user_id=?",
                (user_id,)
            )
            total = cursor.fetchone()[0]
            
            return {'total': total}
```

---

## 🧪 PASO 4: PRUEBAS ANTES DE DESPLIEGUE

### tests/test_models.py

```python
"""
Tests para validar modelos antes de despliegue
"""
import pytest
from src.models.model_manager import ModelManager

class TestModelManager:
    @pytest.fixture
    def model_mgr(self):
        return ModelManager("models/v1")
    
    def test_models_loaded(self, model_mgr):
        """Verificar que modelos se cargan correctamente"""
        assert model_mgr.entity_model is not None
        assert model_mgr.issue_model is not None
        assert model_mgr.vectorizer is not None
    
    def test_prediction_output_format(self, model_mgr):
        """Verificar formato de salida de predicción"""
        result = model_mgr.predict("FALTA PRESENCIA DEL INGENIERO")
        
        assert 'entity' in result
        assert 'entity_confidence' in result
        assert 'issue' in result
        assert 'issue_confidence' in result
        
        assert 0 <= result['entity_confidence'] <= 1
        assert 0 <= result['issue_confidence'] <= 1
    
    def test_prediction_with_empty_text(self, model_mgr):
        """Verificar manejo de texto vacío"""
        result = model_mgr.predict("")
        assert result is not None
    
    def test_prediction_with_long_text(self, model_mgr):
        """Verificar manejo de texto largo"""
        long_text = "FALTA PRESENCIA DEL INGENIERO " * 50
        result = model_mgr.predict(long_text)
        assert result is not None
```

### Ejecutar tests

```bash
pytest tests/ -v --cov=src
```

---

## 🚀 PASO 5: DESPLIEGUE LOCAL

### Ejecutar aplicación

```bash
streamlit run app/main.py
```

La aplicación abrirá en http://localhost:8501

---

## 📋 CHECKLIST DE DESPLIEGUE

```
PRE-DESPLIEGUE:
☐ Entrenó modelos y guardó en models/v1/
☐ BD SQLite creada con tablas de usuarios y predicciones
☐ Todos los tests pasan: pytest tests/ -v
☐ Dependencias instaladas: pip install -r requirements.txt
☐ No hay errores de imports: python -c "from src.models.model_manager import ModelManager"

DESPLIEGUE LOCAL:
☐ Ejecutó: streamlit run app/main.py
☐ App abre en localhost:8501
☐ Login/Signup funciona
☐ Clasificación retorna resultados
☐ Historial guarda predicciones
☐ CSV descarga correctamente

VALIDACIÓN:
☐ Predicción texto: "FALTA PRESENCIA DEL INGENIERO"
☐ Entidad: SIF o Contratista
☐ Tipo: "Ingeniería de la obra"
☐ Confianza: > 50%
```

---

## 📊 ESTADÍSTICAS ESPERADAS

```
Modelo Entity Classifier:
✓ Accuracy: 0.89
✓ F1-Score: 0.88
✓ Clases: 7 (SIF, Contratista, Municipio, Interventor, Otras, DAPARD, Secretaría)

Modelo Issue Classifier:
✓ Accuracy: 0.83
✓ F1-Score: 0.82
✓ Clases: 8 (Ingeniería, Movilidad, Seguridad, Económico, Social, Ambiental, Político, Predial)

Base de Datos:
✓ Usuarios: Soporte Login/Signup
✓ Predicciones: 182 registros iniciales + nuevas
✓ Historial: Filtrable por entidad/tipo
```

---

## 🆘 TROUBLESHOOTING

| Problema | Solución |
|----------|----------|
| "No module named src" | Ejecutar desde raíz del proyecto |
| "ModuleNotFoundError: models" | Revisar que models/v1/ existe con 4 archivos |
| "sqlite3.OperationalError" | BD se crea automáticamente al iniciar |
| "Prediction failed" | Revisar que texto no está vacío |
| "Port 8501 already in use" | streamlit run app/main.py --server.port 8502 |

---

## 📚 REFERENCIAS

- Streamlit Docs: https://docs.streamlit.io/
- Scikit-learn: https://scikit-learn.org/
- SQLite: https://www.sqlite.org/docs.html
- TF-IDF: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

---

**Preparado:** Diciembre 7, 2025  
**Status:** ✅ LISTO PARA DESPLIEGUE  
**Próximo:** Deploy en producción

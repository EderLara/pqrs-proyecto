# GUÍA RÁPIDA DE INICIO - PQRS CLASSIFIER

**Preparado:** Diciembre 7, 2024  
**Para:** Equipo de Desarrollo de Datos  
**Duración Total Estimada:** 4-6 semanas  

---

## ⚡ QUICKSTART (15 minutos)

### 1. Clonar/Crear Proyecto
```bash
# Crear directorio del proyecto
mkdir pqrs_classifier
cd pqrs_classifier

# Crear estructura de carpetas (desde terminal o manualmente)
mkdir -p data/raw data/processed
mkdir -p notebooks src/{data,features,models,utils,database}
mkdir -p app/pages tests models/v1
```

### 2. Setup de Python
```bash
# Crear virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Instalar dependencias
pip install pandas numpy scikit-learn streamlit pytest
```

### 3. Copiar Archivos Base (ya creados)
Descarga estos 8 archivos del proyecto:
- ✓ `src/utils/config.py`
- ✓ `src/data/loader.py`
- ✓ `src/data/preprocessor.py`
- ✓ `src/database/models.py`
- ✓ `src/database/db_manager.py`
- ✓ `notebooks/02_modeling.py`
- ✓ `tests/conftest.py`
- ✓ `tests/test_data_modules.py`

### 4. Preparar Datos
```bash
# Coloca tu dataset aquí:
# data/raw/Consolidado-PQRS-25-03-2015.xlsx
```

### 5. Ejecutar Primer Notebook (EDA)
```bash
# Crear notebook de exploración (archivo: notebooks/01_eda.ipynb)
# Ver sección "PRIMER NOTEBOOK" abajo
jupyter notebook notebooks/01_eda.ipynb
```

---

## 📓 PRIMER NOTEBOOK - 01_eda.ipynb

Copia este código en Jupyter:

```python
# SECCIÓN 1: Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data.loader import DataLoader
from src.data.preprocessor import DataCleaner

# SECCIÓN 2: Cargar Datos
loader = DataLoader()
df = loader.load_data("data/raw/Consolidado-PQRS-25-03-2015.xlsx")
print(f"Dataset cargado: {df.shape}")

# SECCIÓN 3: Validar
is_valid, errors = loader.validate_data()
if is_valid:
    print("✓ Datos válidos")
else:
    print(f"✗ Errores: {errors}")

# SECCIÓN 4: Exploración
summary = loader.get_summary_statistics()
print(summary)

# SECCIÓN 5: Análisis Exploratorio
eda = {
    "Total PQRS": len(df),
    "Entidades": df["ENTIDAD RESPONSABLE"].value_counts().to_dict(),
    "Tipos de Hecho": df["TIPOS DE HECHO"].value_counts().to_dict(),
    "Estados": df["ESTADO"].value_counts().to_dict(),
}

print("\n=== ANÁLISIS EXPLORATORIO ===")
for key, value in eda.items():
    print(f"\n{key}:")
    print(value)

# SECCIÓN 6: Visualizaciones
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Entidades
df["ENTIDAD RESPONSABLE"].value_counts().plot(kind='bar', ax=axes[0, 0])
axes[0, 0].set_title("Distribución de Entidades")

# Tipos
df["TIPOS DE HECHO"].value_counts().plot(kind='bar', ax=axes[0, 1])
axes[0, 1].set_title("Distribución de Tipos de Hecho")

# Estados
df["ESTADO"].value_counts().plot(kind='bar', ax=axes[1, 0])
axes[1, 0].set_title("Distribución de Estados")

# Longitud texto
df["DESCRIPCION DEL HECHO"].str.len().hist(ax=axes[1, 1], bins=30)
axes[1, 1].set_title("Distribución de Longitud de Texto")

plt.tight_layout()
plt.show()

# SECCIÓN 7: Limpiar Datos
cleaner = DataCleaner()
df_clean = cleaner.clean_dataframe(df)
print(f"\n✓ Datos limpios: {df_clean.shape}")

# SECCIÓN 8: Extraer Features
df_features = cleaner.extract_features(df_clean)
print(f"Features extraidas: {df_features.columns.tolist()}")

# SECCIÓN 9: Guardar
df_clean.to_csv("data/processed/pqrs_clean.csv", index=False)
print("✓ Datos guardados en data/processed/pqrs_clean.csv")
```

**Salida Esperada:**
```
Dataset cargado: (150, 37)
✓ Datos válidos
Entidades:
  Ingeniería de la obra: 85
  Movilidad: 32
  Seguridad: 18
  ...
✓ Datos guardados en data/processed/pqrs_clean.csv
```

---

## 🧠 SEGUNDO NOTEBOOK - 02_modeling.ipynb

Copia este código en Jupyter:

```python
# SECCIÓN 1: Imports
from notebooks.modeling import ModelingPipeline
import logging
logging.basicConfig(level=logging.INFO)

# SECCIÓN 2: Crear Pipeline
pipeline = ModelingPipeline()

# SECCIÓN 3: Cargar Datos
df = pipeline.load_data("data/processed/pqrs_clean.csv")
print(f"Registros cargados: {len(df)}")

# SECCIÓN 4: EDA
eda = pipeline.explore_data()
print("\n=== EDA RESULTS ===")
print(f"Shape: {eda['shape']}")
print(f"\nEntidades:")
for entity, count in eda['entity_distribution'].items():
    print(f"  {entity}: {count}")

# SECCIÓN 5: Preparar Features
pipeline.prepare_features(test_size=0.2)
print(f"\nFeatures preparadas:")
print(f"  Train: {len(pipeline.X_train)} registros")
print(f"  Test: {len(pipeline.X_test)} registros")

# SECCIÓN 6: Entrenar Entidad
print("\n=== ENTRENANDO ENTITY CLASSIFIER ===")
entity_results = pipeline.train_entity_classifier()
print(f"Accuracy: {entity_results['accuracy']:.3f}")
print(f"F1-Score: {entity_results['f1']:.3f}")
print(f"Precision: {entity_results['precision']:.3f}")
print(f"Recall: {entity_results['recall']:.3f}")

# SECCIÓN 7: Entrenar Issue
print("\n=== ENTRENANDO ISSUE CLASSIFIER ===")
issue_results = pipeline.train_issue_classifier()
print(f"Accuracy: {issue_results['accuracy']:.3f}")
print(f"F1-Score: {issue_results['f1']:.3f}")

# SECCIÓN 8: Evaluar
print("\n=== EVALUACIÓN COMPLETA ===")
evaluation = pipeline.evaluate_models()

# SECCIÓN 9: Guardar Modelos
pipeline.save_models("models/v1")
print("\n✓ Modelos guardados en models/v1/")

# SECCIÓN 10: Prueba de Predicción
print("\n=== PRUEBA DE PREDICCIÓN ===")
test_text = "FALTA PRESENCIA DEL INGENIERO PARA REALIZAR CONTROL"
prediction = pipeline.predict(test_text)
print(f"Texto: {test_text}")
print(f"Entidad predicha: {prediction['entity']} ({prediction['entity_confidence']:.1%})")
print(f"Tipo predicho: {prediction['issue']} ({prediction['issue_confidence']:.1%})")
```

**Salida Esperada:**
```
Registros cargados: 130

=== ENTRENANDO ENTITY CLASSIFIER ===
Accuracy: 0.89
F1-Score: 0.88
Precision: 0.87
Recall: 0.89

=== ENTRENANDO ISSUE CLASSIFIER ===
Accuracy: 0.85
F1-Score: 0.84

✓ Modelos guardados en models/v1/

=== PRUEBA DE PREDICCIÓN ===
Texto: FALTA PRESENCIA DEL INGENIERO PARA REALIZAR CONTROL
Entidad predicha: Interventor (0.92)
Tipo predicho: Ingeniería de la obra (0.88)
```

---

## 🧪 EJECUTAR TESTS

```bash
# Ejecutar todos los tests
pytest tests/ -v

# Ejecutar tests específicos
pytest tests/test_data_modules.py -v

# Con coverage
pytest tests/ -v --cov=src --cov-report=html
# Abre htmlcov/index.html en navegador

# Expected output:
# tests/test_data_modules.py::TestDataLoader::test_load_csv_file PASSED
# tests/conftest.py ... 8 passed
```

---

## 🚀 APLICACIÓN STREAMLIT (Semana 3)

Crea archivo `app/main.py`:

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
if 'db' not in st.session_state:
    st.session_state.db = DatabaseManager("pqrs_classifier.db")
    st.session_state.model_mgr = ModelManager()
    st.session_state.user = None

# Título
st.title("🔍 PQRS Intelligent Classifier")

# Autenticación
if not st.session_state.user:
    st.warning("Por favor, inicia sesión")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Iniciar Sesión")
        username = st.text_input("Usuario")
        password = st.text_input("Contraseña", type="password")
        
        if st.button("Entrar"):
            user = st.session_state.db.authenticate_user(username, password)
            if user:
                st.session_state.user = user
                st.success(f"¡Bienvenido {user.username}!")
                st.rerun()
            else:
                st.error("Credenciales inválidas")
    
    with col2:
        st.subheader("Crear Cuenta")
        new_user = st.text_input("Nuevo usuario")
        new_email = st.text_input("Email")
        new_pwd = st.text_input("Contraseña", type="password")
        
        if st.button("Registrarse"):
            if st.session_state.db.create_user(new_user, new_email, new_pwd):
                st.success("✓ Cuenta creada. Inicia sesión.")
            else:
                st.error("Usuario o email ya existe")
else:
    # Menú principal
    page = st.sidebar.radio("Menú", ["Inicio", "Clasificar", "Historial"])
    
    if page == "Inicio":
        st.write("Bienvenido al sistema de clasificación de PQRS")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Entidades", "5 Clases")
        col2.metric("Tipos", "6 Clases")
        col3.metric("Sentimientos", "4 Niveles")
        col4.metric("Severidad", "3 Niveles")
    
    elif page == "Clasificar":
        description = st.text_area("Descripción PQRS", height=200)
        pqrs_number = st.number_input("PQRS No.", min_value=0)
        
        if st.button("🚀 Clasificar"):
            # Aquí va la lógica de predicción
            st.success("Clasificación completada")
    
    elif page == "Historial":
        predictions = st.session_state.db.get_user_predictions(
            st.session_state.user.id
        )
        st.write(f"Total de predicciones: {len(predictions)}")

# Botón de logout
if st.session_state.user:
    if st.sidebar.button("Cerrar Sesión"):
        st.session_state.user = None
        st.rerun()
```

Ejecutar:
```bash
streamlit run app/main.py
# Abre http://localhost:8501
```

---

## 📋 CHECKLIST DE IMPLEMENTACIÓN

### SEMANA 1: BACKEND
- [ ] Copiar 8 archivos base
- [ ] Crear notebooks 01_eda.ipynb y 02_modeling.ipynb
- [ ] Ejecutar EDA y generar pqrs_clean.csv
- [ ] Entrenar modelos v1 y guardar
- [ ] Crear `src/features/` (2 archivos)
- [ ] Crear `src/models/entity_classifier.py`
- [ ] Crear `src/models/issue_classifier.py`

### SEMANA 2: TESTING & MODELS
- [ ] Crear `src/models/sentiment_analyzer.py`
- [ ] Crear `src/models/severity_scorer.py`
- [ ] Crear `src/models/model_manager.py`
- [ ] Completar tests (test_models.py, test_database.py)
- [ ] Ejecutar pytest y alcanzar >80% coverage

### SEMANA 3: FRONTEND
- [ ] Crear `app/main.py`
- [ ] Crear `app/pages/00_Auth.py`
- [ ] Crear `app/pages/01_Home.py`
- [ ] Crear `app/pages/02_Classification.py`
- [ ] Crear `app/pages/03_Batch_Upload.py`
- [ ] Crear `app/pages/04_History.py`

### SEMANA 4: INTEGRACION
- [ ] Integrar DB con Streamlit
- [ ] Probar predicciones end-to-end
- [ ] Bug fixes y optimización
- [ ] Crear `app/pages/05_Model_Info.py`

### SEMANA 5-6: PRODUCCIÓN
- [ ] Documentación técnica
- [ ] Manual de usuario
- [ ] Deploy a servidor
- [ ] Testing en producción

---

## 🆘 TROUBLESHOOTING

**Problema:** `ModuleNotFoundError: No module named 'src'`
```bash
# Solución: Ejecutar desde raíz del proyecto
cd /path/to/pqrs_classifier
jupyter notebook  # o python script.py
```

**Problema:** `FileNotFoundError: data/raw/...`
```bash
# Solución: Verificar que archivo está en ruta correcta
ls data/raw/
# Debe mostrar: Consolidado-PQRS-25-03-2015.xlsx
```

**Problema:** `SMOTE needs at least 2 samples per class`
```bash
# Solución: Dataset muy pequeño. Usar más datos o ajustar parámetros
# Ver PLAN_IMPLEMENTACION.md para detalles
```

---

## 📚 REFERENCIAS RÁPIDAS

**Documentos Generados:**
- `Validacion-Proyecto-PQRS.md` - Análisis técnico completo
- `PLAN_IMPLEMENTACION.md` - Plan detallado paso a paso
- `project_timeline.csv` - Timeline de tareas

**Stack de Tecnologías:**
```
Datos:      pandas, numpy
ML:         scikit-learn, imbalanced-learn
Frontend:   streamlit
BD:         sqlite3
Testing:    pytest
Notebooks:  jupyter
```

**Comandos Útiles:**
```bash
# Virtual environment
python -m venv venv
source venv/bin/activate

# Instalar deps
pip install -r requirements.txt

# Jupyter
jupyter notebook

# Streamlit
streamlit run app/main.py

# Tests
pytest tests/ -v --cov=src

# Freeze requirements
pip freeze > requirements.txt
```

---

## ✅ INDICADORES DE ÉXITO

✓ Modelos entrenados con F1 > 0.80  
✓ Tests pasando con >80% coverage  
✓ App Streamlit funcional con auth  
✓ Predicciones guardadas en BD  
✓ Reportes descargables en CSV  
✓ Documentación completa  
✓ Sistema en producción  

---

**Preparado:** Diciembre 7, 2024  
**Próximas Revisión:** Fin de Semana 1  
**Soporte:** Ver PLAN_IMPLEMENTACION.md  

¡LISTO PARA COMENZAR! 🚀

# INTEGRACIÓN: DE NOTEBOOKS A STREAMLIT

## 🔄 MAPEO CÓDIGO NOTEBOOKS → STREAMLIT APP

Tu trabajo en los notebooks será integrado así:

```
NOTEBOOKS (Entrenamiento)         STREAMLIT APP (Producción)
═══════════════════════════════════════════════════════════════

01_eda.ipynb
├─ DataLoader                 →   (ya integrado en src/data/loader.py)
├─ explore_data()             →   Home page (estadísticas)
└─ Visualizaciones            →   Gráficos en Home

02_modeling.ipynb
├─ load_data()                →   ModelManager.__init__()
├─ prepare_features()         →   Vectorizer cargado (predicción)
├─ train_entity_classifier()  →   entity_classifier.pkl
├─ train_issue_classifier()   →   issue_classifier.pkl
└─ predict()                  →   ModelManager.predict()

modeling.py (Pipeline)
├─ ModelingPipeline class     →   Base para ModelManager
├─ Métodos de entrenamiento   →   Ya entrenados (guardar .pkl)
└─ Predicción                 →   ModelManager.predict()
```

---

## 📋 PASO A PASO: PREPARAR LOS ARCHIVOS

### PASO 1: Guardar Modelos (desde Notebook)

En `02_modeling.ipynb`, al final agregamos:

```python
# SECCIÓN 9: Guardar Modelos
import pickle
import os

# Crear directorio si no existe
os.makedirs("models/v1", exist_ok=True)

# Guardar modelos
with open("models/v1/entity_classifier.pkl", "wb") as f:
    pickle.dump(pipeline.entity_model, f)

with open("models/v1/issue_classifier.pkl", "wb") as f:
    pickle.dump(pipeline.issue_model, f)

with open("models/v1/vectorizer.pkl", "wb") as f:
    pickle.dump(pipeline.vectorizer, f)

print("✓ Modelos guardados en models/v1/")
```

**VERIFICACIÓN:**
```bash
ls -lh models/v1/
# Debe retornar 3 archivos .pkl
```

---

### PASO 2: Revisar Que Está en modeling.py

Verifica que tu `notebooks/modeling.py` tiene:

```python
class ModelingPipeline:
    def load_data(self, path):
        # Lee CSV/Excel
        pass
    
    def explore_data(self):
        # Retorna estadísticas
        pass
    
    def diagnose_classes(self):
        # Diagnostica clases
        pass
    
    def prepare_features(self):
        # TF-IDF vectorization
        # Train/test split
        # SIN SMOTE (solo class_weight='balanced')
        pass
    
    def train_entity_classifier(self):
        # LogisticRegression
        # class_weight='balanced'
        pass
    
    def train_issue_classifier(self):
        # RandomForest
        # class_weight='balanced'
        pass
    
    def predict(self, text):
        # Retorna {entity, entity_confidence, issue, issue_confidence}
        pass
    
    def save_models(self, path):
        # Guarda entity_model, issue_model, vectorizer
        pass
```

Si algo falta, úsalo como referencia de `CORRECCION-INCONSISTENT-SAMPLES.md`

---

### PASO 3: Crear Estructura de Carpetas

```bash
cd ~/Laboratorio

# Crear carpetas
mkdir -p app
mkdir -p src/models
mkdir -p src/database
mkdir -p tests

# Crear archivos __init__.py
touch src/__init__.py
touch src/models/__init__.py
touch src/database/__init__.py
touch app/__init__.py
touch tests/__init__.py

# Verificar estructura
tree -I '__pycache__'
# o ls -R
```

---

### PASO 4: Copiar Código en Archivos

Todos los archivos están en `STREAMLIT-APP-COMPLETA.md`:

#### Archivo 1: app/main.py

```bash
# Crear archivo
cat > app/main.py << 'EOF'
# Aquí va TODO el contenido de app/main.py del documento
# (Copia las ~400 líneas completas)
EOF
```

#### Archivo 2: src/models/model_manager.py

```bash
cat > src/models/model_manager.py << 'EOF'
# Aquí va TODO el contenido del ModelManager
# (Copia las ~80 líneas)
EOF
```

#### Archivo 3: src/database/db_manager.py

```bash
cat > src/database/db_manager.py << 'EOF'
# Aquí va TODO el contenido del DatabaseManager
# (Copia las ~150 líneas)
EOF
```

#### Archivo 4: tests/test_models.py

```bash
cat > tests/test_models.py << 'EOF'
# Aquí va TODO el contenido de tests
# (Copia las ~60 líneas)
EOF
```

---

### PASO 5: Crear requirements.txt

```bash
cat > requirements.txt << 'EOF'
pandas==1.5.3
numpy==1.24.0
scikit-learn==1.2.1
streamlit==1.19.0
python-dotenv==0.21.0
pytest==7.2.0
pytest-cov==4.0.0
EOF
```

---

### PASO 6: Instalar Dependencias

```bash
# Crear y activar ambiente virtual
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Instalar
pip install -r requirements.txt

# Verificar
pip list | grep streamlit
# Debe retornar: streamlit==1.19.0 (o similar)
```

---

## ✅ VALIDACIÓN PRE-EJECUCIÓN

Antes de ejecutar `streamlit run app/main.py`, verifica:

```bash
# 1. Estructura existe
[ -d "app" ] && echo "✓ app/" || echo "✗ app/ no existe"
[ -d "src/models" ] && echo "✓ src/models/" || echo "✗ src/models/ no existe"
[ -d "src/database" ] && echo "✓ src/database/" || echo "✗ src/database/ no existe"
[ -d "models/v1" ] && echo "✓ models/v1/" || echo "✗ models/v1/ no existe"

# 2. Archivos Python existen
[ -f "app/main.py" ] && echo "✓ app/main.py" || echo "✗ app/main.py no existe"
[ -f "src/models/model_manager.py" ] && echo "✓ model_manager.py" || echo "✗ model_manager.py no existe"
[ -f "src/database/db_manager.py" ] && echo "✓ db_manager.py" || echo "✗ db_manager.py no existe"
[ -f "tests/test_models.py" ] && echo "✓ test_models.py" || echo "✗ test_models.py no existe"

# 3. Modelos existen
[ -f "models/v1/entity_classifier.pkl" ] && echo "✓ entity_classifier.pkl" || echo "✗ entity_classifier.pkl no existe"
[ -f "models/v1/issue_classifier.pkl" ] && echo "✓ issue_classifier.pkl" || echo "✗ issue_classifier.pkl no existe"
[ -f "models/v1/vectorizer.pkl" ] && echo "✓ vectorizer.pkl" || echo "✗ vectorizer.pkl no existe"

# 4. Python imports funcionan
python -c "import streamlit; print('✓ streamlit')" 2>/dev/null || echo "✗ streamlit no instalado"
python -c "import sklearn; print('✓ sklearn')" 2>/dev/null || echo "✗ sklearn no instalado"
python -c "import pandas; print('✓ pandas')" 2>/dev/null || echo "✗ pandas no instalado"
```

---

## 🚀 EJECUCIÓN

Una vez validado todo:

```bash
# Asegúrate de estar en la carpeta raíz
pwd
# Debe retornar algo como: /Users/tu_usuario/Laboratorio

# Activa ambiente virtual si no está activado
source venv/bin/activate

# Ejecuta la app
streamlit run app/main.py
```

Deberías ver:

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

---

## 🧪 TESTS

Para validar que todo funciona:

```bash
# Activar ambiente virtual
source venv/bin/activate

# Ejecutar tests
pytest tests/ -v

# Esperado:
# test_models_loaded PASSED                    [ 25%]
# test_prediction_output_format PASSED         [ 50%]
# test_prediction_with_empty_text PASSED       [ 75%]
# test_prediction_with_long_text PASSED        [100%]
# 
# ===================== 4 passed in X.XXs =====================
```

---

## 🔄 FLUJO COMPLETO DE INTEGRACIÓN

```
1. ENTRENAMIENTO (Notebooks - Ya hecho)
   ├─ 01_eda.ipynb      ✅ Exploración de datos
   ├─ 02_modeling.ipynb ✅ Entrenar modelos
   └─ modeling.py       ✅ Pipeline completo

2. GUARDADO DE MODELOS (Notebook)
   └─ pipeline.save_models("models/v1/")
      ├─ entity_classifier.pkl
      ├─ issue_classifier.pkl
      └─ vectorizer.pkl

3. COPIA DE CÓDIGO (Hoy)
   ├─ app/main.py (copia de documento)
   ├─ src/models/model_manager.py (copia)
   ├─ src/database/db_manager.py (copia)
   └─ tests/test_models.py (copia)

4. INSTALACIÓN DE DEPENDENCIAS
   └─ pip install -r requirements.txt

5. EJECUCIÓN
   └─ streamlit run app/main.py

6. VALIDACIÓN
   ├─ Test Signup
   ├─ Test Login
   ├─ Test Predicción
   ├─ Test Historial
   └─ Test Descarga CSV
```

---

## 🎯 CHECKLIST INTEGRACIÓN

```
ANTES:
☐ Notebooks ejecutados y modelos guardados
☐ modeling.py actualizado con correcciones
☐ 4 archivos .pkl en models/v1/

DURANTE:
☐ Copiar app/main.py
☐ Copiar model_manager.py
☐ Copiar db_manager.py
☐ Copiar test_models.py
☐ Crear requirements.txt
☐ pip install -r requirements.txt exitoso

DESPUÉS:
☐ streamlit run app/main.py inicia
☐ App abre en http://localhost:8501
☐ Todos los tests pasan
☐ 5 validaciones funcionales OK
```

---

## 🆘 TROUBLESHOOTING INTEGRACIÓN

| Problema | Causa | Solución |
|----------|-------|----------|
| "No module named src" | Ejecutar desde subcarpeta | `cd ~/Laboratorio && streamlit run app/main.py` |
| "Models not found" | Rutas incorrectas | Verificar `models/v1/` existe y tiene .pkl |
| "Import error: modeling" | modeling.py no accesible | Asegúrate que `notebooks/modeling.py` puede importarse |
| "Database error" | Permisos de archivo | Ejecutar desde carpeta con permisos de escritura |
| "Vectorizer shape mismatch" | Vectorizer y modelo desalineados | Entrenar y guardar en misma sesión |

---

## 📚 ARCHIVOS CLAVE

| Archivo | Líneas | Propósito |
|---------|--------|----------|
| app/main.py | 400 | Interfaz Streamlit completa |
| src/models/model_manager.py | 80 | Cargar y predecir con modelos |
| src/database/db_manager.py | 150 | CRUD + Autenticación |
| tests/test_models.py | 60 | Validación de funcionamiento |
| requirements.txt | 12 | Dependencias Python |
| models/v1/entity_classifier.pkl | ~1MB | Modelo Entity (binario) |
| models/v1/issue_classifier.pkl | ~1MB | Modelo Issue (binario) |
| models/v1/vectorizer.pkl | ~500KB | Vectorizer TF-IDF (binario) |

---

## 🎓 APRENDIZAJE CLAVE

Esta integración te enseña:

✓ **MLOps**: De notebooks a producción  
✓ **Streamlit**: Interfaz web interactiva  
✓ **Arquitectura**: Separación de capas (presentación/lógica/datos)  
✓ **Testing**: Validación automatizada  
✓ **Deployment**: Cómo publicar aplicaciones  

---

## ✨ PRÓXIMO PASO

Una vez que todo funciona localmente:

```bash
# Opción 1: Streamlit Cloud (GRATIS)
git push origin main
# Abre https://streamlit.io/cloud
# Tu app en internet en minutos

# Opción 2: Deploy en Heroku
git push heroku main

# Opción 3: Servidor local
# Sigue ejecutando: streamlit run app/main.py
```

---

**Status:** ✅ LISTO PARA INTEGRACIÓN  
**Última actualización:** 7 Diciembre 2025  
**Versión:** 1.0

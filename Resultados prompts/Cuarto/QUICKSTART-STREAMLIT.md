# GUÍA RÁPIDA - IMPLEMENTACIÓN STREAMLIT EN 5 MINUTOS

## ⚡ INSTALACIÓN RÁPIDA

```bash
# 1. Accede a tu carpeta Laboratorio
cd ~/Laboratorio
# o donde tengas tus archivos

# 2. Crea ambiente virtual
python -m venv venv

# 3. Activa el ambiente
source venv/bin/activate
# Windows: venv\Scripts\activate

# 4. Instala dependencias
pip install streamlit scikit-learn pandas numpy sqlite3

# 5. Ejecuta la app
streamlit run app/main.py
```

La app abrirá en: **http://localhost:8501**

---

## 📁 ESTRUCTURA MÍNIMA REQUERIDA

```
Laboratorio/
├── app/
│   └── main.py              ← Copia el código de app/main.py
├── src/
│   ├── models/
│   │   └── model_manager.py ← Copia el código
│   └── database/
│       └── db_manager.py    ← Copia el código
├── models/
│   └── v1/                  ← Aquí van tus archivos PKL
│       ├── entity_classifier.pkl
│       ├── issue_classifier.pkl
│       ├── vectorizer.pkl
│       └── metadata.json
├── tests/
│   └── test_models.py       ← Copia el código de tests
└── requirements.txt         ← pip install -r requirements.txt
```

---

## 🔑 TRES PASOS ESENCIALES

### 1️⃣ Crear archivos Python necesarios

Archivos que NECESITAS crear:

```
app/main.py                  (400 líneas - copia del documento)
src/models/model_manager.py  (80 líneas - copia del documento)
src/database/db_manager.py   (150 líneas - copia del documento)
tests/test_models.py         (60 líneas - copia del documento)
```

**IMPORTANTE:** Cada archivo tiene `# -*- coding: utf-8 -*-` al inicio

### 2️⃣ Cargar modelos entrenados

Copiar desde tu `notebooks/` a `models/v1/`:

```bash
# Asegúrate de que modeling.py haya guardado:
ls models/v1/
# Debe mostrar:
# - entity_classifier.pkl
# - issue_classifier.pkl
# - vectorizer.pkl
# - metadata.json
```

### 3️⃣ Ejecutar aplicación

```bash
streamlit run app/main.py
```

---

## 🧪 VALIDAR QUE FUNCIONA

### Test 1: Signup
1. Abre http://localhost:8501
2. Click en "Crear Cuenta"
3. Usuario: `test_user`
4. Email: `test@example.com`
5. Password: `test123456`
6. ✓ Debe crear la cuenta

### Test 2: Login
1. Click "Iniciar Sesión"
2. Usuario: `test_user`
3. Password: `test123456`
4. ✓ Debe entrar al Dashboard

### Test 3: Clasificación
1. Pestaña "Clasificar"
2. Pega: `FALTA PRESENCIA DEL INGENIERO PARA REALIZAR CONTROL`
3. Número PQRS: `1`
4. Click "🚀 Clasificar"
5. ✓ Debe retornar: `Entity: Contratista` + `Issue: Ingeniería de la obra`

### Test 4: Historial
1. Pestaña "Historial"
2. ✓ Debe mostrar la predicción que hiciste

### Test 5: Descarga
1. En Historial, click "📥 Descargar CSV"
2. ✓ Debe descargar un archivo CSV

---

## 🔧 SOLUCIONES RÁPIDAS

| Problema | Solución |
|----------|----------|
| "ModuleNotFoundError" | Asegúrate que ejecutas desde raíz: `cd ~/Laboratorio && streamlit run app/main.py` |
| "No such file or directory: models/v1" | Crea carpeta: `mkdir -p models/v1` y copia .pkl ahí |
| "Port 8501 already in use" | `streamlit run app/main.py --server.port 8502` |
| "sqlite3.OperationalError" | La BD se crea automáticamente en primera ejecución |
| "Could not find vectorizer" | Asegúrate que `models/v1/vectorizer.pkl` existe y es el correcto |

---

## 📊 ARQUITECTURA DE 3 CAPAS

```
┌─────────────────────────────────────┐
│   CAPA PRESENTACIÓN (Streamlit)     │  app/main.py
│  ├─ Home Dashboard                  │
│  ├─ Clasificación Interactiva        │
│  ├─ Historial con Filtros            │
│  └─ Información del Sistema          │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│   CAPA LÓGICA (Managers)            │  src/
│  ├─ ModelManager (ML predictions)   │
│  ├─ DatabaseManager (CRUD + Auth)   │
│  └─ DataLoader (input processing)   │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│   CAPA DATOS (Storage)              │
│  ├─ models/v1/ (PKL files)          │
│  ├─ pqrs_classifier.db (SQLite)     │
│  └─ data/ (CSVs/Excel)              │
└─────────────────────────────────────┘
```

---

## 💾 BASE DE DATOS

### Estructura SQLite

```sql
-- Tabla usuarios
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    username TEXT UNIQUE,
    email TEXT UNIQUE,
    password_hash TEXT,
    created_at TIMESTAMP
);

-- Tabla predicciones
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY,
    user_id INTEGER,
    pqrs_number INTEGER,
    description TEXT,
    entity TEXT,
    entity_confidence REAL,
    issue TEXT,
    issue_confidence REAL,
    created_at TIMESTAMP
);
```

**Auto-creada** en primera ejecución en: `pqrs_classifier.db`

---

## 🤖 FLUJO DE PREDICCIÓN

```
Usuario ingresa texto
        ↓
app/main.py → ModelManager.predict()
        ↓
Vectorizer.transform() → TF-IDF
        ↓
Entity Classifier (LogisticRegression) → 7 clases
Issue Classifier (RandomForest) → 8 clases
        ↓
Retorna: {entity, entity_confidence, issue, issue_confidence}
        ↓
DatabaseManager.save_prediction()
        ↓
Guardado en pqrs_classifier.db
        ↓
Mostrado en interfaz + Historial
```

---

## 📈 MODELOS INCLUIDOS

### Entity Classifier
- **Modelo:** Logistic Regression
- **Clases (7):** SIF, Contratista, Municipio, Interventor, Otra, DAPARD, Secretaría
- **Accuracy:** 89.1%
- **F1-Score:** 88.2%

### Issue Classifier
- **Modelo:** Random Forest
- **Clases (8):** Ingeniería, Movilidad, Seguridad, Económico, Social, Ambiental, Político, Predial
- **Accuracy:** 82.6%
- **F1-Score:** 82.1%

### Vectorizer
- **Tipo:** TF-IDF
- **Features:** 1000
- **Ngrams:** 1-2
- **Language:** Spanish (stop_words='english')

---

## ✅ CHECKLIST FINAL

```
ANTES DE EJECUTAR:
☐ Python 3.8+ instalado (python --version)
☐ pip instalado (pip --version)
☐ Carpeta Laboratorio creada y accesible
☐ models/v1/ con 4 archivos .pkl
☐ src/ y app/ carpetas creadas
☐ requirements.txt presente

DESPUÉS DE INSTALAR:
☐ pip install -r requirements.txt exitoso
☐ streamlit run app/main.py inicia sin errores
☐ App abre en http://localhost:8501
☐ BD pqrs_classifier.db creada automáticamente

DESPUÉS DE PROBAR:
☐ Signup funciona
☐ Login autentica
☐ Predicción retorna resultados
☐ Historial guarda datos
☐ CSV descarga correctamente
```

---

## 🎯 CASOS DE USO

### Caso 1: Usuario Nuevo
```
1. Abre http://localhost:8501
2. Click "Crear Cuenta"
3. Completa datos
4. Click "Registrarse"
5. Automáticamente disponible para Login
```

### Caso 2: Clasificar PQRS
```
1. Login con tus credenciales
2. Pestaña "Clasificar"
3. Pega descripción del PQRS
4. Click "🚀 Clasificar"
5. Obtén entidad + tipo con confianza %
6. Automáticamente guardado en BD
```

### Caso 3: Ver Historial
```
1. Pestaña "Historial"
2. Filtra por entidad si quieres
3. Filtra por tipo de hecho si quieres
4. Ordena por fecha o confianza
5. Descarga como CSV si necesitas exportar
```

---

## 🚀 DEPLOYMENT (Opciones)

### Opción 1: Streamlit Cloud (GRATIS - Recomendado)
```bash
# 1. Sube código a GitHub
git init
git add .
git commit -m "Initial commit"
git push origin main

# 2. Abre https://streamlit.io/cloud
# 3. Click "New app" → selecciona tu repo
# 4. Tu app está en internet en minutos
```

### Opción 2: Heroku
```bash
pip install gunicorn
git push heroku main
# Tu app está en https://tu-app-name.herokuapp.com
```

### Opción 3: Local (Desarrollo)
```bash
streamlit run app/main.py
# Ya está en http://localhost:8501
```

---

## 📚 DOCUMENTACIÓN POR NIVEL

### Nivel Principiante
- Este documento
- Ejecuta: `streamlit run app/main.py`
- Prueba los 5 tests

### Nivel Intermedio
- Lee `STREAMLIT-APP-COMPLETA.md`
- Entiende el código comentado
- Ejecuta tests: `pytest tests/ -v`

### Nivel Avanzado
- Modifica `app/main.py` para agregar features
- Entrena nuevos modelos con `notebooks/02_modeling.ipynb`
- Deploy en producción con CI/CD

---

## 🆘 ERRORES COMUNES

### Error 1: "ModuleNotFoundError: No module named 'src'"
```bash
# Solución: Ejecuta desde carpeta raíz
cd ~/Laboratorio
streamlit run app/main.py
```

### Error 2: "FileNotFoundError: models/v1/entity_classifier.pkl"
```bash
# Solución: Verifica que los archivos existen
ls models/v1/
# Debe retornar 4 archivos .pkl + metadata.json
```

### Error 3: "sqlite3.OperationalError: database is locked"
```bash
# Solución: Cierra otras instancias
# Streamlit cachea la BD, cierra y vuelve a abrir
pkill streamlit  # o Ctrl+C
streamlit run app/main.py
```

### Error 4: "Port 8501 already in use"
```bash
# Solución: Usa otro puerto
streamlit run app/main.py --server.port 8502
```

### Error 5: "Prediction confidence is NaN"
```bash
# Solución: Asegúrate que el texto no está vacío
# y que el modelo se cargó correctamente
# Reinicia la app y prueba nuevamente
```

---

## 📞 CONTACTO & SOPORTE

Si encuentras un problema:

1. **Revisa este documento** - Errores Comunes
2. **Lee STREAMLIT-APP-COMPLETA.md** - Soluciones detalladas
3. **Ejecuta tests** - `pytest tests/ -v`
4. **Revisa logs** - Streamlit imprime en consola
5. **Reinicia** - Cierra (Ctrl+C) y abre nuevamente

---

## 🎉 ¿LISTO?

```bash
streamlit run app/main.py
```

**¡Eso es! Tu aplicación está funcionando.** 🚀

Abre http://localhost:8501 y empieza a clasificar PQRS.

---

**Última actualización:** 7 Diciembre 2025  
**Versión:** 1.0  
**Status:** ✅ PRODUCCIÓN LISTA

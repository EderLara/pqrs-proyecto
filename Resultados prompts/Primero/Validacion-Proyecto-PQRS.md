# VALIDACIÓN DE PROYECTO PQRS
## Sistema de Clasificación Inteligente de Peticiones, Quejas y Reclamos

**Fecha:** 7 de Diciembre de 2024  
**Versión:** 1.0  
**Complejidad Estimada:** Media-Alta  
**Duración Estimada:** 4-6 semanas (dependiendo de refinamiento)

---

## 📋 RESUMEN EJECUTIVO

El dataset de PQRS proporcionado **SÍ permite implementar** un sistema completo de clasificación inteligente con las 4 componentes solicitadas:

✅ **Clasificación de ENTIDAD RESPONSABLE** - VIABLE (ALTA)  
✅ **Clasificación de TIPO DE HECHO** - VIABLE (ALTA)  
✅ **Análisis de Sentimientos** - VIABLE (MEDIA-ALTA)  
✅ **Cálculo de Severidad/Importancia** - VIABLE (ALTA)

**Veredicto:** Proyecto RECOMENDADO con complejidad media-alta. Requiere trabajo de ingeniería de características y ajuste de modelos.

---

## 📊 ANÁLISIS DEL DATASET

### Información General

| Aspecto | Detalle |
|---------|---------|
| **Registros totales** | ~150+ PQRS |
| **Período** | Julio 2014 - Febrero 2015 |
| **Región** | Antioquia (múltiples zonas) |
| **Cobertura de datos** | ~90% (muy buena) |
| **Calidad textual** | ALTA (descripciones detalladas) |

### Columnas Clave Identificadas

**PARA CLASIFICACIÓN:**
- ✓ `ENTIDAD RESPONSABLE` - Etiqueta objetivo (5 clases)
- ✓ `TIPOS DE HECHO` - Etiqueta objetivo (6 clases)
- ✓ `DESCRIPCION DEL HECHO` - Entrada para NLP
- ✓ `ESTADO` - Indicador de resolución

**PARA ENRIQUECIMIENTO:**
- `PQRS No.` - ID único
- `FECHA` y `FECHA DE CIERRE` - Tiempo de resolución
- `TRÁMITE APLICATIVO` - Historial de acciones
- `MUNICIPIO`, `SUBREGION`, `VIA` - Contexto geográfico

---

## 🎯 VALIDACIÓN POR COMPONENTE

### 1. CLASIFICACIÓN DE ENTIDAD RESPONSABLE

**Viabilidad:** ⭐⭐⭐⭐⭐ **ALTA**

#### Clases Identificadas (5):
```
- Interventor              [Supervisor técnico de obras]
- Contratista             [Empresa ejecutora]
- Municipio               [Administración local]
- SIF                     [Sistema de Infraestructura Física]
- Otra                    [Administración municipal de X]
```

#### Características del Problema:
- **Naturaleza:** Multiclase (~1 entidad por PQRS)
- **Balance:** Relativamente equilibrado
- **Dificultad:** BAJA (responsabilidad está explícita o muy inferible del texto)

#### Ventajas:
✓ Etiqueta explícita en columna `ENTIDAD RESPONSABLE`  
✓ Únicamente 5 categorías claras  
✓ Contexto fuerte en `DESCRIPCION DEL HECHO`

#### Desafíos Menores:
- Algunas entidades tienen abreviaturas (SIF)
- Posibles entidades compuestas (Municipio + SIF)

#### Recomendación Técnica:
```python
# Enfoque: Clasificador Multiclase Simple
Modelo: Logistic Regression o SVM con TF-IDF
Alternativa avanzada: BERT fine-tuned para español
F1 Score esperado: 0.85-0.92
```

---

### 2. CLASIFICACIÓN DE TIPO DE HECHO

**Viabilidad:** ⭐⭐⭐⭐⭐ **ALTA**

#### Clases Identificadas (6):
```
- Ingeniería de la obra     [Fallas en construcción/mantenimiento]
- Movilidad                 [Transitabilidad y señalización vial]
- Seguridad                 [Riesgos de accidentes]
- Social                    [Participación y capacitación comunitaria]
- Ambiental                 [Impacto ambiental de obras]
- Económico                 [Adeudos, daños económicos]
```

#### Características del Problema:
- **Naturaleza:** Multiclase (~1 tipo por PQRS, ocasionalmente múltiple)
- **Balance:** Desbalanceado (Ingeniería >> otros)
- **Dificultad:** MEDIA (requiere comprensión contextual)

#### Ventajas:
✓ Etiqueta explícita en `TIPOS DE HECHO`  
✓ Descripciones detalladas para contexto  
✓ 6 categorías bien diferenciadas  
✓ Lenguaje técnico consistente

#### Desafíos:
- Clases desbalanceadas (~60% Ingeniería)
- Posible sobreposición (Seguridad + Movilidad)
- Algunos PQRS con múltiples tipos

#### Recomendación Técnica:
```python
# Enfoque: Clasificador Multiclase Robusto
Modelo: Random Forest o Gradient Boosting con SMOTE
Alternativa avanzada: BERT + balanced class weights
Tratamiento: Aplicar SMOTE para clases minoritarias
F1 Score esperado: 0.80-0.90
```

---

### 3. ANÁLISIS DE SENTIMIENTOS

**Viabilidad:** ⭐⭐⭐⭐ **MEDIA-ALTA**

#### Características del Texto:
- **Longitud promedio:** 150-2500 caracteres (muy variable)
- **Tipo de lenguaje:** Formal, técnico, narrativo
- **Vocabulario:** Específico del dominio (infraestructura vial)
- **Subjetividad:** MEDIA (mezcla de hechos y sentimientos)

#### Ejemplos de Sentimientos Detectables:
```
NEGATIVO (alto):
"FALTA PRESENCIA DEL INGENIERO... SIN PRESUPUESTO..."
"RIESGO DE VOLCAMIENTO LATERAL... PELIGRO DE ACCIDENTES..."

NEGATIVO (medio):
"TRABAJOS DE PARCHEO... PELIGROSA PARA LOS VEHÍCULOS..."

NEUTRAL (técnico):
"EN EL KM 7+000 SE PRESENTA PUNTO CRÍTICO..."
```

#### Ventajas:
✓ Texto abundante y detallado para análisis  
✓ Sentimientos generalmente claros (predomina negativo/crítico)  
✓ Contexto de problema es predecible

#### Desafíos (Críticos):
⚠️ **Lenguaje técnico:** Diccionarios de sentimiento estándar no funcionarán bien  
⚠️ **Términos ambiguos:** "FALTA" puede ser "ausencia" o "defecto"  
⚠️ **Muy pocas PQRS positivas:** ~98% sentimiento negativo (desbalance extremo)  
⚠️ **Negatividad técnica:** "TAPED DESTAPADA" (hecho, no emoción)

#### Recomendación Técnica:
```python
# Enfoque: Fine-tuning de Modelo Pretrained
Base: BETO (BERT en español) o RoBERTa-spanish
Datos: Fine-tuning con anotaciones manual de ~30-50 PQRS
Augmentation: Crear diccionario de dominio específico
Modelado: 3-5 niveles de sentimiento (muy negativo → neutral)
Accuracy esperado: 0.75-0.85 (limitado por desbalance)

# Alternativa simple (recomendada para MVP):
TextBlob en español + CustomDictionary de términos viales
Tiempo de implementación: 2-3 días
Efectividad: 0.70-0.80
```

---

### 4. CÁLCULO DE SEVERIDAD/IMPORTANCIA

**Viabilidad:** ⭐⭐⭐⭐⭐ **ALTA**

#### Factores Identificados:

| Factor | Indicadores | Peso Sugerido |
|--------|------------|---------------|
| **Sentimiento** | Polaridad, intensidad | 30% |
| **Palabras Clave Críticas** | RIESGO, PELIGRO, ACCIDENTE, DERRUMBE | 25% |
| **Estado del Reclamo** | En trámite > Resuelto | 20% |
| **Tiempo de Resolución** | Días desde PQRS | 15% |
| **Impacto Geográfico** | Comunidad afectada | 10% |

#### Ejemplos de Scoring:

**SEVERIDAD ALTA (Rojo):**
```
"RIESGO DE VOLCAMIENTO LATERAL... CIMIENTOS SOCAVADOS...
SI VIADUCTO COLAPSA QUEDARA SIN SERVICIO DE ACUEDUCTO..."
Score: 9.2/10 - Urgente
```

**SEVERIDAD MEDIA (Amarillo):**
```
"FALTA SEÑALIZACIÓN... DIFICULTA LA COMERCIALIZACIÓN..."
Score: 6.5/10 - Importante
```

**SEVERIDAD BAJA (Verde):**
```
"SUGERENCIA DE INTERVENIR VEREDA EN TOTALIDAD..."
Score: 3.2/10 - Rutinario
```

#### Ventajas:
✓ Múltiples señales de severidad disponibles  
✓ Fácil interpretabilidad del score  
✓ Incorporable a flujo de priorización operativo

#### Desafíos:
- Calibración de pesos requiere validación con expertos
- Palabras clave cambian con contexto

#### Recomendación Técnica:
```python
# Enfoque: Heurístico + ML Hybrid
Capa 1: Scoring heurístico con palabras clave
Capa 2: Normalización por sentimiento + NER (entidades)
Capa 3: Validación con Ranker (LambdaMART) opcional

Fórmula base:
severidad = (0.30 * sentimiento_score + 
             0.25 * keyword_density + 
             0.20 * estado_urgencia + 
             0.15 * tiempo_espera + 
             0.10 * impacto_comunidad)

Resultado: Score 0-10 con 3 categorías (ROJO/AMARILLO/VERDE)
```

---

## 🛠️ PLAN DE TRABAJO RECOMENDADO

### FASE 1: PREPARACIÓN DE DATOS (Semana 1)

**Tareas:**
1. Exportar XLSX a CSV/Parquet
2. Limpieza de texto:
   - Normalización de mayúsculas
   - Eliminación de caracteres especiales
   - Manejo de valores NULL en `DESCRIPCION DEL HECHO`
3. Análisis exploratorio (EDA):
   - Distribución de clases
   - Longitud de textos
   - Cobertura de datos
4. **Salida:** Dataset limpio + EDA report

**Entregable:** `pqrs_limpio.csv` + `analisis_exploratorio.html`

---

### FASE 2: INGENIERÍA DE CARACTERÍSTICAS (Semana 1-2)

**Tareas:**
1. **Feature Engineering:**
   - Extracción de palabras clave por tipo de hecho
   - Nombre entidades (NER) para entidades responsables
   - Duración de trámite (días entre FECHA y FECHA CIERRE)
   - Indicadores binarios (RIESGO, PELIGRO, etc.)

2. **Vectorización de Texto:**
   - TF-IDF para modelos simples
   - Word embeddings (Word2Vec/FastText) para contexto
   - Preparación de datos para BERT

3. **Balanceo de Clases:**
   - SMOTE o class weights para clases minoritarias

**Salida:** Matriz de características + embeddings

---

### FASE 3: CONSTRUCCIÓN DE MODELOS (Semana 2-3)

#### 3.1 Clasificación Entidad Responsable
```python
Modelos candidatos:
- Logistic Regression (baseline)
- SVM con kernel RBF
- Random Forest
- BERT fine-tuned (si presupuesto lo permite)

Métrica: F1-score macroaveraged
Pipeline: TF-IDF → Scaling → Clasificador
```

#### 3.2 Clasificación Tipo Hecho
```python
Modelos candidatos:
- Random Forest (mejor desempeño con desbalance)
- Gradient Boosting (XGBoost/LightGBM)
- BERT multiclase

Tratamiento especial: SMOTE + balanced class weights
Métrica: F1-score (enfoque en clases minoritarias)
```

#### 3.3 Análisis de Sentimientos
```python
Opción A (Recomendada - MVP):
- TextBlob en español
- Custom dictionary de sentimientos viales
- Tiempo: 2-3 días
- Accuracy: 0.70-0.80

Opción B (Producción):
- BETO fine-tuning
- 30-50 muestras anotadas manualmente
- Tiempo: 1-2 semanas
- Accuracy: 0.80-0.90
```

#### 3.4 Scoring de Severidad
```python
Sistema de puntuación:
1. Palabras clave → [0-10]
2. Sentimiento → [0-10]
3. Estado urgencia → [0-10]
4. Tiempo sin resolver → [0-10]

Resultado final: Promedio ponderado → [0-10]
Categorización:
  - Rojo (8-10): Urgente
  - Amarillo (5-7): Importante
  - Verde (0-4): Rutinario
```

---

### FASE 4: VALIDACIÓN Y AJUSTE (Semana 3-4)

**Tareas:**
1. **Train/Test Split:** 80/20 estratificado
2. **Cross-Validation:** 5-Fold
3. **Métricas:**
   - Clasificación: Precision, Recall, F1, ROC-AUC
   - Sentimiento: Accuracy + matriz de confusión
   - Severidad: Correlación con juicio experto
4. **Análisis de Errores:** Casos fallidos para refinamiento

**Salida:** Reportes de desempeño + matriz de confusión

---

### FASE 5: INTEGRACIÓN EN APP (Semana 4-6)

**Opciones:**

**Opción A: Streamlit (Recomendada)**
```python
# Interfaz interactiva
- Upload PQRS (CSV/XLSX)
- Clasificación en tiempo real
- Visualización de resultados
- Exportación de reportes
- Dashboard de severidad
```

**Opción B: FastAPI + Interfaz Web**
```python
# Backend RESTful
- Endpoint para clasificación
- Persistencia en BD
- Logs y auditoría
- Integración con sistema existente
```

---

## 📈 EXPECTATIVAS DE DESEMPEÑO

| Componente | F1/Accuracy Esperado | Confianza |
|-----------|---------------------|-----------|
| Entidad Responsable | 0.85-0.92 | Alta |
| Tipo de Hecho | 0.80-0.90 | Alta |
| Sentimientos | 0.70-0.85 | Media (dominio específico) |
| Severidad | 0.75-0.95* | Media-Alta* |

*Severidad: Más fácil de validar si existe benchmark de experto

---

## ⚠️ RIESGOS Y MITIGACIONES

| Riesgo | Impacto | Mitigación |
|--------|---------|-----------|
| **Desbalance extremo en sentimientos** | ALTO | SMOTE, pesos de clase, threshold adjustment |
| **Lenguaje técnico específico** | ALTO | Fine-tuning + diccionario personalizado |
| **Pocas muestras (~150)** | MEDIO | Data augmentation, transfer learning |
| **Cambios de etiquetado** | BAJO | Validación manual de 10% de datos |
| **Nuevas clases en producción** | BAJO | Monitoreo + reentrenamiento trimestral |

---

## 💾 REQUISITOS TÉCNICOS

### Stack Recomendado

```
Python 3.10+
├── Data Processing
│   ├── pandas
│   ├── numpy
│   └── scikit-learn
├── NLP
│   ├── spacy (para NER)
│   ├── nltk (para tokenización)
│   └── textblob (para sentimientos MVP)
├── ML
│   ├── scikit-learn
│   ├── xgboost/lightgbm
│   └── imbalanced-learn (SMOTE)
├── Deep Learning (opcional)
│   ├── transformers (HuggingFace)
│   └── torch/tensorflow
└── Visualización
    ├── streamlit (recomendado)
    ├── plotly
    └── matplotlib
```

### Hardware
- CPU: i5 o superior (suficiente)
- RAM: 8GB mínimo (16GB recomendado)
- SSD: 10GB para modelos + datos

---

## 📅 CRONOGRAMA SUGERIDO

| Semana | Fase | Entregables |
|--------|------|------------|
| 1 | Preparación | Dataset limpio + EDA |
| 1-2 | Features | Matriz de características |
| 2-3 | Modelos | 4 clasificadores entrenados |
| 3-4 | Validación | Reportes + análisis errores |
| 4-6 | Integración | App funcional o API |
| 6 | Testing | Documentación + deploy |

---

## ✅ CONCLUSIÓN Y RECOMENDACIÓN FINAL

### Veredicto: **PROYECTO VIABLE Y RECOMENDADO**

**Razones:**
1. ✅ Dataset suficiente y de buena calidad
2. ✅ Todas las columnas necesarias disponibles
3. ✅ Clases claras para clasificación
4. ✅ Texto detallado para análisis
5. ✅ Aplicación práctica inmediata

**Empezar por:**
1. Fase 1-2: Preparación + Features (baja complejidad)
2. Fase 3: Iniciar con Entidad Responsable (mayor éxito inicial)
3. Escalar a Sentimientos + Severidad (mayor valor)

**Próximos pasos:**
1. Confirmar acceso a datos completamente limpios
2. Definir métricas de éxito con stakeholders
3. Asignar recursos (tiempo + computación)
4. Crear equipo de validación experto (2-3 personas)

---

## 📞 Preguntas Frecuentes

**P: ¿Necesitamos más datos para entrenar los modelos?**  
R: 150 PQRS es aceptable para empezar, pero 500+ sería ideal para producción. Podemos usar transfer learning para optimizar.

**P: ¿Cuánta anotación manual se requiere?**  
R: Para Entidad y Tipo: ~10% de validación (15 muestras). Para Sentimientos: 30-50 muestras si hacemos fine-tuning.

**P: ¿Podemos hacer predicciones en tiempo real?**  
R: Sí. Modelos simples (<50ms), BERT (~500ms). Recomendamos API con caché.

**P: ¿Cómo validamos que los modelos funcionan bien?**  
R: Cross-validation + validación manual de expertos + test set independiente.

---

**Documento preparado para:** Equipo de Desarrollo PQRS  
**Próxima revisión:** Después de completar Fase 1

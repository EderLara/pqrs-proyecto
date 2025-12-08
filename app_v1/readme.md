# PQRS Intelligent Classifier

Aplicación basada en Machine Learning para la clasificación automática de Peticiones, Quejas, Reclamos y Sugerencias en proyectos de infraestructura vial.

## Características

1. **Gestión de Datos**: Carga de archivos CSV/Excel, limpieza automática y extracción de features (TF-IDF).
2. **Análisis Exploratorio (EDA)**: Visualización interactiva de distribuciones de datos.
3. **Modelado**: Entrenamiento de modelos (Regresión Logística y Random Forest) con versionamiento automático.
4. **Predicción**: Interfaz para clasificar nuevos textos con selección de versión de modelo.
5. **Persistencia**: Base de datos SQLite para gestión de usuarios e historial de predicciones.
6. **Seguridad**: Sistema de Login/Registro con contraseñas encriptadas.

## Arquitectura

El proyecto sigue los principios SOLID:
- **AuthManager**: Responsabilidad única de autenticación.
- **DatabaseManager**: Abstracción de la capa de datos.
- **DataPipeline**: Lógica de transformación ETL.
- **ModelEngine**: Gestión del ciclo de vida del modelo ML.
- **Visualizer**: Generación de gráficos desacoplada.

## Instalación

1. Clonar el repositorio.
2. Crear un entorno virtual: `python -m venv venv`.
3. Activar entorno e instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
4. Ejecutar la aplicación:
    ```bash
    streamlit run app.py
    ```

## Uso

1. Regístrese con un usuario y contraseña.
2. Vaya a la pestaña "Carga de Datos" y suba el archivo pqrs.csv.
3. Ejecute "Limpiar Datos".
4. Vaya a "Entrenamiento", asigne un nombre a la versión y entrene.
5. Use la pestaña "Predicción" para probar el modelo.

### Notas de Implementación

1.  **SOLID**:
    * **SRP**: Cada clase (`AuthManager`, `ModelEngine`, etc.) hace una sola cosa.
    * **OCP**: `Visualizer` y `ModelEngine` pueden extenderse (nuevos tipos de gráficos o modelos) sin modificar el código que los llama.
    * **DIP**: `app.py` depende de las clases abstractas/servicios, no de implementaciones de bajo nivel directas.

2.  **Versioning**: Se implementa guardando carpetas con timestamp o nombre personalizado dentro de `models/`. El archivo `metrics.json` permite revisar qué tan bueno fue ese modelo posteriormente.

3.  **Base de Datos**: Se utiliza SQLite por simplicidad y portabilidad, guardando tanto usuarios como logs de predicción.

4.  **Flujo**: El usuario no puede predecir si no ha seleccionado una versión válida, y no puede ver versiones si no ha entrenado al menos una vez.
import streamlit as st
import pandas as pd
from datetime import datetime
from src.auth import AuthManager
from src.database_manager import DatabaseManager
from src.data_loader import DataPipeline
from src.model_engine import ModelEngine
from src.visualizer import Visualizer

# Configuración de página
st.set_page_config(page_title="PQRS Classifier", layout="wide")

# Inicialización de servicios
if 'db' not in st.session_state:
    st.session_state.db = DatabaseManager()
if 'auth' not in st.session_state:
    st.session_state.auth = AuthManager(st.session_state.db)

# Gestión de Sesión
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

def login_page():
    st.title("PQRS Intelligent Classifier - Acceso")
    tab1, tab2 = st.tabs(["Iniciar Sesión", "Registrarse"])
    
    with tab1:
        username = st.text_input("Usuario", key="login_user")
        password = st.text_input("Contraseña", type="password", key="login_pass")
        if st.button("Entrar"):
            if st.session_state.auth.login(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Credenciales inválidas")

    with tab2:
        new_user = st.text_input("Nuevo Usuario", key="reg_user")
        new_pass = st.text_input("Nueva Contraseña", type="password", key="reg_pass")
        if st.button("Crear Cuenta"):
            if st.session_state.auth.register(new_user, new_pass):
                st.success("Usuario creado. Por favor inicie sesión.")
            else:
                st.error("El usuario ya existe")

def main_app():
    st.sidebar.title(f"Bienvenido, {st.session_state.username}")
    if st.sidebar.button("Cerrar Sesión"):
        st.session_state.authenticated = False
        st.rerun()

    st.title("🏭 PQRS Intelligent Classifier")
    
    # Pestañas principales
    tabs = st.tabs([
        "1. Carga de Datos", 
        "2. EDA & Features", 
        "3. Entrenamiento", 
        "4. Predicción", 
        "5. Historial"
    ])

    data_pipeline = DataPipeline()
    model_engine = ModelEngine()

    # --- Pestaña 1: Datos ---
    with tabs[0]:
        st.header("Carga y Preparación de Datos")
        uploaded_file = st.file_uploader("Subir dataset (CSV/Excel)", type=['csv', 'xlsx'])
        
        if uploaded_file:
            df = data_pipeline.load_data(uploaded_file)
            st.session_state.df_raw = df
            st.write("Vista previa de datos crudos:", df.head())
            
            if st.button("Limpiar Datos"):
                df_clean = data_pipeline.clean_data(df)
                st.session_state.df_clean = df_clean
                st.success(f"Datos limpiados. Filas originales: {len(df)}, Filas limpias: {len(df_clean)}")
                st.dataframe(df_clean.head())

    # --- Pestaña 2: EDA ---
    with tabs[1]:
        st.header("Análisis Exploratorio")
        if 'df_clean' in st.session_state:
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(Visualizer.plot_distribution(
                    st.session_state.df_clean, 'ENTIDAD RESPONSABLE', 'Distribución de Entidades'))
            with col2:
                st.pyplot(Visualizer.plot_distribution(
                    st.session_state.df_clean, 'TIPOS DE HECHO', 'Distribución de Tipos de Hecho'))
        else:
            st.warning("Por favor cargue y limpie los datos primero.")

    # --- Pestaña 3: Entrenamiento ---
    with tabs[2]:
        st.header("Modelado y Evaluación")
        
        if 'df_clean' in st.session_state:
            version_name = st.text_input("Nombre de la versión del modelo", 
                                       value=f"v_{datetime.now().strftime('%Y%m%d_%H%M')}")
            
            if st.button("Entrenar Modelo"):
                with st.spinner("Entrenando modelos..."):
                    X, y_ent, y_iss, vectorizer = data_pipeline.get_features(st.session_state.df_clean)
                    metrics = model_engine.train(X, y_ent, y_iss, vectorizer)
                    
                    # Guardar
                    model_engine.save_version(version_name)
                    st.success(f"Modelo {version_name} entrenado y guardado exitosamente.")
                    
                    # Mostrar métricas
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Clasificador de Entidad")
                        st.json(metrics['entity'])
                        st.pyplot(Visualizer.plot_confusion_matrix(
                            metrics['cm_entity'], metrics['labels_entity'], "Matriz Confusión - Entidad"))
                    
                    with col2:
                        st.subheader("Clasificador de Tipo de Hecho")
                        st.json(metrics['issue'])
                        st.pyplot(Visualizer.plot_confusion_matrix(
                            metrics['cm_issue'], metrics['labels_issue'], "Matriz Confusión - Hecho"))
        else:
            st.info("Requiere datos limpios para entrenar.")

    # --- Pestaña 4: Predicción ---
    with tabs[3]:
        st.header("Realizar Predicción")
        
        versions = model_engine.get_available_versions()
        selected_version = st.selectbox("Seleccionar Versión del Modelo", versions)
        
        if selected_version:
            model_engine.load_version(selected_version)
            st.success(f"Modelo {selected_version} cargado.")
            
            input_text = st.text_area("Ingrese la descripción del PQRS:")
            
            if st.button("Clasificar"):
                if input_text:
                    result = model_engine.predict(input_text)
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Entidad Responsable", result['entity'], f"{result['entity_confidence']*100}% Confianza")
                    col2.metric("Tipo de Hecho", result['issue'], f"{result['issue_confidence']*100}% Confianza")
                    
                    # Guardar en DB
                    st.session_state.db.save_prediction(
                        st.session_state.username, input_text, result, selected_version
                    )
                else:
                    st.warning("Ingrese un texto.")
        else:
            st.warning("No hay modelos disponibles. Entrene uno primero.")

    # --- Pestaña 5: Reporte ---
    with tabs[4]:
        st.header("Historial de Predicciones")
        order = st.radio("Orden", ["Más recientes primero", "Más antiguas primero"])
        sql_order = "DESC" if order == "Más recientes primero" else "ASC"
        
        df_history = st.session_state.db.get_predictions_history(order=sql_order)
        st.dataframe(df_history)
        
        csv = df_history.to_csv(index=False).encode('utf-8')
        st.download_button("Descargar Reporte CSV", csv, "reporte_predicciones.csv")

if __name__ == "__main__":
    if st.session_state.authenticated:
        main_app()
    else:
        login_page()
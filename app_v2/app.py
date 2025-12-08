# app_improved.py
"""
Aplicación Streamlit mejorada con:
- Análisis de sentimientos en predicciones
- Gráficos avanzados con Plotly
- Dashboard de calidad de datos
- Análisis profundo de datasets
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from src.auth import AuthManager
from src.database_manager import DatabaseManager
from src.data_loader import DataPipeline
from src.model_engine import ModelEngine
from src.visualizer import Visualizer
from src.sentiment_analyzer import SentimentAnalyzer
from src.visualizer_enhanced import EnhancedVisualizer

# Configuración de página
st.set_page_config(
    page_title="PQRS Classifier Mejorado",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicialización de servicios (usar cache_resource para evitar reinicialización)
@st.cache_resource
def init_services():
    """Inicializa servicios una sola vez."""
    db = DatabaseManager()
    auth = AuthManager(db)
    sentiment = SentimentAnalyzer()
    return db, auth, sentiment

db_manager, auth_manager, sentiment_analyzer = init_services()

# Gestión de Sesión
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'db' not in st.session_state:
    st.session_state.db = db_manager
if 'auth' not in st.session_state:
    st.session_state.auth = auth_manager

# Estilos personalizados
st.markdown("""
    <style>
    .sentiment-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .very-negative {
        background-color: #d62828;
        color: white;
    }
    .negative {
        background-color: #f77f00;
        color: white;
    }
    .neutral {
        background-color: #ffd60a;
        color: black;
    }
    .positive {
        background-color: #90e0ef;
        color: black;
    }
    .very-positive {
        background-color: #06a77d;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

def login_page():
    """Página de autenticación."""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.title("🔐 PQRS Classifier")
        st.markdown("### Acceso a la Aplicación")
        
        tab1, tab2 = st.tabs(["Iniciar Sesión", "Registrarse"])
        
        with tab1:
            username = st.text_input("Usuario", key="login_user")
            password = st.text_input("Contraseña", type="password", key="login_pass")
            
            if st.button("🔓 Entrar", use_container_width=True):
                if st.session_state.auth.login(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("❌ Credenciales inválidas")
        
        with tab2:
            new_user = st.text_input("Nuevo Usuario", key="reg_user")
            new_pass = st.text_input("Nueva Contraseña", type="password", key="reg_pass")
            confirm_pass = st.text_input("Confirmar Contraseña", type="password", key="reg_pass_conf")
            
            if st.button("📝 Crear Cuenta", use_container_width=True):
                if new_pass != confirm_pass:
                    st.error("Las contraseñas no coinciden")
                elif len(new_pass) < 6:
                    st.error("La contraseña debe tener al menos 6 caracteres")
                elif st.session_state.auth.register(new_user, new_pass):
                    st.success("✅ Usuario creado. Por favor inicie sesión.")
                else:
                    st.error("❌ El usuario ya existe")

def main_app():
    """Aplicación principal."""
    st.sidebar.title(f"👤 {st.session_state.username}")
    
    if st.sidebar.button("🚪 Cerrar Sesión", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()
    
    st.title("🏭 PQRS Intelligent Classifier - Versión Mejorada")
    
    # Pestañas principales
    tabs = st.tabs([
        "1️⃣ Carga de Datos",
        "2️⃣ Dashboard de Calidad",
        "3️⃣ EDA Avanzado",
        "4️⃣ Entrenamiento",
        "5️⃣ Predicción con Sentimiento",
        "6️⃣ Historial"
    ])
    
    data_pipeline = DataPipeline()
    model_engine = ModelEngine()
    
    # === PESTAÑA 1: CARGA DE DATOS ===
    with tabs[0]:
        st.header("📥 Carga y Preparación de Datos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_file = st.file_uploader(
                "Subir dataset (CSV/Excel)",
                type=['csv', 'xlsx'],
                help="Cargue el archivo de PQRS"
            )
        
        if uploaded_file:
            df = data_pipeline.load_data(uploaded_file)
            st.session_state.df_raw = df
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Registros", len(df))
            col2.metric("Columnas", len(df.columns))
            col3.metric("Tamaño (MB)", round(df.memory_usage(deep=True).sum() / 1024**2, 2))
            
            st.subheader("📋 Vista previa de datos crudos")
            st.dataframe(df.head(10), use_container_width=True)
            
            if st.button("🧹 Limpiar Datos", use_container_width=True):
                df_clean = data_pipeline.clean_data(df)
                st.session_state.df_clean = df_clean
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Registros limpios", len(df_clean))
                col2.metric("Eliminados", len(df) - len(df_clean))
                col3.metric("% Eliminados", f"{(len(df)-len(df_clean))/len(df)*100:.1f}%")
                col4.metric("Duplicados", df_clean.duplicated().sum())
                
                st.success("✅ Datos limpiados exitosamente")
                st.dataframe(df_clean.head(10), use_container_width=True)
    
    # === PESTAÑA 2: DASHBOARD DE CALIDAD ===
    with tabs[1]:
        st.header("📊 Dashboard de Calidad del Dataset")
        
        if 'df_raw' in st.session_state and 'df_clean' in st.session_state:
            df_raw = st.session_state.df_raw
            df_clean = st.session_state.df_clean
            
            # Crear reporte de calidad
            quality_report = EnhancedVisualizer.create_quality_report(df_raw, df_clean)
            
            # Métricas principales
            st.subheader("📈 Métricas de Calidad")
            
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric(
                "Completitud Antes",
                f"{quality_report['raw_completitud']:.1f}%",
                f"{quality_report['clean_completitud'] - quality_report['raw_completitud']:.1f}%",
                delta_color="inverse"
            )
            col2.metric(
                "Completitud Después",
                f"{quality_report['clean_completitud']:.1f}%"
            )
            col3.metric(
                "Registros Eliminados",
                f"{quality_report['records_removed']} ({quality_report['records_removed_pct']}%)"
            )
            col4.metric(
                "Score de Calidad",
                f"{quality_report['quality_score']:.1f}/100"
            )
            
            st.markdown("---")
            
            # Gráfico comparativo
            st.subheader("📉 Comparación Antes/Después")
            fig_comparison = EnhancedVisualizer.plot_data_quality_before_after(df_raw, df_clean)
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Tabla detallada
            st.subheader("📋 Detalles de Calidad")
            
            quality_data = {
                'Métrica': [
                    'Total Registros',
                    'Valores Nulos',
                    'Duplicados',
                    'Completitud (%)'
                ],
                'Antes': [
                    quality_report['raw_records'],
                    quality_report['raw_nulls'],
                    quality_report['raw_duplicates'],
                    f"{quality_report['raw_completitud']:.2f}%"
                ],
                'Después': [
                    quality_report['clean_records'],
                    quality_report['clean_nulls'],
                    quality_report['clean_duplicates'],
                    f"{quality_report['clean_completitud']:.2f}%"
                ]
            }
            
            df_quality = pd.DataFrame(quality_data)
            st.dataframe(df_quality, use_container_width=True, hide_index=True)
            
        else:
            st.info("⚠️ Cargue y limpie los datos primero en la pestaña anterior.")
    
    # === PESTAÑA 3: EDA AVANZADO ===
    with tabs[2]:
        st.header("🔍 Análisis Exploratorio Avanzado")
        
        if 'df_clean' in st.session_state:
            df = st.session_state.df_clean
            
            # Seleccionar visualizaciones
            viz_options = st.multiselect(
                "Seleccione visualizaciones",
                [
                    "📊 Distribución Entidades (Barras)",
                    "🥧 Distribución Entidades (Pastel)",
                    "📊 Distribución Tipos de Hecho (Barras)",
                    "🥧 Distribución Tipos de Hecho (Pastel)",
                    "📏 Longitud de Texto",
                    "🔤 Palabras Más Frecuentes",
                    "🔥 Correlación Entidad vs Hecho"
                ],
                default=["📊 Distribución Entidades (Barras)", "📊 Distribución Tipos de Hecho (Barras)"]
            )
            
            if "📊 Distribución Entidades (Barras)" in viz_options:
                st.plotly_chart(
                    EnhancedVisualizer.plot_distribution_bar(
                        df, 'ENTIDAD RESPONSABLE', 'Distribución de Entidades'
                    ),
                    use_container_width=True
                )
            
            if "🥧 Distribución Entidades (Pastel)" in viz_options:
                st.plotly_chart(
                    EnhancedVisualizer.plot_distribution_pie(
                        df, 'ENTIDAD RESPONSABLE', 'Entidades Responsables'
                    ),
                    use_container_width=True
                )
            
            if "📊 Distribución Tipos de Hecho (Barras)" in viz_options:
                st.plotly_chart(
                    EnhancedVisualizer.plot_distribution_bar(
                        df, 'TIPOS DE HECHO', 'Distribución de Tipos de Hecho'
                    ),
                    use_container_width=True
                )
            
            if "🥧 Distribución Tipos de Hecho (Pastel)" in viz_options:
                st.plotly_chart(
                    EnhancedVisualizer.plot_distribution_pie(
                        df, 'TIPOS DE HECHO', 'Tipos de Hecho'
                    ),
                    use_container_width=True
                )
            
            if "📏 Longitud de Texto" in viz_options:
                st.plotly_chart(
                    EnhancedVisualizer.plot_text_length_distribution(
                        df, 'DESCRIPCION_LIMPIA'
                    ),
                    use_container_width=True
                )
            
            if "🔤 Palabras Más Frecuentes" in viz_options:
                top_n = st.slider("Número de palabras a mostrar", 10, 50, 20)
                st.plotly_chart(
                    EnhancedVisualizer.plot_top_words(df, 'DESCRIPCION_LIMPIA', top_n),
                    use_container_width=True
                )
            
            if "🔥 Correlación Entidad vs Hecho" in viz_options:
                st.plotly_chart(
                    EnhancedVisualizer.plot_correlation_heatmap(
                        df, 'ENTIDAD RESPONSABLE', 'TIPOS DE HECHO'
                    ),
                    use_container_width=True
                )
        else:
            st.warning("⚠️ Por favor cargue y limpie los datos primero.")
    
    # === PESTAÑA 4: ENTRENAMIENTO ===
    with tabs[3]:
        st.header("🧠 Modelado y Evaluación")
        
        if 'df_clean' in st.session_state:
            col1, col2 = st.columns(2)
            
            with col1:
                version_name = st.text_input(
                    "Nombre de la versión del modelo",
                    value=f"v_{datetime.now().strftime('%Y%m%d_%H%M')}"
                )
            
            with col2:
                if st.button("🚀 Entrenar Modelo", use_container_width=True):
                    with st.spinner("Entrenando modelos..."):
                        X, y_ent, y_iss, vectorizer = data_pipeline.get_features(st.session_state.df_clean)
                        metrics = model_engine.train(X, y_ent, y_iss, vectorizer)
                        model_engine.save_version(version_name)
                        st.success(f"✅ Modelo {version_name} entrenado y guardado")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Entity Classifier")
                            st.metric("Accuracy", f"{metrics['entity']['accuracy']:.2%}")
                            st.json(metrics['entity'])
                            st.pyplot(Visualizer.plot_confusion_matrix(
                                metrics['cm_entity'], 
                                metrics['labels_entity'],
                                "Matriz Confusión - Entidad"
                            ))
                        
                        with col2:
                            st.subheader("Issue Classifier")
                            st.metric("Accuracy", f"{metrics['issue']['accuracy']:.2%}")
                            st.json(metrics['issue'])
                            st.pyplot(Visualizer.plot_confusion_matrix(
                                metrics['cm_issue'],
                                metrics['labels_issue'],
                                "Matriz Confusión - Hecho"
                            ))
        else:
            st.info("⚠️ Cargue y limpie los datos primero.")
    
    # === PESTAÑA 5: PREDICCIÓN CON SENTIMIENTO ===
    with tabs[4]:
        st.header("🎯 Realizar Predicción con Análisis de Sentimientos")
        
        versions = model_engine.get_available_versions()
        
        if versions:
            selected_version = st.selectbox("Seleccionar Versión del Modelo", versions)
            
            if selected_version:
                model_engine.load_version(selected_version)
                st.success(f"✅ Modelo {selected_version} cargado")
                
                input_text = st.text_area(
                    "📝 Ingrese la descripción del PQRS:",
                    height=150,
                    placeholder="Escriba aquí la descripción del PQRS..."
                )
                
                if st.button("🔍 Clasificar y Analizar Sentimiento", use_container_width=True):
                    if input_text.strip():
                        # Predicción ML
                        result = model_engine.predict(input_text)
                        
                        # Análisis de sentimientos
                        sentiment_result = sentiment_analyzer.analyze_sentiment(input_text)
                        
                        st.markdown("---")
                        st.subheader("📊 Resultados de Clasificación ML")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric(
                                "🏢 Entidad Responsable",
                                result['entity'],
                                f"{result['entity_confidence']*100:.1f}% Confianza"
                            )
                        
                        with col2:
                            st.metric(
                                "📋 Tipo de Hecho",
                                result['issue'],
                                f"{result['issue_confidence']*100:.1f}% Confianza"
                            )
                        
                        st.markdown("---")
                        st.subheader("😊 Análisis de Sentimientos")
                        
                        # Mostrar gauge de sentimiento
                        fig_sentiment = EnhancedVisualizer.plot_sentiment_gauge(
                            sentiment_result['sentiment_score'],
                            sentiment_result['confidence']
                        )
                        st.plotly_chart(fig_sentiment, use_container_width=True)
                        
                        # Detalles de sentimiento
                        col1, col2, col3, col4 = st.columns(4)
                        
                        col1.metric(
                            "Sentimiento",
                            sentiment_result['sentiment_label'],
                            sentiment_result['emoji']
                        )
                        col2.metric(
                            "Score",
                            f"{sentiment_result['sentiment_score']:.2f}",
                            "(-1 a +1)"
                        )
                        col3.metric(
                            "Confianza",
                            f"{sentiment_result['confidence']*100:.0f}%"
                        )
                        col4.metric(
                            "Emociones",
                            f"{len(sentiment_result['emotions'])} detectadas"
                        )
                        
                        # Emociones detectadas
                        st.markdown("**🎭 Emociones detectadas:**")
                        emotion_cols = st.columns(len(sentiment_result['emotions']))
                        for i, emotion in enumerate(sentiment_result['emotions']):
                            with emotion_cols[i]:
                                st.info(emotion)
                        
                        # Guardar predicción
                        st.session_state.db.save_prediction(
                            st.session_state.username,
                            input_text,
                            result,
                            selected_version
                        )
                        
                        st.success("✅ Predicción guardada en historial")
                    else:
                        st.warning("⚠️ Por favor ingrese un texto")
        else:
            st.warning("⚠️ No hay modelos disponibles. Entrene uno primero en la pestaña anterior.")
    
    # === PESTAÑA 6: HISTORIAL ===
    with tabs[5]:
        st.header("📜 Historial de Predicciones")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            order = st.radio(
                "Orden",
                ["⬇️ Más recientes primero", "⬆️ Más antiguas primero"],
                horizontal=True
            )
        
        sql_order = "DESC" if "⬇️" in order else "ASC"
        
        df_history = st.session_state.db.get_predictions_history(order=sql_order)
        
        if not df_history.empty:
            st.dataframe(df_history, use_container_width=True)
            
            csv = df_history.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descargar Reporte CSV",
                data=csv,
                file_name=f"reporte_predicciones_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("📭 No hay predicciones en el historial aún.")

# Lógica principal
if __name__ == "__main__":
    if st.session_state.authenticated:
        main_app()
    else:
        login_page()

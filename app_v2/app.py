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
            
            if st.button("🔓 Entrar", width='stretch'):
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
            
            if st.button("📝 Crear Cuenta", width='stretch'):
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
    
    if st.sidebar.button("🚪 Cerrar Sesión", width='stretch'):
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
            st.dataframe(df.head(10), width='stretch')
            
            if st.button("🧹 Limpiar Datos", width='stretch'):
                df_clean = data_pipeline.clean_data(df)
                st.session_state.df_clean = df_clean
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Registros limpios", len(df_clean))
                col2.metric("Eliminados", len(df) - len(df_clean))
                col3.metric("% Eliminados", f"{(len(df)-len(df_clean))/len(df)*100:.1f}%")
                col4.metric("Duplicados", df_clean.duplicated().sum())
                
                st.success("✅ Datos limpiados exitosamente")
                st.dataframe(df_clean.head(10), width='stretch')
    
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
            st.plotly_chart(fig_comparison, width='stretch')
            
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
            st.dataframe(df_quality, width='stretch', hide_index=True)
            
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
                    width='stretch'
                )
            
            if "🥧 Distribución Entidades (Pastel)" in viz_options:
                st.plotly_chart(
                    EnhancedVisualizer.plot_distribution_pie(
                        df, 'ENTIDAD RESPONSABLE', 'Entidades Responsables'
                    ),
                    width='stretch'
                )
            
            if "📊 Distribución Tipos de Hecho (Barras)" in viz_options:
                st.plotly_chart(
                    EnhancedVisualizer.plot_distribution_bar(
                        df, 'TIPOS DE HECHO', 'Distribución de Tipos de Hecho'
                    ),
                    width='stretch'
                )
            
            if "🥧 Distribución Tipos de Hecho (Pastel)" in viz_options:
                st.plotly_chart(
                    EnhancedVisualizer.plot_distribution_pie(
                        df, 'TIPOS DE HECHO', 'Tipos de Hecho'
                    ),
                    width='stretch'
                )
            
            if "📏 Longitud de Texto" in viz_options:
                st.plotly_chart(
                    EnhancedVisualizer.plot_text_length_distribution(
                        df, 'DESCRIPCION_LIMPIA'
                    ),
                    width='stretch'
                )
            
            if "🔤 Palabras Más Frecuentes" in viz_options:
                top_n = st.slider("Número de palabras a mostrar", 10, 50, 20)
                st.plotly_chart(
                    EnhancedVisualizer.plot_top_words(df, 'DESCRIPCION_LIMPIA', top_n),
                    width='stretch'
                )
            
            if "🔥 Correlación Entidad vs Hecho" in viz_options:
                st.plotly_chart(
                    EnhancedVisualizer.plot_correlation_heatmap(
                        df, 'ENTIDAD RESPONSABLE', 'TIPOS DE HECHO'
                    ),
                    width='stretch'
                )
        else:
            st.warning("⚠️ Por favor cargue y limpie los datos primero.")
    


    # === PESTAÑA 4: ENTRENAMIENTO MEJORADA ===
    with tabs[3]:
        st.header("🧠 Modelado y Evaluación")
        
        if 'df_clean' not in st.session_state:
            st.warning("⚠️ Cargue y limpie los datos primero en la Pestaña 1")
            st.stop()
        
        # ─────────────────────────────────────────────────────────────────────
        # SECCIÓN 1: CONFIGURACIÓN DEL MODELO
        # ─────────────────────────────────────────────────────────────────────
        
        st.markdown("### ⚙️ Configuración del Modelo")
        
        col1, col2, col3 = st.columns([2, 1, 1], vertical_alignment='center')
        
        with col1:
            version_name = st.text_input(
                "📝 Nombre de la versión",
                value=f"v_{datetime.now().strftime('%Y%m%d_%H%M')}",
                help="Nombre único para identificar esta versión del modelo"
            )
        
        with col2:
            train_btn = st.button("🚀 Entrenar", width='stretch', key="train_btn")
        
        with col3:
            st.info(f"📊 Datos: {len(st.session_state.df_clean)} registros")
        
        # ─────────────────────────────────────────────────────────────────────
        # SECCIÓN 2: ENTRENAMIENTO
        # ─────────────────────────────────────────────────────────────────────
        
        if train_btn:
            # Placeholder para progreso
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            
            try:
                # Paso 1: Extraer features
                with status_placeholder.container():
                    with st.spinner("📊 Extrayendo features..."):
                        X, y_ent, y_iss, vectorizer = data_pipeline.get_features(
                            st.session_state.df_clean
                        )
                
                # Paso 2: Entrenar modelos
                with status_placeholder.container():
                    with st.spinner("🧠 Entrenando modelos..."):
                        metrics = model_engine.train(X, y_ent, y_iss, vectorizer)
                
                # Paso 3: Guardar
                with status_placeholder.container():
                    with st.spinner("💾 Guardando..."):
                        model_engine.save_version(version_name)
                
                # Éxito
                st.success(f"✅ Modelo **{version_name}** entrenado y guardado exitosamente!")
                
                # ─────────────────────────────────────────────────────────────
                # SECCIÓN 3: RESUMEN DE RESULTADOS
                # ─────────────────────────────────────────────────────────────
                
                st.markdown("---")
                st.markdown("### 📊 Resultados del Entrenamiento")
                
                # Métricas principales en cards
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    entity_acc = metrics['entity'].get('accuracy', 0)
                    st.metric(
                        "Entity Accuracy",
                        f"{entity_acc:.1%}",
                        delta=f"+{(entity_acc-0.85)*100:.1f}%" if entity_acc > 0.85 else None,
                        delta_color="inverse" if entity_acc < 0.85 else "off"
                    )
                
                with metric_col2:
                    issue_acc = metrics['issue'].get('accuracy', 0)
                    st.metric(
                        "Issue Accuracy",
                        f"{issue_acc:.1%}",
                        delta=f"+{(issue_acc-0.80)*100:.1f}%" if issue_acc > 0.80 else None,
                        delta_color="inverse" if issue_acc < 0.80 else "off"
                    )
                
                with metric_col3:
                    entity_f1 = metrics['entity'].get('weighted avg', {}).get('f1-score', 0)
                    st.metric(
                        "Entity F1-Score",
                        f"{entity_f1:.1%}"
                    )
                
                with metric_col4:
                    issue_f1 = metrics['issue'].get('weighted avg', {}).get('f1-score', 0)
                    st.metric(
                        "Issue F1-Score",
                        f"{issue_f1:.1%}"
                    )
                
                st.markdown("---")
                
                # ─────────────────────────────────────────────────────────────
                # SECCIÓN 4: COMPARACIÓN DE MODELOS
                # ─────────────────────────────────────────────────────────────
                
                st.markdown("### 🤖 Detalles de Modelos")
                
                tab_entity, tab_issue = st.tabs([
                    "🏢 Entity Classifier (Logistic Regression)",
                    "📋 Issue Classifier (Random Forest)"
                ])
                
                # ─── TAB 1: ENTITY CLASSIFIER ───
                with tab_entity:
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.subheader("📈 Métricas")
                        entity_metrics = metrics['entity']
                        
                        # Accuracy destacado
                        acc = entity_metrics.get('accuracy', 0)
                        st.metric("Accuracy (Global)", f"{acc:.2%}")
                        
                        # Precision, Recall, F1
                        if 'weighted avg' in entity_metrics:
                            weighted = entity_metrics['weighted avg']
                            st.metric("Precision", f"{weighted.get('precision', 0):.2%}")
                            st.metric("Recall", f"{weighted.get('recall', 0):.2%}")
                            st.metric("F1-Score", f"{weighted.get('f1-score', 0):.2%}")
                    
                    with col2:
                        st.subheader("📊 Matriz de Confusión")
                        fig_cm = Visualizer.plot_confusion_matrix(
                            metrics['cm_entity'],
                            metrics['labels_entity'],
                            "Entity Classifier"
                        )
                        st.pyplot(fig_cm, width='stretch')
                    
                    # Expandible: Detalles por clase
                    with st.expander("📋 Detalles por clase"):
                        entity_detail = pd.DataFrame(
                            entity_metrics
                        ).drop(columns=['accuracy', 'macro avg', 'weighted avg'], errors='ignore').T
                        
                        st.dataframe(
                            entity_detail.style.format("{:.2%}"),
                            width='stretch'
                        )
                
                # ─── TAB 2: ISSUE CLASSIFIER ───
                with tab_issue:
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.subheader("📈 Métricas")
                        issue_metrics = metrics['issue']
                        
                        # Accuracy destacado
                        acc = issue_metrics.get('accuracy', 0)
                        st.metric("Accuracy (Global)", f"{acc:.2%}")
                        
                        # Precision, Recall, F1
                        if 'weighted avg' in issue_metrics:
                            weighted = issue_metrics['weighted avg']
                            st.metric("Precision", f"{weighted.get('precision', 0):.2%}")
                            st.metric("Recall", f"{weighted.get('recall', 0):.2%}")
                            st.metric("F1-Score", f"{weighted.get('f1-score', 0):.2%}")
                    
                    with col2:
                        st.subheader("📊 Matriz de Confusión")
                        fig_cm = Visualizer.plot_confusion_matrix(
                            metrics['cm_issue'],
                            metrics['labels_issue'],
                            "Issue Classifier"
                        )
                        st.pyplot(fig_cm, width='stretch')
                    
                    # Expandible: Detalles por clase
                    with st.expander("📋 Detalles por clase"):
                        issue_detail = pd.DataFrame(
                            issue_metrics
                        ).drop(columns=['accuracy', 'macro avg', 'weighted avg'], errors='ignore').T
                        
                        st.dataframe(
                            issue_detail.style.format("{:.2%}"),
                            width='stretch'
                        )
                
                # ─────────────────────────────────────────────────────────────
                # SECCIÓN 5: COMPARACIÓN VISUAL
                # ─────────────────────────────────────────────────────────────
                
                st.markdown("---")
                st.markdown("### 📊 Comparación de Modelos")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Gráfico de barras: Accuracy por modelo
                    import plotly.graph_objects as go
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=['Entity', 'Issue'],
                            y=[
                                metrics['entity'].get('accuracy', 0),
                                metrics['issue'].get('accuracy', 0)
                            ],
                            marker=dict(
                                color=['#06a77d', '#90e0ef']
                            ),
                            text=[
                                f"{metrics['entity'].get('accuracy', 0):.1%}",
                                f"{metrics['issue'].get('accuracy', 0):.1%}"
                            ],
                            textposition='auto'
                        )
                    ])
                    
                    fig.update_layout(
                        title="Accuracy por Modelo",
                        yaxis_title="Accuracy",
                        xaxis_title="Clasificador",
                        height=400,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, width='stretch')
                
                with col2:
                    # Tabla comparativa
                    comparison_data = {
                        'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                        'Entity': [
                            f"{metrics['entity'].get('accuracy', 0):.2%}",
                            f"{metrics['entity'].get('weighted avg', {}).get('precision', 0):.2%}",
                            f"{metrics['entity'].get('weighted avg', {}).get('recall', 0):.2%}",
                            f"{metrics['entity'].get('weighted avg', {}).get('f1-score', 0):.2%}"
                        ],
                        'Issue': [
                            f"{metrics['issue'].get('accuracy', 0):.2%}",
                            f"{metrics['issue'].get('weighted avg', {}).get('precision', 0):.2%}",
                            f"{metrics['issue'].get('weighted avg', {}).get('recall', 0):.2%}",
                            f"{metrics['issue'].get('weighted avg', {}).get('f1-score', 0):.2%}"
                        ]
                    }
                    
                    df_comparison = pd.DataFrame(comparison_data)
                    st.dataframe(df_comparison, width='stretch', hide_index=True)
                
                # ─────────────────────────────────────────────────────────────
                # SECCIÓN 6: INFORMACIÓN Y ACCIONES
                # ─────────────────────────────────────────────────────────────
                
                st.markdown("---")
                st.markdown("### 💾 Información del Modelo")
                
                info_col1, info_col2, info_col3 = st.columns(3)
                
                with info_col1:
                    st.info(f"📦 **Versión**: `{version_name}`")
                
                with info_col2:
                    st.info(f"🕐 **Fecha**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                with info_col3:
                    st.success(f"✅ **Status**: Guardado en disco")
                
                # Notas y recomendaciones
                st.markdown("### 💡 Recomendaciones")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if metrics['entity'].get('accuracy', 0) < 0.85:
                        st.warning(
                            "⚠️ **Entity Accuracy bajo**: Considera recolectar más datos o ajustar features"
                        )
                    else:
                        st.success("✅ Entity Classifier tiene buena precisión")
                
                with col2:
                    if metrics['issue'].get('accuracy', 0) < 0.80:
                        st.warning(
                            "⚠️ **Issue Accuracy bajo**: Revisa balance de clases o aumenta datos"
                        )
                    else:
                        st.success("✅ Issue Classifier tiene buena precisión")
                
                # Opción para usar en predicciones
                st.markdown("---")
                st.markdown("### 🎯 Próximos Pasos")
                
                st.info(
                    f"""
                    ✅ Modelo **{version_name}** entrenado correctamente
                    
                    **Próximo paso**: Ve a la pestaña **"5️⃣ Predicción"** para:
                    - Usar este modelo en predicciones
                    - Analizar sentimientos
                    - Ver resultados con confianza
                    """
                )
            
            except Exception as e:
                st.error(f"❌ Error durante entrenamiento: {str(e)}")
                st.error("Revisa los logs para más detalles")

        
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
                    
                    if st.button("🔍 Clasificar y Analizar Sentimiento", width='stretch'):
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
                            st.plotly_chart(fig_sentiment, width='stretch')
                            
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
            st.dataframe(df_history, width='stretch')
            
            csv = df_history.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descargar Reporte CSV",
                data=csv,
                file_name=f"reporte_predicciones_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                width='stretch'
            )
        else:
            st.info("📭 No hay predicciones en el historial aún.")

# Lógica principal
if __name__ == "__main__":
    if st.session_state.authenticated:
        main_app()
    else:
        login_page()

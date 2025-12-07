# APLICACIÓN STREAMLIT COMPLETA

# ═════════════════════════════════════════════════════════════════════════════
# FILE: app/main.py
# ═════════════════════════════════════════════════════════════════════════════

"""
Aplicación principal de PQRS Intelligent Classifier.

Estructura de navegación:
- 🔐 Auth (Login/Signup)
- 🏠 Home (Dashboard)
- 🔍 Classification (Predicción)
- 📤 Batch Upload (Carga masiva)
- 📊 History (Historial)
- ℹ️  Model Info (Info de modelos)
"""

import streamlit as st
import logging
from pathlib import Path
import sys

# Agregar ruta del proyecto
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.db_manager import DatabaseManager
from src.models.model_manager import ModelManager

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE PÁGINA
# ═════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="PQRS Intelligent Classifier",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═════════════════════════════════════════════════════════════════════════════
# ESTILOS PERSONALIZADOS
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# INICIALIZAR SESIÓN
# ═════════════════════════════════════════════════════════════════════════════

if 'db' not in st.session_state:
    st.session_state.db = DatabaseManager("pqrs_classifier.db")
    logger.info("Database initialized")

if 'model_mgr' not in st.session_state:
    try:
        st.session_state.model_mgr = ModelManager()
        st.session_state.model_mgr.load_latest()
        st.session_state.model_loaded = True
        logger.info("Models loaded successfully")
    except Exception as e:
        st.session_state.model_loaded = False
        logger.warning(f"Could not load models: {e}")

if 'user' not in st.session_state:
    st.session_state.user = None

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# ═════════════════════════════════════════════════════════════════════════════
# BARRA LATERAL
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🔍 PQRS Classifier")
    st.markdown("---")
    
    if st.session_state.authenticated:
        st.success(f"✓ Logged in as: **{st.session_state.user.username}**")
        st.markdown("---")
        
        page = st.radio(
            "Navegación",
            [
                "🏠 Inicio",
                "🔍 Clasificar",
                "📤 Carga Masiva",
                "📊 Historial",
                "ℹ️  Info de Modelos"
            ]
        )
        
        st.markdown("---")
        
        if st.button("🚪 Cerrar Sesión", key="logout_btn", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user = None
            st.rerun()
    else:
        st.warning("Por favor, inicia sesión o registrate")

# ═════════════════════════════════════════════════════════════════════════════
# PÁGINA DE AUTENTICACIÓN (si no está autenticado)
# ═════════════════════════════════════════════════════════════════════════════

if not st.session_state.authenticated:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔓 Iniciar Sesión")
        
        with st.form("login_form", clear_on_submit=True):
            login_username = st.text_input("Usuario")
            login_password = st.text_input("Contraseña", type="password")
            login_submit = st.form_submit_button("Entrar", use_container_width=True)
            
            if login_submit:
                if login_username and login_password:
                    try:
                        user = st.session_state.db.authenticate_user(login_username, login_password)
                        if user:
                            st.session_state.user = user
                            st.session_state.authenticated = True
                            st.success(f"✓ ¡Bienvenido {user.username}!")
                            st.rerun()
                        else:
                            st.error("❌ Usuario o contraseña incorrectos")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                else:
                    st.warning("Por favor completa todos los campos")
    
    with col2:
        st.subheader("📝 Crear Cuenta")
        
        with st.form("signup_form", clear_on_submit=True):
            signup_username = st.text_input("Nuevo usuario")
            signup_email = st.text_input("Email")
            signup_password = st.text_input("Contraseña", type="password")
            signup_confirm = st.text_input("Confirmar contraseña", type="password")
            signup_submit = st.form_submit_button("Registrarse", use_container_width=True)
            
            if signup_submit:
                if not all([signup_username, signup_email, signup_password, signup_confirm]):
                    st.warning("Por favor completa todos los campos")
                elif signup_password != signup_confirm:
                    st.error("Las contraseñas no coinciden")
                elif len(signup_password) < 6:
                    st.error("La contraseña debe tener al menos 6 caracteres")
                else:
                    try:
                        if st.session_state.db.create_user(signup_username, signup_email, signup_password):
                            st.success("✓ Cuenta creada exitosamente. Por favor, inicia sesión.")
                        else:
                            st.error("❌ El usuario o email ya existe")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    st.markdown("---")
    st.info("""
    **Demo Credentials:**
    - Usuario: `admin`
    - Contraseña: `password123`
    """)

# ═════════════════════════════════════════════════════════════════════════════
# PÁGINAS PRINCIPALES (si está autenticado)
# ═════════════════════════════════════════════════════════════════════════════

else:
    # PÁGINA: INICIO
    if page == "🏠 Inicio":
        st.title("🔍 PQRS Intelligent Classifier")
        st.markdown("Sistema inteligente de clasificación de peticiones, quejas y reclamos")
        st.markdown("---")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Entidades", "5 Clases", "Responsables")
        with col2:
            st.metric("Tipos", "6 Clases", "Hechos")
        with col3:
            st.metric("Sentimientos", "4 Niveles", "Análisis")
        with col4:
            st.metric("Severidad", "3 Niveles", "Prioridad")
        
        st.markdown("---")
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("✨ Características")
            st.markdown("""
            ✓ Clasificación automática de entidades responsables
            ✓ Identificación de tipos de hechos
            ✓ Análisis de sentimientos en tiempo real
            ✓ Cálculo automático de severidad
            ✓ Gestión de histórico de predicciones
            ✓ Reportes descargables
            """)
        
        with col_right:
            st.subheader("📊 Modelos")
            st.markdown(f"""
            **Estado:** {'✓ Activos' if st.session_state.model_loaded else '✗ No disponibles'}
            
            Modelos entrenados:
            - Entity Classifier (Random Forest)
            - Issue Classifier (Balanced RF)
            - Sentiment Analyzer (Dictionary + ML)
            - Severity Scorer (Weighted)
            
            **Versión actual:** v1
            """)
        
        st.markdown("---")
        
        # Estadísticas del usuario
        st.subheader("📈 Tus Estadísticas")
        
        try:
            user_predictions = st.session_state.db.get_user_predictions(st.session_state.user.id)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Predicciones", len(user_predictions))
            
            with col2:
                if user_predictions:
                    avg_severity = sum(p[8] for p in user_predictions) / len(user_predictions)
                    st.metric("Severidad Promedio", f"{avg_severity:.2f}/10")
                else:
                    st.metric("Severidad Promedio", "N/A")
            
            with col3:
                st.metric("Últimas 7 días", len([p for p in user_predictions if True]))  # TODO: Filtrar por fecha
        
        except Exception as e:
            st.error(f"Error cargando estadísticas: {e}")
    
    
    # PÁGINA: CLASIFICAR
    elif page == "🔍 Clasificar":
        st.title("🔍 Clasificar PQRS")
        st.markdown("Ingresa los detalles de la petición, queja o reclamo para obtener clasificación automática")
        st.markdown("---")
        
        if not st.session_state.model_loaded:
            st.error("❌ Los modelos no están disponibles. Contacta al administrador.")
        else:
            with st.form("classification_form", clear_on_submit=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    pqrs_number = st.number_input("PQRS No.", min_value=1, value=1)
                    entity_responsible = st.selectbox(
                        "Entidad Responsable (aproximada)",
                        ["Interventor", "Contratista", "Municipio", "SIF", "Otra"]
                    )
                
                with col2:
                    issue_type = st.selectbox(
                        "Tipo de Hecho (aproximado)",
                        ["Ingeniería de la obra", "Movilidad", "Seguridad", "Social", "Ambiental", "Económico"]
                    )
                    municipality = st.text_input("Municipio")
                
                description = st.text_area(
                    "Descripción del Hecho",
                    height=200,
                    placeholder="Escribe aquí la descripción detallada del PQRS..."
                )
                
                model_version = st.selectbox(
                    "Versión de Modelo",
                    ["v1", "v2", "Última"]
                )
                
                submit_button = st.form_submit_button("🚀 Clasificar", use_container_width=True)
                
                if submit_button:
                    if not description.strip():
                        st.error("Por favor ingresa una descripción")
                    else:
                        with st.spinner("Clasificando..."):
                            try:
                                # Realizar predicción
                                prediction = st.session_state.model_mgr.predict(
                                    description,
                                    version=None if model_version == "Última" else model_version
                                )
                                
                                # Guardar en BD
                                st.session_state.db.create_prediction(
                                    user_id=st.session_state.user.id,
                                    pqrs_number=int(pqrs_number),
                                    description=description,
                                    entity=prediction['entity'],
                                    entity_confidence=prediction['entity_confidence'],
                                    issue_type=prediction['issue'],
                                    issue_confidence=prediction['issue_confidence'],
                                    sentiment_level=prediction['sentiment']['level'],
                                    sentiment_score=prediction['sentiment']['polarity'],
                                    severity_score=prediction['severity']['final_score'],
                                    severity_level=prediction['severity']['level'],
                                    model_version=prediction['version']
                                )
                                
                                # Mostrar resultados
                                st.success("✓ Clasificación completada")
                                
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric(
                                        "Entidad",
                                        prediction['entity'],
                                        f"{prediction['entity_confidence']:.1%} confianza"
                                    )
                                
                                with col2:
                                    st.metric(
                                        "Tipo de Hecho",
                                        prediction['issue'],
                                        f"{prediction['issue_confidence']:.1%} confianza"
                                    )
                                
                                with col3:
                                    sentiment = prediction['sentiment']['level']
                                    emoji = "😞" if "NEGATIVE" in sentiment else "😐" if sentiment == "NEUTRAL" else "😊"
                                    st.metric("Sentimiento", f"{emoji} {sentiment}")
                                
                                with col4:
                                    level = prediction['severity']['level']
                                    color = "🔴" if level == "RED" else "🟡" if level == "YELLOW" else "🟢"
                                    score = prediction['severity']['final_score']
                                    st.metric("Severidad", f"{color} {score:.1f}/10")
                                
                                # Detalles adicionales
                                st.markdown("---")
                                st.subheader("📋 Detalles Adicionales")
                                
                                with st.expander("Palabras clave detectadas"):
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.write(f"**Críticas:** {len(prediction['sentiment']['critical_words'])}")
                                        for word in prediction['sentiment']['critical_words'][:5]:
                                            st.write(f"  • {word}")
                                    
                                    with col2:
                                        st.write(f"**Negativas:** {len(prediction['sentiment']['negative_words'])}")
                                        for word in prediction['sentiment']['negative_words'][:5]:
                                            st.write(f"  • {word}")
                                    
                                    with col3:
                                        st.write(f"**Positivas:** {len(prediction['sentiment']['positive_words'])}")
                                        for word in prediction['sentiment']['positive_words'][:5]:
                                            st.write(f"  • {word}")
                                
                                with st.expander("Desglose de Severidad"):
                                    components = prediction['severity']['components']
                                    for component, value in components.items():
                                        st.write(f"{component.capitalize()}: {value:.2f}/10")
                            
                            except Exception as e:
                                st.error(f"Error durante clasificación: {str(e)}")
    
    
    # PÁGINA: HISTORIAL
    elif page == "📊 Historial":
        st.title("📊 Historial de Predicciones")
        st.markdown("---")
        
        try:
            predictions = st.session_state.db.get_user_predictions(st.session_state.user.id)
            
            if not predictions:
                st.info("Aún no tienes predicciones. ¡Comienza clasificando un PQRS!")
            else:
                # Filtros
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    severity_filter = st.selectbox(
                        "Filtrar por Severidad",
                        ["Todos", "RED (Urgente)", "YELLOW (Importante)", "GREEN (Rutinario)"]
                    )
                
                with col2:
                    sort_order = st.radio("Ordenar", ["Más reciente", "Más antiguo", "Mayor severidad"], horizontal=True)
                
                with col3:
                    show_count = st.number_input("Mostrar últimos N registros", min_value=5, max_value=100, value=10)
                
                # Filtrar y ordenar
                filtered = predictions
                
                if "RED" in severity_filter:
                    filtered = [p for p in filtered if p[9] == "RED"]
                elif "YELLOW" in severity_filter:
                    filtered = [p for p in filtered if p[9] == "YELLOW"]
                elif "GREEN" in severity_filter:
                    filtered = [p for p in filtered if p[9] == "GREEN"]
                
                if "Más reciente" in sort_order:
                    filtered = sorted(filtered, key=lambda x: x[0], reverse=True)
                elif "Más antiguo" in sort_order:
                    filtered = sorted(filtered, key=lambda x: x[0])
                else:
                    filtered = sorted(filtered, key=lambda x: x[8], reverse=True)
                
                filtered = filtered[:show_count]
                
                # Mostrar tabla
                st.markdown("---")
                st.subheader(f"Total: {len(filtered)} registros")
                
                # Datos para tabla
                import pandas as pd
                data = []
                for p in filtered:
                    data.append({
                        'ID': p[0],
                        'Fecha': p[1].strftime("%d/%m/%Y %H:%M") if hasattr(p[1], 'strftime') else p[1],
                        'PQRS': p[2],
                        'Entidad': p[4],
                        'Tipo': p[6],
                        'Sentimiento': p[10],
                        'Severidad': f"{p[8]:.1f}/10",
                        'Nivel': p[9],
                        'Modelo': p[11]
                    })
                
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True, height=400)
                
                # Descargar
                st.markdown("---")
                
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Descargar CSV",
                    data=csv,
                    file_name=f"predicciones_{st.session_state.user.username}.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            st.error(f"Error cargando historial: {e}")
    
    
    # PÁGINA: INFO DE MODELOS
    elif page == "ℹ️  Info de Modelos":
        st.title("ℹ️  Información de Modelos")
        st.markdown("---")
        
        try:
            versions = st.session_state.model_mgr.list_versions()
            
            st.subheader("Versiones Disponibles")
            
            for version in versions:
                with st.expander(f"📦 {version}"):
                    st.write(f"Versión: {version}")
                    
                    # TODO: Cargar metadata y mostrar
                    st.write("""
                    **Componentes:**
                    - Entity Classifier
                    - Issue Classifier
                    - Sentiment Analyzer
                    - Severity Scorer
                    
                    **Métricas:** (Por completar desde metadata.json)
                    """)
        
        except Exception as e:
            st.error(f"Error cargando información: {e}")

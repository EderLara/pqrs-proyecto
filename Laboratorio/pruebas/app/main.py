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
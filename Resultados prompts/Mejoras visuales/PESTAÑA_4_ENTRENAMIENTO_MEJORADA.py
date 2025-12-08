# PESTAÑA 4 MEJORADA - ENTRENAMIENTO Y EVALUACIÓN

"""
Pestaña reorganizada con:
- Layout mejorado
- Visualización de progreso
- Métricas destacadas
- Mejor organización de información
"""

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
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        version_name = st.text_input(
            "📝 Nombre de la versión",
            value=f"v_{datetime.now().strftime('%Y%m%d_%H%M')}",
            help="Nombre único para identificar esta versión del modelo"
        )
    
    with col2:
        train_btn = st.button("🚀 Entrenar", use_container_width=True, key="train_btn")
    
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
                    st.pyplot(fig_cm, use_container_width=True)
                
                # Expandible: Detalles por clase
                with st.expander("📋 Detalles por clase"):
                    entity_detail = pd.DataFrame(
                        entity_metrics
                    ).drop(columns=['accuracy', 'macro avg', 'weighted avg'], errors='ignore').T
                    
                    st.dataframe(
                        entity_detail.style.format("{:.2%}"),
                        use_container_width=True
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
                    st.pyplot(fig_cm, use_container_width=True)
                
                # Expandible: Detalles por clase
                with st.expander("📋 Detalles por clase"):
                    issue_detail = pd.DataFrame(
                        issue_metrics
                    ).drop(columns=['accuracy', 'macro avg', 'weighted avg'], errors='ignore').T
                    
                    st.dataframe(
                        issue_detail.style.format("{:.2%}"),
                        use_container_width=True
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
                
                st.plotly_chart(fig, use_container_width=True)
            
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
                st.dataframe(df_comparison, use_container_width=True, hide_index=True)
            
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

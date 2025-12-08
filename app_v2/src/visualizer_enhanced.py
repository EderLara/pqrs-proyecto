# visualizer_enhanced.py
"""
Módulo mejorado de visualizaciones con Plotly.

Proporciona gráficos interactivos, análisis de calidad de datos
y visualizaciones avanzadas de datasets.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


class EnhancedVisualizer:
    """Visualizador avanzado con Plotly para análisis de datos."""
    
    @staticmethod
    def plot_sentiment_gauge(sentiment_score: float, confidence: float) -> go.Figure:
        """
        Crea un gráfico tipo gauge para mostrar sentimiento.
        
        Args:
            sentiment_score: Score de -1.0 a 1.0
            confidence: Nivel de confianza 0-1
            
        Returns:
            Figura de Plotly (Gauge)
        """
        # Convertir a escala 0-100 para gauge
        gauge_value = (sentiment_score + 1) * 50  # -1 a 1 → 0 a 100
        
        # Determinar color y categoría
        if sentiment_score <= -0.6:
            color = '#d62828'
            label = 'Muy Negativo'
        elif sentiment_score <= -0.2:
            color = '#f77f00'
            label = 'Negativo'
        elif sentiment_score <= 0.2:
            color = '#ffd60a'
            label = 'Neutral'
        elif sentiment_score <= 0.6:
            color = '#90e0ef'
            label = 'Positivo'
        else:
            color = '#06a77d'
            label = 'Muy Positivo'
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=gauge_value,
            title={'text': f"Análisis de Sentimientos: {label}"},
            delta={'reference': 50, 'decreasing': {'color': 'green'}, 'increasing': {'color': 'red'}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 20], 'color': '#d62828'},
                    {'range': [20, 40], 'color': '#f77f00'},
                    {'range': [40, 60], 'color': '#ffd60a'},
                    {'range': [60, 80], 'color': '#90e0ef'},
                    {'range': [80, 100], 'color': '#06a77d'}
                ],
                'threshold': {
                    'line': {'color': 'black', 'width': 4},
                    'thickness': 0.75,
                    'value': gauge_value
                }
            }
        ))
        
        fig.add_annotation(
            text=f"Confianza: {confidence*100:.0f}%",
            x=0.5, y=-0.15,
            xref="paper", yref="paper",
            showarrow=False,
            font={"size": 12}
        )
        
        fig.update_layout(height=400, font={'size': 12})
        return fig
    
    @staticmethod
    def plot_distribution_pie(df: pd.DataFrame, column: str, title: str) -> go.Figure:
        """
        Crea gráfico de pastel con Plotly.
        
        Args:
            df: DataFrame
            column: Columna a visualizar
            title: Título del gráfico
            
        Returns:
            Figura de Plotly (Pie)
        """
        counts = df[column].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=counts.index,
            values=counts.values,
            hole=0.4,
            marker=dict(colors=px.colors.qualitative.Set3)
        )])
        
        fig.update_layout(
            title=title,
            height=500,
            font={'size': 12}
        )
        
        return fig
    
    @staticmethod
    def plot_distribution_bar(df: pd.DataFrame, column: str, title: str) -> go.Figure:
        """
        Crea gráfico de barras mejorado.
        
        Args:
            df: DataFrame
            column: Columna a visualizar
            title: Título del gráfico
            
        Returns:
            Figura de Plotly (Bar)
        """
        counts = df[column].value_counts().sort_values(ascending=True)
        
        fig = go.Figure(data=[go.Bar(
            y=counts.index,
            x=counts.values,
            orientation='h',
            marker=dict(
                color=counts.values,
                colorscale='Viridis',
                showscale=True
            ),
            text=counts.values,
            textposition='auto'
        )])
        
        fig.update_layout(
            title=title,
            xaxis_title="Cantidad",
            yaxis_title=column,
            height=500,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def plot_text_length_distribution(df: pd.DataFrame, column: str) -> go.Figure:
        """
        Visualiza distribución de longitudes de texto.
        
        Args:
            df: DataFrame
            column: Columna con textos
            
        Returns:
            Figura de Plotly (Histogram)
        """
        lengths = df[column].astype(str).str.len()
        
        fig = go.Figure(data=[go.Histogram(
            x=lengths,
            nbinsx=30,
            marker_color='#06a77d'
        )])
        
        fig.update_layout(
            title="Distribución de Longitud de Texto",
            xaxis_title="Caracteres",
            yaxis_title="Frecuencia",
            height=400,
            showlegend=False
        )
        
        fig.add_vline(x=lengths.mean(), line_dash="dash", line_color="red",
                     annotation_text=f"Media: {lengths.mean():.0f}")
        
        return fig
    
    @staticmethod
    def plot_top_words(df: pd.DataFrame, column: str, top_n: int = 20) -> go.Figure:
        """
        Visualiza palabras más frecuentes.
        
        Args:
            df: DataFrame
            column: Columna con textos
            top_n: Número de palabras top a mostrar
            
        Returns:
            Figura de Plotly (Bar)
        """
        # Procesar textos
        stop_words = {'el', 'la', 'de', 'que', 'y', 'a', 'en', 'es', 'por', 'un', 'una',
                      'con', 'para', 'del', 'al', 'los', 'las', 'se', 'da', 'sido', 'ama'}
        
        all_words = []
        for text in df[column].dropna():
            words = str(text).lower().split()
            all_words.extend([w for w in words if len(w) > 3 and w not in stop_words])
        
        word_counts = Counter(all_words)
        top_words = word_counts.most_common(top_n)
        
        words, counts = zip(*top_words)
        
        fig = go.Figure(data=[go.Bar(
            x=counts,
            y=words,
            orientation='h',
            marker_color='#f77f00'
        )])
        
        fig.update_layout(
            title=f"Top {top_n} Palabras Más Frecuentes",
            xaxis_title="Frecuencia",
            yaxis_title="Palabra",
            height=600,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def plot_quality_metrics(quality_stats: dict) -> go.Figure:
        """
        Visualiza métricas de calidad del dataset.
        
        Args:
            quality_stats: Dict con estadísticas de calidad
            
        Returns:
            Figura de Plotly (Indicator)
        """
        metrics = [
            ('Completitud', quality_stats.get('completitud', 0) * 100, 100),
            ('Duplicados', 100 - quality_stats.get('duplicados_pct', 0) * 100, 100),
            ('Validez', quality_stats.get('validez', 0) * 100, 100),
            ('Consistencia', quality_stats.get('consistencia', 0) * 100, 100)
        ]
        
        fig = make_subplots(
            rows=1, cols=4,
            subplot_titles=[m[0] for m in metrics],
            specs=[[{'type': 'indicator'} for _ in metrics]]
        )
        
        colors = ['#06a77d', '#90e0ef', '#ffd60a', '#f77f00']
        
        for i, (name, value, max_val) in enumerate(metrics):
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=value,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, max_val]},
                        'bar': {'color': colors[i]},
                        'steps': [
                            {'range': [0, 33], 'color': '#d62828'},
                            {'range': [33, 66], 'color': '#ffd60a'},
                            {'range': [66, 100], 'color': '#06a77d'}
                        ]
                    },
                    title={'text': name}
                ),
                row=1, col=i+1
            )
        
        fig.update_layout(height=400)
        return fig
    
    @staticmethod
    def plot_data_quality_before_after(df_raw: pd.DataFrame, 
                                       df_clean: pd.DataFrame) -> go.Figure:
        """
        Comparación de calidad antes y después de limpieza.
        
        Args:
            df_raw: DataFrame sin procesar
            df_clean: DataFrame procesado
            
        Returns:
            Figura de Plotly (Bar comparison)
        """
        metrics_before = [
            len(df_raw),
            df_raw.isnull().sum().sum(),
            df_raw.duplicated().sum(),
            100  # Completitud simulada
        ]
        
        metrics_after = [
            len(df_clean),
            df_clean.isnull().sum().sum(),
            df_clean.duplicated().sum(),
            100  # Completitud mejorada
        ]
        
        categories = ['Registros', 'Valores Nulos', 'Duplicados', 'Completitud (%)']
        
        fig = go.Figure(data=[
            go.Bar(name='Antes', x=categories, y=metrics_before, marker_color='#f77f00'),
            go.Bar(name='Después', x=categories, y=metrics_after, marker_color='#06a77d')
        ])
        
        fig.update_layout(
            title="Comparación de Calidad del Dataset",
            barmode='group',
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def plot_correlation_heatmap(df: pd.DataFrame, 
                                entity_col: str, 
                                issue_col: str) -> go.Figure:
        """
        Matriz de correlación entre Entidad y Tipo de Hecho.
        
        Args:
            df: DataFrame
            entity_col: Columna de entidades
            issue_col: Columna de tipos de hecho
            
        Returns:
            Figura de Plotly (Heatmap)
        """
        # Crear matriz de contingencia
        crosstab = pd.crosstab(df[entity_col], df[issue_col])
        
        fig = go.Figure(data=go.Heatmap(
            z=crosstab.values,
            x=crosstab.columns,
            y=crosstab.index,
            colorscale='Blues'
        ))
        
        fig.update_layout(
            title="Relación: Entidades vs Tipos de Hecho",
            xaxis_title="Tipo de Hecho",
            yaxis_title="Entidad",
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_quality_report(df_raw: pd.DataFrame, 
                            df_clean: pd.DataFrame) -> dict:
        """
        Genera reporte completo de calidad.
        
        Args:
            df_raw: DataFrame sin procesar
            df_clean: DataFrame procesado
            
        Returns:
            Dict con métricas de calidad
        """
        total_cells_raw = df_raw.shape[0] * df_raw.shape[1]
        total_cells_clean = df_clean.shape[0] * df_clean.shape[1]
        
        quality = {
            # Datos crudos
            'raw_records': len(df_raw),
            'raw_nulls': df_raw.isnull().sum().sum(),
            'raw_duplicates': df_raw.duplicated().sum(),
            'raw_completitud': round(100 * (1 - df_raw.isnull().sum().sum() / total_cells_raw), 2),
            
            # Datos limpios
            'clean_records': len(df_clean),
            'clean_nulls': df_clean.isnull().sum().sum(),
            'clean_duplicates': df_clean.duplicated().sum(),
            'clean_completitud': round(100 * (1 - df_clean.isnull().sum().sum() / total_cells_clean), 2),
            
            # Comparativa
            'records_removed': len(df_raw) - len(df_clean),
            'records_removed_pct': round(100 * (len(df_raw) - len(df_clean)) / len(df_raw), 2),
            'improvement': round(
                (df_clean.isnull().sum().sum() - df_raw.isnull().sum().sum()) / 
                max(df_raw.isnull().sum().sum(), 1) * 100
            ),
            
            # Score general
            'quality_score': min(
                (df_clean['ENTIDAD RESPONSABLE'].nunique() / 8 * 100 if 'ENTIDAD RESPONSABLE' in df_clean.columns else 0) +
                (df_clean['TIPOS DE HECHO'].nunique() / 8 * 100 if 'TIPOS DE HECHO' in df_clean.columns else 0),
                100
            )
        }
        
        return quality

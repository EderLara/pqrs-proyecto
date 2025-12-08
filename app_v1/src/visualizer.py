import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class Visualizer:
    """Clase utilitaria para generar gráficos."""

    @staticmethod
    def plot_distribution(df, column, title):
        """Grafica distribución de una variable categórica."""
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(y=column, data=df, order=df[column].value_counts().index, ax=ax)
        ax.set_title(title)
        return fig

    @staticmethod
    def plot_confusion_matrix(cm, labels, title):
        """Grafica matriz de confusión."""
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(title)
        ax.set_ylabel('Real')
        ax.set_xlabel('Predicho')
        return fig
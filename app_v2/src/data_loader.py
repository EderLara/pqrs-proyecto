import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

class DataPipeline:
    """Responsable de la carga, limpieza y transformación de datos."""

    def load_data(self, file_buffer):
        """Carga datos desde un archivo subido."""
        if file_buffer.name.endswith('.xlsx'):
            df = pd.read_excel(file_buffer)
        else:
            df = pd.read_csv(file_buffer)
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Realiza la limpieza básica de datos.
        
        Args:
            df: DataFrame crudo.
        Returns:
            DataFrame limpio.
        """
        # Limpieza básica basada en notebook 01
        df_clean = df.copy()
        
        # Unificar descripción
        if 'DESCRIPCION DEL HECHO' in df_clean.columns:
            df_clean['DESCRIPCION_LIMPIA'] = df_clean['DESCRIPCION DEL HECHO'].astype(str).str.lower().str.strip()
        
        # Filtrar columnas relevantes
        required_cols = ['ENTIDAD RESPONSABLE', 'TIPOS DE HECHO', 'DESCRIPCION_LIMPIA']
        df_clean = df_clean.dropna(subset=required_cols)
        
        # Filtrar clases minoritarias (mínimo 2 ejemplos)
        for col in ['ENTIDAD RESPONSABLE', 'TIPOS DE HECHO']:
            counts = df_clean[col].value_counts()
            valid = counts[counts >= 2].index
            df_clean = df_clean[df_clean[col].isin(valid)]
            
        return df_clean

    def get_features(self, df_clean):
        """
        Genera features TF-IDF.
        
        Returns:
            X, y_entity, y_issue, vectorizer
        """
        vectorizer = TfidfVectorizer(max_features=1000, stop_words=['el', 'la', 'de', 'que'])
        X = vectorizer.fit_transform(df_clean['DESCRIPCION_LIMPIA'])
        
        return X, df_clean['ENTIDAD RESPONSABLE'], df_clean['TIPOS DE HECHO'], vectorizer
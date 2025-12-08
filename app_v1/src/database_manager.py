import sqlite3
import pandas as pd
from datetime import datetime
from config import DB_PATH

class DatabaseManager:
    """Clase responsable de la interacción con la base de datos SQLite."""

    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.create_tables()

    def create_tables(self):
        """Inicializa las tablas de usuarios y predicciones."""
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password TEXT NOT NULL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                input_text TEXT,
                entity_pred TEXT,
                entity_conf REAL,
                issue_pred TEXT,
                issue_conf REAL,
                model_version TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()

    def add_user(self, username, password_hash):
        """Registra un nuevo usuario."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", 
                         (username, password_hash))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def get_user(self, username):
        """Recupera un usuario por su nombre."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
        return cursor.fetchone()

    def save_prediction(self, username, text, result, version):
        """Guarda el log de una predicción."""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO predictions 
            (username, input_text, entity_pred, entity_conf, issue_pred, issue_conf, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (username, text, result['entity'], result['entity_confidence'], 
              result['issue'], result['issue_confidence'], version))
        self.conn.commit()

    def get_predictions_history(self, order='DESC'):
        """Obtiene el historial de predicciones."""
        query = f"SELECT * FROM predictions ORDER BY timestamp {order}"
        return pd.read_sql_query(query, self.conn)
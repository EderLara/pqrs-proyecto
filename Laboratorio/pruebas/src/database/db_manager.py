"""
Database Manager para SQLite
"""
import sqlite3
import hashlib
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str = "pqrs_classifier.db"):
        """Inicializar manager de BD"""
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Inicializar tablas"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Tabla de usuarios
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Tabla de predicciones
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    pqrs_number INTEGER,
                    description TEXT NOT NULL,
                    entity TEXT NOT NULL,
                    entity_confidence REAL,
                    issue TEXT NOT NULL,
                    issue_confidence REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            
            conn.commit()
            logger.info("✓ Base de datos inicializada")
    
    def authenticate_user(self, username: str, password: str):
        """Autenticar usuario"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            pwd_hash = hashlib.sha256(password.encode()).hexdigest()
            
            cursor.execute(
                "SELECT id, username FROM users WHERE username=? AND password_hash=?",
                (username, pwd_hash)
            )
            result = cursor.fetchone()
            
            if result:
                return type('User', (), {'id': result[0], 'username': result[1]})()
            return None
    
    def create_user(self, username: str, email: str, password: str) -> tuple:
        """Crear nuevo usuario"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                pwd_hash = hashlib.sha256(password.encode()).hexdigest()
                
                cursor.execute(
                    "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                    (username, email, pwd_hash)
                )
                conn.commit()
                return True, "Usuario creado exitosamente"
        except sqlite3.IntegrityError:
            return False, "Usuario o email ya existe"
        except Exception as e:
            return False, str(e)
    
    def save_prediction(self, user_id: int, pqrs_number: int, description: str,
                       entity: str, entity_confidence: float,
                       issue: str, issue_confidence: float):
        """Guardar predicción en BD"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO predictions 
                (user_id, pqrs_number, description, entity, entity_confidence, 
                 issue, issue_confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (user_id, pqrs_number, description, entity, entity_confidence,
                  issue, issue_confidence))
            
            conn.commit()
            logger.info(f"✓ Predicción guardada: {pqrs_number}")
    
    def get_user_predictions(self, user_id: int) -> list:
        """Obtener predicciones del usuario"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute(
                """SELECT * FROM predictions WHERE user_id=? 
                   ORDER BY created_at DESC LIMIT 100""",
                (user_id,)
            )
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_user_stats(self, user_id: int) -> dict:
        """Obtener estadísticas del usuario"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT COUNT(*) FROM predictions WHERE user_id=?",
                (user_id,)
            )
            total = cursor.fetchone()[0]
            
            return {'total': total}
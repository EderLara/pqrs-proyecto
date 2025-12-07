# src/database/db_manager.py
"""
SQLite Database Manager for PQRS Classifier
Handles all database operations including user management and prediction storage
"""

import sqlite3
import logging
from datetime import datetime
from typing import List, Optional, Tuple
from pathlib import Path
import hashlib

from src.database.models import User, Prediction, DatabaseSchema

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages SQLite database operations.
    
    Handles:
    - Database initialization
    - User management (CRUD)
    - Prediction storage and retrieval
    - Authentication
    
    Attributes:
        db_path: Path to SQLite database file
        connection: SQLite connection object
    """
    
    def __init__(self, db_path: str):
        """
        Initialize DatabaseManager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.connection = None
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize database and create tables if they don't exist"""
        try:
            self.connection = sqlite3.connect(str(self.db_path), timeout=10.0)
            self.connection.row_factory = sqlite3.Row
            
            cursor = self.connection.cursor()
            
            # Create tables
            cursor.execute(DatabaseSchema.CREATE_USERS_TABLE)
            cursor.execute(DatabaseSchema.CREATE_PREDICTIONS_TABLE)
            cursor.execute(DatabaseSchema.CREATE_PREDICTIONS_INDEX)
            cursor.execute(DatabaseSchema.CREATE_PREDICTIONS_DATE_INDEX)
            cursor.execute(DatabaseSchema.CREATE_PREDICTIONS_MODEL_INDEX)
            
            self.connection.commit()
            logger.info(f"✓ Database initialized: {self.db_path}")
            
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {str(e)}")
            raise
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")
    
    @staticmethod
    def _hash_password(password: str) -> str:
        """
        Hash password using SHA256.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password
        """
        return hashlib.sha256(password.encode()).hexdigest()
    
    # ============= USER OPERATIONS =============
    
    def create_user(self, username: str, email: str, password: str) -> bool:
        """
        Create new user.
        
        Args:
            username: Username
            email: Email address
            password: Plain text password
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            sqlite3.IntegrityError: If username or email already exists
            
        Example:
            >>> db = DatabaseManager("pqrs.db")
            >>> db.create_user("john_doe", "john@example.com", "secure_password")
            True
        """
        try:
            cursor = self.connection.cursor()
            password_hash = self._hash_password(password)
            
            cursor.execute("""
                INSERT INTO users (username, email, password_hash, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
            """, (username, email, password_hash, datetime.utcnow(), datetime.utcnow()))
            
            self.connection.commit()
            logger.info(f"✓ User created: {username}")
            return True
            
        except sqlite3.IntegrityError as e:
            logger.error(f"User creation error (duplicate): {str(e)}")
            return False
        except sqlite3.Error as e:
            logger.error(f"User creation error: {str(e)}")
            return False
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """
        Authenticate user with username and password.
        
        Args:
            username: Username
            password: Plain text password
            
        Returns:
            User object if authentication successful, None otherwise
            
        Example:
            >>> db = DatabaseManager("pqrs.db")
            >>> user = db.authenticate_user("john_doe", "secure_password")
            >>> if user:
            ...     print(f"Welcome {user.username}")
        """
        try:
            cursor = self.connection.cursor()
            password_hash = self._hash_password(password)
            
            cursor.execute("""
                SELECT id, username, email, created_at, is_active
                FROM users
                WHERE username = ? AND password_hash = ? AND is_active = 1
            """, (username, password_hash))
            
            row = cursor.fetchone()
            
            if row:
                user = User(
                    id=row[0],
                    username=row[1],
                    email=row[2],
                    created_at=datetime.fromisoformat(row[3]) if row[3] else None,
                    is_active=bool(row[4])
                )
                logger.info(f"✓ User authenticated: {username}")
                return user
            else:
                logger.warning(f"Authentication failed: {username}")
                return None
                
        except sqlite3.Error as e:
            logger.error(f"Authentication error: {str(e)}")
            return None
    
    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """
        Get user by ID.
        
        Args:
            user_id: User ID
            
        Returns:
            User object or None if not found
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT id, username, email, created_at, updated_at, is_active
                FROM users
                WHERE id = ?
            """, (user_id,))
            
            row = cursor.fetchone()
            
            if row:
                return User(
                    id=row[0],
                    username=row[1],
                    email=row[2],
                    created_at=datetime.fromisoformat(row[3]) if row[3] else None,
                    updated_at=datetime.fromisoformat(row[4]) if row[4] else None,
                    is_active=bool(row[5])
                )
            return None
            
        except sqlite3.Error as e:
            logger.error(f"Error retrieving user: {str(e)}")
            return None
    
    # ============= PREDICTION OPERATIONS =============
    
    def save_prediction(self, prediction: Prediction) -> bool:
        """
        Save prediction to database.
        
        Args:
            prediction: Prediction object to save
            
        Returns:
            True if successful, False otherwise
            
        Example:
            >>> pred = Prediction(
            ...     user_id=1,
            ...     pqrs_number=1000,
            ...     description="FALTA PRESENCIA DEL INGENIERO...",
            ...     entity_predicted="Interventor",
            ...     entity_confidence=0.92,
            ...     issue_type_predicted="Ingeniería de la obra",
            ...     issue_confidence=0.87,
            ...     sentiment_predicted="Negativo",
            ...     sentiment_score=0.75,
            ...     severity_score=7.8,
            ...     severity_level="Importante",
            ... )
            >>> db.save_prediction(pred)
            True
        """
        try:
            cursor = self.connection.cursor()
            
            cursor.execute("""
                INSERT INTO predictions (
                    user_id, pqrs_number, description,
                    entity_predicted, entity_confidence,
                    issue_type_predicted, issue_confidence,
                    sentiment_predicted, sentiment_score,
                    severity_score, severity_level,
                    model_version, processing_time_ms,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                prediction.user_id,
                prediction.pqrs_number,
                prediction.description,
                prediction.entity_predicted,
                prediction.entity_confidence,
                prediction.issue_type_predicted,
                prediction.issue_confidence,
                prediction.sentiment_predicted,
                prediction.sentiment_score,
                prediction.severity_score,
                prediction.severity_level,
                prediction.model_version,
                prediction.processing_time_ms,
                prediction.created_at,
                prediction.updated_at
            ))
            
            self.connection.commit()
            logger.info(f"✓ Prediction saved for user {prediction.user_id}")
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Error saving prediction: {str(e)}")
            return False
    
    def get_user_predictions(
        self, 
        user_id: int, 
        limit: int = 100,
        order_by: str = "DESC"
    ) -> List[Prediction]:
        """
        Get all predictions for a user.
        
        Args:
            user_id: User ID
            limit: Maximum number of records to return
            order_by: Sort order (ASC or DESC)
            
        Returns:
            List of Prediction objects
            
        Example:
            >>> db = DatabaseManager("pqrs.db")
            >>> predictions = db.get_user_predictions(user_id=1, limit=50)
            >>> for pred in predictions:
            ...     print(f"{pred.entity_predicted}: {pred.severity_level}")
        """
        try:
            order_by = "DESC" if order_by.upper() == "DESC" else "ASC"
            
            cursor = self.connection.cursor()
            cursor.execute(f"""
                SELECT 
                    id, user_id, pqrs_number, description,
                    entity_predicted, entity_confidence,
                    issue_type_predicted, issue_confidence,
                    sentiment_predicted, sentiment_score,
                    severity_score, severity_level,
                    model_version, processing_time_ms,
                    created_at, updated_at
                FROM predictions
                WHERE user_id = ?
                ORDER BY created_at {order_by}
                LIMIT ?
            """, (user_id, limit))
            
            rows = cursor.fetchall()
            predictions = []
            
            for row in rows:
                pred = Prediction(
                    id=row[0],
                    user_id=row[1],
                    pqrs_number=row[2],
                    description=row[3],
                    entity_predicted=row[4],
                    entity_confidence=row[5],
                    issue_type_predicted=row[6],
                    issue_confidence=row[7],
                    sentiment_predicted=row[8],
                    sentiment_score=row[9],
                    severity_score=row[10],
                    severity_level=row[11],
                    model_version=row[12],
                    processing_time_ms=row[13],
                    created_at=datetime.fromisoformat(row[14]) if row[14] else None,
                    updated_at=datetime.fromisoformat(row[15]) if row[15] else None
                )
                predictions.append(pred)
            
            logger.info(f"✓ Retrieved {len(predictions)} predictions for user {user_id}")
            return predictions
            
        except sqlite3.Error as e:
            logger.error(f"Error retrieving predictions: {str(e)}")
            return []
    
    def get_statistics(self, user_id: int) -> dict:
        """
        Get statistics for user's predictions.
        
        Args:
            user_id: User ID
            
        Returns:
            Dictionary with statistics
            
        Example:
            >>> stats = db.get_statistics(user_id=1)
            >>> print(f"Total predictions: {stats['total']}")
            >>> print(f"Average severity: {stats['avg_severity']:.2f}")
        """
        try:
            cursor = self.connection.cursor()
            
            # Total predictions
            cursor.execute(
                "SELECT COUNT(*) FROM predictions WHERE user_id = ?",
                (user_id,)
            )
            total = cursor.fetchone()[0]
            
            # Average severity
            cursor.execute(
                "SELECT AVG(severity_score) FROM predictions WHERE user_id = ?",
                (user_id,)
            )
            avg_severity = cursor.fetchone()[0] or 0.0
            
            # Severity distribution
            cursor.execute("""
                SELECT severity_level, COUNT(*) as count
                FROM predictions
                WHERE user_id = ?
                GROUP BY severity_level
            """, (user_id,))
            
            severity_dist = dict(cursor.fetchall())
            
            # Model version distribution
            cursor.execute("""
                SELECT model_version, COUNT(*) as count
                FROM predictions
                WHERE user_id = ?
                GROUP BY model_version
            """, (user_id,))
            
            model_dist = dict(cursor.fetchall())
            
            return {
                "total": total,
                "avg_severity": round(avg_severity, 2),
                "severity_distribution": severity_dist,
                "model_versions": model_dist
            }
            
        except sqlite3.Error as e:
            logger.error(f"Error computing statistics: {str(e)}")
            return {}

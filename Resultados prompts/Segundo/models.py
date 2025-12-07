# src/database/models.py
"""
SQLite database models and schemas for PQRS Classifier
Defines User and Prediction models
"""

from datetime import datetime
from typing import Optional
import sqlite3
from dataclasses import dataclass, asdict


@dataclass
class User:
    """User model for authentication"""
    id: Optional[int] = None
    username: str = ""
    email: str = ""
    password_hash: str = ""
    created_at: datetime = None
    updated_at: datetime = None
    is_active: bool = True
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()


@dataclass
class Prediction:
    """Prediction model for storing classification results"""
    id: Optional[int] = None
    user_id: int = 0
    pqrs_number: int = 0
    description: str = ""
    
    # Predictions
    entity_predicted: str = ""
    entity_confidence: float = 0.0
    issue_type_predicted: str = ""
    issue_confidence: float = 0.0
    sentiment_predicted: str = ""
    sentiment_score: float = 0.0
    severity_score: float = 0.0
    severity_level: str = ""
    
    # Metadata
    model_version: str = "v1"
    processing_time_ms: float = 0.0
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()


class DatabaseSchema:
    """Database schema definitions"""
    
    CREATE_USERS_TABLE = """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        is_active BOOLEAN DEFAULT 1
    )
    """
    
    CREATE_PREDICTIONS_TABLE = """
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        pqrs_number INTEGER,
        description TEXT NOT NULL,
        
        entity_predicted TEXT NOT NULL,
        entity_confidence REAL NOT NULL,
        issue_type_predicted TEXT NOT NULL,
        issue_confidence REAL NOT NULL,
        sentiment_predicted TEXT NOT NULL,
        sentiment_score REAL NOT NULL,
        severity_score REAL NOT NULL,
        severity_level TEXT NOT NULL,
        
        model_version TEXT DEFAULT 'v1',
        processing_time_ms REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    )
    """
    
    CREATE_PREDICTIONS_INDEX = """
    CREATE INDEX IF NOT EXISTS idx_predictions_user_id 
    ON predictions(user_id)
    """
    
    CREATE_PREDICTIONS_DATE_INDEX = """
    CREATE INDEX IF NOT EXISTS idx_predictions_created_at 
    ON predictions(created_at)
    """
    
    CREATE_PREDICTIONS_MODEL_INDEX = """
    CREATE INDEX IF NOT EXISTS idx_predictions_model_version 
    ON predictions(model_version)
    """

# src/utils/config.py
"""
Configuration and constants for PQRS Classifier
"""

from pathlib import Path
from typing import Dict

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"
MODELS_PATH = PROJECT_ROOT / "models"
DATABASE_PATH = PROJECT_ROOT / "pqrs_classifier.db"

# Model versions
MODEL_VERSIONS = {
    "v1": {
        "entity_model": MODELS_PATH / "v1" / "entity_classifier.pkl",
        "issue_model": MODELS_PATH / "v1" / "issue_classifier.pkl",
        "sentiment_model": MODELS_PATH / "v1" / "sentiment_analyzer.pkl",
        "vectorizer": MODELS_PATH / "v1" / "vectorizer.pkl",
    }
}

# Class definitions
ENTITY_CLASSES = {
    0: "Interventor",
    1: "Contratista",
    2: "Municipio",
    3: "SIF",
    4: "Otra"
}

ISSUE_CLASSES = {
    0: "Ingeniería de la obra",
    1: "Movilidad",
    2: "Seguridad",
    3: "Social",
    4: "Ambiental",
    5: "Económico"
}

SENTIMENT_CLASSES = {
    0: "Muy Negativo",
    1: "Negativo",
    2: "Neutral",
    3: "Positivo"
}

SEVERITY_LEVELS = {
    "RED": (8.0, 10.0, "Urgente"),
    "YELLOW": (5.0, 7.9, "Importante"),
    "GREEN": (0.0, 4.9, "Rutinario")
}

# Text preprocessing
MAX_TEXT_LENGTH = 5000
MIN_TEXT_LENGTH = 20

# Severity scoring weights
SEVERITY_WEIGHTS = {
    "sentiment": 0.30,
    "keywords": 0.25,
    "state": 0.20,
    "time": 0.15,
    "impact": 0.10
}

# Critical keywords for severity detection
CRITICAL_KEYWORDS = {
    "risk": ["riesgo", "peligro", "crítico", "emergencia", "urgente"],
    "accident": ["accidente", "volcamiento", "derrumbe", "colapso", "caída"],
    "damage": ["daño", "falla", "ruptura", "rotura", "fisura"],
    "incomplete": ["falta", "ausencia", "sin", "incompleto", "pendiente"],
}

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "[%(asctime)s] %(name)s - %(levelname)s - %(message)s"

# Database
DATABASE_TIMEOUT = 10
DATABASE_CHECK_INTERVAL = 3600  # seconds

# Streamlit app
APP_TITLE = "PQRS Intelligent Classifier"
APP_ICON = "🔍"
APP_LAYOUT = "wide"

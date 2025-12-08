# tests/conftest.py
"""
Pytest configuration and fixtures for PQRS Classifier tests
"""

import pytest
import tempfile
from pathlib import Path
import pandas as pd
from datetime import datetime

# Sample test data
SAMPLE_PQRS_DATA = {
    "PQRS No.": [1000, 1001, 1002, 1003, 1004],
    "DESCRIPCION DEL HECHO": [
        "FALTA PRESENCIA DEL INGENIERO CIVIL DE LA INTERVENTORIA PARA REALIZAR CONTROL A LAS OBRAS",
        "DURANTE EL RECORRIDO DE CONTROL SOCIAL SE REPORTARON FALENCIAS EN OBRAS DE INGENIERIA",
        "EN EL PUENTE LA ARENOSA SE LLEVA A CABO UNA INTERVENCION DEL TALUD",
        "SE SOLICITA IMPLEMENTACION DE UN FRENTE DE TRABAJO PARA CAMBIO DE TUBERIA",
        "SE PROPONE POR PARTE DE LA COMUNIDAD LA REALIZACION DE UN CONVITE PARA MANTENIMIENTO DE VIA"
    ],
    "ENTIDAD RESPONSABLE": [
        "Interventor",
        "Contratista",
        "Contratista",
        "Municipio",
        "SIF"
    ],
    "TIPOS DE HECHO": [
        "Ingeniería de la obra",
        "Ingeniería de la obra",
        "Ingeniería de la obra",
        "Movilidad",
        "Social"
    ],
    "ESTADO": [
        "Resuelto",
        "Resuelto",
        "En trámite",
        "Resuelto",
        "Resuelto"
    ]
}


@pytest.fixture
def sample_dataframe():
    """Create sample PQRS DataFrame for testing"""
    return pd.DataFrame(SAMPLE_PQRS_DATA)


@pytest.fixture
def temp_database():
    """Create temporary database for testing"""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_pqrs.db"
    yield str(db_path)
    # Cleanup
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def temp_data_dir():
    """Create temporary directory for test data"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_csv_file(temp_data_dir, sample_dataframe):
    """Create temporary CSV file with sample PQRS data"""
    csv_path = temp_data_dir / "pqrs_sample.csv"
    sample_dataframe.to_csv(csv_path, index=False)
    return str(csv_path)

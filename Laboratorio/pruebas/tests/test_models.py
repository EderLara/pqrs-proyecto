"""
Tests para validar modelos antes de despliegue
"""
import pytest
from src.models.model_manager import ModelManager

class TestModelManager:
    @pytest.fixture
    def model_mgr(self):
        return ModelManager("models/v1")
    
    def test_models_loaded(self, model_mgr):
        """Verificar que modelos se cargan correctamente"""
        assert model_mgr.entity_model is not None
        assert model_mgr.issue_model is not None
        assert model_mgr.vectorizer is not None
    
    def test_prediction_output_format(self, model_mgr):
        """Verificar formato de salida de predicción"""
        result = model_mgr.predict("FALTA PRESENCIA DEL INGENIERO")
        
        assert 'entity' in result
        assert 'entity_confidence' in result
        assert 'issue' in result
        assert 'issue_confidence' in result
        
        assert 0 <= result['entity_confidence'] <= 1
        assert 0 <= result['issue_confidence'] <= 1
    
    def test_prediction_with_empty_text(self, model_mgr):
        """Verificar manejo de texto vacío"""
        result = model_mgr.predict("")
        assert result is not None
    
    def test_prediction_with_long_text(self, model_mgr):
        """Verificar manejo de texto largo"""
        long_text = "FALTA PRESENCIA DEL INGENIERO " * 50
        result = model_mgr.predict(long_text)
        assert result is not None
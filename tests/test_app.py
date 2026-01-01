"""
Tests para el servicio LLM de LiquidAI.
"""

import pytest
from unittest.mock import MagicMock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.src.config.settings import AEPConfig, AEPModelConfig


class TestAEPModelConfig:
    """Tests para la configuración del modelo."""

    def test_default_values(self) -> None:
        """Verifica valores por defecto de la configuración del modelo."""
        config = AEPModelConfig()

        assert config.model_id == "LiquidAI/LFM2-2.6B"
        assert config.device_map == "auto"
        assert config.torch_dtype == "bfloat16"
        assert config.max_new_tokens == 512
        assert config.temperature == 0.3
        assert config.min_p == 0.15
        assert config.repetition_penalty == 1.05
        assert config.use_flash_attention is False

    def test_to_dict(self) -> None:
        """Verifica conversión a diccionario."""
        config = AEPModelConfig()
        result = config.to_dict()

        assert isinstance(result, dict)
        assert "model_id" in result
        assert "temperature" in result
        assert result["model_id"] == "LiquidAI/LFM2-2.6B"


class TestAEPConfig:
    """Tests para la configuración de la aplicación."""

    def test_default_values(self) -> None:
        """Verifica valores por defecto de la configuración."""
        config = AEPConfig()

        assert config.app_name == "LiquidAI Chat"
        assert config.host == "0.0.0.0"
        assert config.port == 5049
        assert config.debug is False
        assert isinstance(config.model_config, AEPModelConfig)

    def test_from_env(self) -> None:
        """Verifica carga de configuración desde entorno."""
        with patch.dict(os.environ, {
            "AEP_HOST": "127.0.0.1",
            "AEP_PORT": "8080",
            "AEP_DEBUG": "true",
        }):
            config = AEPConfig.from_env()

            assert config.host == "127.0.0.1"
            assert config.port == 8080
            assert config.debug is True

    def test_to_dict(self) -> None:
        """Verifica conversión a diccionario."""
        config = AEPConfig()
        result = config.to_dict()

        assert isinstance(result, dict)
        assert "app_name" in result
        assert "model_config" in result
        assert isinstance(result["model_config"], dict)


class TestAEPLLMService:
    """Tests para el servicio LLM (mock)."""

    @patch('app.src.models.llm_service.AutoModelForCausalLM')
    @patch('app.src.models.llm_service.AutoTokenizer')
    def test_singleton_pattern(
        self,
        mock_tokenizer: MagicMock,
        mock_model: MagicMock
    ) -> None:
        """Verifica que el servicio sigue el patrón singleton."""
        # Reset singleton for testing
        from app.src.models.llm_service import AEPLLMService
        AEPLLMService._instance = None

        service1 = AEPLLMService()
        service2 = AEPLLMService()

        assert service1 is service2

    def test_get_model_info_not_loaded(self) -> None:
        """Verifica info del modelo cuando no está cargado."""
        from app.src.models.llm_service import AEPLLMService
        AEPLLMService._instance = None

        service = AEPLLMService()
        info = service.get_model_info()

        assert info["is_loaded"] is False
        assert "model_id" in info


class TestAPIEndpoints:
    """Tests para los endpoints de la API."""

    @pytest.fixture
    def client(self):
        """Crea cliente de pruebas Flask."""
        from app.src.app import create_app

        app = create_app()
        app.config["TESTING"] = True

        with app.test_client() as client:
            yield client

    def test_health_endpoint(self, client) -> None:
        """Verifica endpoint de health check."""
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.get_json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_model_info_endpoint(self, client) -> None:
        """Verifica endpoint de info del modelo."""
        response = client.get("/api/v1/model/info")

        assert response.status_code == 200
        data = response.get_json()
        assert "success" in data
        assert "data" in data

    def test_chat_completions_without_model(self, client) -> None:
        """Verifica error cuando el modelo no está cargado."""
        response = client.post(
            "/api/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hola"}]
            }
        )

        assert response.status_code == 503
        data = response.get_json()
        assert "error" in data

    def test_chat_completions_missing_messages(self, client) -> None:
        """Verifica error cuando faltan mensajes."""
        response = client.post(
            "/api/v1/chat/completions",
            json={}
        )

        # Puede ser 400 o 503 dependiendo de si el modelo está cargado
        assert response.status_code in [400, 503]

    def test_generate_without_model(self, client) -> None:
        """Verifica error en generate cuando el modelo no está cargado."""
        response = client.post(
            "/api/v1/generate",
            json={"prompt": "Hola"}
        )

        assert response.status_code == 503

    def test_generate_missing_prompt(self, client) -> None:
        """Verifica error cuando falta el prompt."""
        response = client.post(
            "/api/v1/generate",
            json={}
        )

        # Puede ser 400 o 503 dependiendo de si el modelo está cargado
        assert response.status_code in [400, 503]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Configuración centralizada de la aplicación AEP LiquidAI.

Este módulo contiene todas las constantes y configuraciones necesarias
para el funcionamiento de la aplicación.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# Constantes Globales
# =============================================================================

AEP_MODEL_ID = "LiquidAI/LFM2-2.6B"
AEP_MODEL_CACHE_DIR = os.environ.get("HF_HOME", os.environ.get("AEP_MODEL_CACHE_DIR", "./models"))
AEP_DEFAULT_MAX_TOKENS = 512
AEP_DEFAULT_TEMPERATURE = 0.3
AEP_DEFAULT_MIN_P = 0.15
AEP_DEFAULT_REPETITION_PENALTY = 1.05
AEP_MAX_CONTEXT_LENGTH = 32768
AEP_API_VERSION = "v1"
AEP_APP_NAME = "LiquidAI Chat"
AEP_APP_DESCRIPTION = "Interfaz de chat para el modelo LiquidAI LFM2-2.6B"


@dataclass
class AEPModelConfig:
    """
    Configuración del modelo LiquidAI LFM2-2.6B.

    Attributes:
        model_id: Identificador del modelo en HuggingFace.
        cache_dir: Directorio donde se almacena el modelo descargado.
        device_map: Configuración de mapeo de dispositivos (auto/cpu/cuda).
        torch_dtype: Tipo de datos de PyTorch (bfloat16 recomendado).
        max_new_tokens: Número máximo de tokens a generar.
        temperature: Temperatura para la generación (0.3 recomendado).
        min_p: Probabilidad mínima para sampling.
        repetition_penalty: Penalización por repetición.
        use_flash_attention: Si usar Flash Attention 2 (requiere GPU compatible).
    """

    model_id: str = AEP_MODEL_ID
    cache_dir: str = field(default_factory=lambda: AEP_MODEL_CACHE_DIR)
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"
    max_new_tokens: int = AEP_DEFAULT_MAX_TOKENS
    temperature: float = AEP_DEFAULT_TEMPERATURE
    min_p: float = AEP_DEFAULT_MIN_P
    repetition_penalty: float = AEP_DEFAULT_REPETITION_PENALTY
    use_flash_attention: bool = False

    def to_dict(self) -> dict:
        """
        Convierte la configuración a un diccionario.

        Returns:
            Diccionario con la configuración del modelo.
        """
        return {
            "model_id": self.model_id,
            "cache_dir": self.cache_dir,
            "device_map": self.device_map,
            "torch_dtype": self.torch_dtype,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "min_p": self.min_p,
            "repetition_penalty": self.repetition_penalty,
            "use_flash_attention": self.use_flash_attention,
        }


@dataclass
class AEPConfig:
    """
    Configuración general de la aplicación.

    Attributes:
        app_name: Nombre de la aplicación.
        debug: Modo debug de Flask.
        host: Host donde se ejecuta la aplicación.
        port: Puerto de la aplicación.
        secret_key: Clave secreta para Flask.
        api_key: Clave API para autenticación (opcional).
        model_config: Configuración del modelo.
        cors_origins: Orígenes permitidos para CORS.
    """

    app_name: str = AEP_APP_NAME
    debug: bool = field(
        default_factory=lambda: os.environ.get("AEP_DEBUG", "false").lower() == "true"
    )
    host: str = field(default_factory=lambda: os.environ.get("AEP_HOST", "0.0.0.0"))
    port: int = field(
        default_factory=lambda: int(os.environ.get("AEP_PORT", "5049"))
    )
    secret_key: str = field(
        default_factory=lambda: os.environ.get(
            "AEP_SECRET_KEY", "aep-liquidai-secret-key-change-in-production"
        )
    )
    api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("AEP_API_KEY")
    )
    model_config: AEPModelConfig = field(default_factory=AEPModelConfig)
    cors_origins: list[str] = field(default_factory=lambda: ["*"])

    @classmethod
    def from_env(cls) -> "AEPConfig":
        """
        Crea una configuración desde variables de entorno.

        Returns:
            Instancia de AEPConfig con valores del entorno.
        """
        model_config = AEPModelConfig(
            cache_dir=os.environ.get("AEP_MODEL_CACHE_DIR", AEP_MODEL_CACHE_DIR),
            device_map=os.environ.get("AEP_DEVICE_MAP", "auto"),
            max_new_tokens=int(
                os.environ.get("AEP_MAX_TOKENS", str(AEP_DEFAULT_MAX_TOKENS))
            ),
            temperature=float(
                os.environ.get("AEP_TEMPERATURE", str(AEP_DEFAULT_TEMPERATURE))
            ),
            min_p=float(os.environ.get("AEP_MIN_P", str(AEP_DEFAULT_MIN_P))),
            repetition_penalty=float(
                os.environ.get(
                    "AEP_REPETITION_PENALTY", str(AEP_DEFAULT_REPETITION_PENALTY)
                )
            ),
            use_flash_attention=os.environ.get(
                "AEP_USE_FLASH_ATTENTION", "false"
            ).lower() == "true",
        )
        return cls(model_config=model_config)

    def to_dict(self) -> dict:
        """
        Convierte la configuración a un diccionario.

        Returns:
            Diccionario con la configuración de la aplicación.
        """
        return {
            "app_name": self.app_name,
            "debug": self.debug,
            "host": self.host,
            "port": self.port,
            "api_key_set": self.api_key is not None,
            "model_config": self.model_config.to_dict(),
            "cors_origins": self.cors_origins,
        }

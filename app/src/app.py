"""
Aplicación principal Flask para LiquidAI LFM2-2.6B Chat.

Este módulo crea y configura la aplicación Flask con todos los
blueprints y extensiones necesarias.
"""

import logging
import os
from typing import Optional

from flask import Flask, render_template
from flask_cors import CORS

from .api.routes import api_bp
from .config.settings import AEPConfig
from .models.llm_service import AEPLLMService

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_app(config: Optional[AEPConfig] = None) -> Flask:
    """
    Factory function para crear la aplicación Flask.

    Args:
        config: Configuración de la aplicación (opcional).

    Returns:
        Instancia configurada de Flask.
    """
    # Crear instancia de Flask
    app = Flask(
        __name__,
        template_folder="../templates",
        static_folder="../static",
    )

    # Cargar configuración
    if config is None:
        config = AEPConfig.from_env()

    # Configurar Flask
    app.config["SECRET_KEY"] = config.secret_key
    app.config["AEP_API_KEY"] = config.api_key
    app.config["DEBUG"] = config.debug

    # Configurar CORS
    CORS(app, origins=config.cors_origins)

    # Registrar blueprints
    app.register_blueprint(api_bp)

    # Inicializar servicio LLM
    llm_service = AEPLLMService(config.model_config)

    # Almacenar configuración en app
    app.config["AEP_CONFIG"] = config

    # Rutas principales
    @app.route("/")
    def index():
        """Página principal del chat."""
        return render_template("index.html", config=config)

    @app.route("/docs")
    def api_docs():
        """Documentación de la API."""
        return render_template("api_docs.html", config=config)

    # Cargar modelo al inicio si está configurado
    if os.environ.get("AEP_LOAD_MODEL_ON_STARTUP", "false").lower() == "true":
        logger.info("Cargando modelo al inicio...")
        try:
            llm_service.load_model()
        except Exception as e:
            logger.error("Error al cargar modelo al inicio: %s", str(e))

    logger.info("Aplicación Flask creada exitosamente")
    logger.info("Configuración: %s", config.to_dict())

    return app


# Instancia de la aplicación para Gunicorn
app = create_app()


if __name__ == "__main__":
    config = AEPConfig.from_env()
    application = create_app(config)
    application.run(
        host=config.host,
        port=config.port,
        debug=config.debug,
    )

"""
Rutas de la API REST para el chat con LiquidAI LFM2-2.6B.

Este módulo define todos los endpoints de la API para interactuar
con el modelo de lenguaje.
"""

import logging
import threading
import time
from functools import wraps
from typing import Any, Callable, Optional

from flask import Blueprint, Response, current_app, jsonify, request, stream_with_context

from ..config.settings import AEP_API_VERSION
from ..models.llm_service import AEPLLMService

# Configuración de logging
logger = logging.getLogger(__name__)

# Blueprint de la API
api_bp = Blueprint("api", __name__, url_prefix=f"/api/{AEP_API_VERSION}")


def require_api_key(func: Callable) -> Callable:
    """
    Decorador para requerir autenticación por API key.

    Args:
        func: Función a decorar.

    Returns:
        Función decorada con validación de API key.
    """
    @wraps(func)
    def decorated_function(*args: Any, **kwargs: Any) -> Any:
        api_key = current_app.config.get("AEP_API_KEY")

        # Si no hay API key configurada, permitir acceso
        if not api_key:
            return func(*args, **kwargs)

        # Verificar API key en headers
        provided_key = request.headers.get("X-API-Key") or request.headers.get(
            "Authorization", ""
        ).replace("Bearer ", "")

        if provided_key != api_key:
            return jsonify({
                "error": "Unauthorized",
                "message": "API key inválida o no proporcionada",
            }), 401

        return func(*args, **kwargs)

    return decorated_function


def get_llm_service() -> AEPLLMService:
    """
    Obtiene la instancia del servicio LLM.

    Returns:
        Instancia singleton del servicio LLM.
    """
    return AEPLLMService()


@api_bp.route("/health", methods=["GET"])
def health_check() -> tuple[Response, int]:
    """
    Endpoint de health check.

    Returns:
        Estado de salud de la API y el modelo.
    """
    service = get_llm_service()

    return jsonify({
        "status": "healthy",
        "model_loaded": service.is_loaded,
        "timestamp": time.time(),
    }), 200


@api_bp.route("/model/info", methods=["GET"])
@require_api_key
def model_info() -> tuple[Response, int]:
    """
    Obtiene información del modelo.

    Returns:
        Información detallada del modelo cargado.
    """
    service = get_llm_service()

    return jsonify({
        "success": True,
        "data": service.get_model_info(),
    }), 200


@api_bp.route("/model/load", methods=["POST"])
@require_api_key
def load_model() -> tuple[Response, int]:
    """
    Carga el modelo en memoria de forma asíncrona.

    Returns:
        Estado de la carga del modelo.
    """
    service = get_llm_service()

    if service.is_loaded:
        return jsonify({
            "success": True,
            "message": "El modelo ya está cargado",
        }), 200

    if getattr(service, 'is_loading', False):
        return jsonify({
            "success": True,
            "message": "La carga del modelo está en progreso",
        }), 200

    # Iniciar carga en segundo plano
    service.is_loading = True

    def load_worker():
        try:
            logger.info("Iniciando carga del modelo en thread separado")
            service.load_model()
            logger.info("Modelo cargado exitosamente")
        except Exception as e:
            logger.error("Error al cargar el modelo: %s", str(e))
        finally:
            service.is_loading = False

    thread = threading.Thread(target=load_worker, daemon=True)
    thread.start()

    return jsonify({
        "success": True,
        "message": "Carga del modelo iniciada en segundo plano",
    }), 200


@api_bp.route("/model/unload", methods=["POST"])
@require_api_key
def unload_model() -> tuple[Response, int]:
    """
    Descarga el modelo de memoria.

    Returns:
        Estado de la descarga del modelo.
    """
    service = get_llm_service()

    try:
        service.unload_model()
        return jsonify({
            "success": True,
            "message": "Modelo descargado de memoria",
        }), 200

    except Exception as e:
        logger.error("Error al descargar el modelo: %s", str(e))
        return jsonify({
            "success": False,
            "error": str(e),
        }), 500


@api_bp.route("/model/status", methods=["GET"])
def get_model_status() -> tuple[Response, int]:
    """
    Obtiene el estado de carga del modelo.

    Returns:
        Estado del modelo: loaded, loading, not_loaded
    """
    service = get_llm_service()

    if service.is_loaded:
        return jsonify({
            "status": "loaded",
            "message": "El modelo está cargado y listo para usar"
        }), 200
    elif getattr(service, 'is_loading', False):
        return jsonify({
            "status": "loading",
            "message": "El modelo se está cargando"
        }), 200
    else:
        return jsonify({
            "status": "not_loaded",
            "message": "El modelo no está cargado"
        }), 200


@api_bp.route("/chat/completions", methods=["POST"])
@require_api_key
def chat_completions() -> tuple[Response, int]:
    """
    Genera una respuesta de chat (formato compatible con OpenAI).

    Request Body:
        messages: Lista de mensajes del chat.
        max_tokens: Número máximo de tokens (opcional).
        temperature: Temperatura de generación (opcional).
        stream: Si usar streaming (opcional).
        system: Prompt del sistema (opcional).

    Returns:
        Respuesta generada por el modelo.
    """
    service = get_llm_service()

    if not service.is_loaded:
        return jsonify({
            "error": "Model not loaded",
            "message": "El modelo no está cargado. Use POST /api/v1/model/load",
        }), 503

    # Obtener datos del request
    data = request.get_json()

    if not data:
        return jsonify({
            "error": "Bad Request",
            "message": "Se requiere un cuerpo JSON válido",
        }), 400

    messages = data.get("messages", [])
    if not messages:
        return jsonify({
            "error": "Bad Request",
            "message": "Se requiere al menos un mensaje",
        }), 400

    # Validar formato de mensajes
    for msg in messages:
        if "role" not in msg or "content" not in msg:
            return jsonify({
                "error": "Bad Request",
                "message": "Cada mensaje debe tener 'role' y 'content'",
            }), 400

    # Parámetros opcionales
    max_tokens = data.get("max_tokens")
    temperature = data.get("temperature")
    min_p = data.get("min_p")
    repetition_penalty = data.get("repetition_penalty")
    system_prompt = data.get("system")
    stream = data.get("stream", False)

    try:
        if stream:
            return _stream_response(
                service,
                messages,
                max_tokens,
                temperature,
                min_p,
                repetition_penalty,
                system_prompt,
            )

        start_time = time.time()
        response_text = service.generate(
            messages=messages,
            max_new_tokens=max_tokens,
            temperature=temperature,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            system_prompt=system_prompt,
        )
        generation_time = time.time() - start_time

        return jsonify({
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": service.config.model_id,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text,
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "generation_time_seconds": round(generation_time, 2),
            },
        }), 200

    except Exception as e:
        logger.error("Error en generación: %s", str(e))
        return jsonify({
            "error": "Internal Server Error",
            "message": str(e),
        }), 500


def _stream_response(
    service: AEPLLMService,
    messages: list[dict[str, str]],
    max_tokens: Optional[int],
    temperature: Optional[float],
    min_p: Optional[float],
    repetition_penalty: Optional[float],
    system_prompt: Optional[str],
) -> Response:
    """
    Genera una respuesta en streaming SSE.

    Args:
        service: Servicio LLM.
        messages: Lista de mensajes.
        max_tokens: Máximo de tokens.
        temperature: Temperatura.
        min_p: Min P.
        repetition_penalty: Penalización de repetición.
        system_prompt: Prompt del sistema.

    Returns:
        Response con streaming SSE.
    """
    def generate():
        try:
            for chunk in service.generate_stream(
                messages=messages,
                max_new_tokens=max_tokens,
                temperature=temperature,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                system_prompt=system_prompt,
            ):
                data = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": service.config.model_id,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "content": chunk,
                        },
                        "finish_reason": None,
                    }],
                }
                yield f"data: {jsonify(data).get_data(as_text=True)}\n\n"

            # Enviar mensaje de finalización
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error("Error en streaming: %s", str(e))
            error_data = {"error": str(e)}
            yield f"data: {jsonify(error_data).get_data(as_text=True)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@api_bp.route("/generate", methods=["POST"])
@require_api_key
def generate_text() -> tuple[Response, int]:
    """
    Endpoint simplificado para generación de texto.

    Request Body:
        prompt: Texto del prompt.
        max_tokens: Número máximo de tokens (opcional).
        temperature: Temperatura de generación (opcional).
        system: Prompt del sistema (opcional).

    Returns:
        Texto generado por el modelo.
    """
    service = get_llm_service()

    if not service.is_loaded:
        return jsonify({
            "error": "Model not loaded",
            "message": "El modelo no está cargado. Use POST /api/v1/model/load",
        }), 503

    data = request.get_json()

    if not data or "prompt" not in data:
        return jsonify({
            "error": "Bad Request",
            "message": "Se requiere el campo 'prompt'",
        }), 400

    prompt = data["prompt"]
    messages = [{"role": "user", "content": prompt}]

    try:
        start_time = time.time()
        response_text = service.generate(
            messages=messages,
            max_new_tokens=data.get("max_tokens"),
            temperature=data.get("temperature"),
            min_p=data.get("min_p"),
            repetition_penalty=data.get("repetition_penalty"),
            system_prompt=data.get("system"),
        )
        generation_time = time.time() - start_time

        return jsonify({
            "success": True,
            "response": response_text,
            "generation_time_seconds": round(generation_time, 2),
        }), 200

    except Exception as e:
        logger.error("Error en generación: %s", str(e))
        return jsonify({
            "error": "Internal Server Error",
            "message": str(e),
        }), 500


@api_bp.errorhandler(404)
def not_found(error: Exception) -> tuple[Response, int]:
    """
    Manejador de errores 404.

    Args:
        error: Excepción del error.

    Returns:
        Respuesta JSON con el error.
    """
    return jsonify({
        "error": "Not Found",
        "message": "El recurso solicitado no existe",
    }), 404


@api_bp.errorhandler(500)
def internal_error(error: Exception) -> tuple[Response, int]:
    """
    Manejador de errores 500.

    Args:
        error: Excepción del error.

    Returns:
        Respuesta JSON con el error.
    """
    logger.error("Error interno: %s", str(error))
    return jsonify({
        "error": "Internal Server Error",
        "message": "Ocurrió un error interno del servidor",
    }), 500

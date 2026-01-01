"""
Servicio de LLM para el modelo LiquidAI LFM2-2.6B.

Este módulo proporciona una clase singleton para gestionar la carga
y generación de texto con el modelo LFM2-2.6B.
"""

import logging
import threading
from typing import Generator, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from ..config.settings import AEPModelConfig, AEP_MODEL_ID

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class AEPLLMService:
    """
    Servicio singleton para gestionar el modelo LiquidAI LFM2-2.6B.

    Esta clase implementa el patrón singleton para asegurar que solo
    existe una instancia del modelo cargado en memoria.

    Attributes:
        _instance: Instancia singleton de la clase.
        _lock: Lock para thread-safety.
        model: Modelo cargado de HuggingFace.
        tokenizer: Tokenizador del modelo.
        config: Configuración del modelo.
        is_loaded: Indica si el modelo está cargado.
    """

    _instance: Optional["AEPLLMService"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls, config: Optional[AEPModelConfig] = None) -> "AEPLLMService":
        """
        Crea o retorna la instancia singleton.

        Args:
            config: Configuración del modelo (solo se usa en la primera llamada).

        Returns:
            Instancia singleton del servicio.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Optional[AEPModelConfig] = None) -> None:
        """
        Inicializa el servicio de LLM.

        Args:
            config: Configuración del modelo.
        """
        if self._initialized:
            return

        self.config = config or AEPModelConfig()
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        self.is_loading = False
        self.load_thread: Optional[threading.Thread] = None
        self._initialized = True
        logger.info("AEPLLMService inicializado con config: %s", self.config.to_dict())

    def load_model(self) -> bool:
        """
        Carga el modelo y el tokenizador.

        Returns:
            True si el modelo se cargó correctamente, False en caso contrario.

        Raises:
            Exception: Si ocurre un error durante la carga del modelo.
        """
        if self.is_loaded:
            logger.info("El modelo ya está cargado")
            return True

        try:
            logger.info("Iniciando carga del modelo: %s", self.config.model_id)
            logger.info("Cache dir: %s", self.config.cache_dir)

            # Determinar el dtype
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            torch_dtype = dtype_map.get(self.config.torch_dtype, torch.bfloat16)

            # Configurar device_map basado en disponibilidad de GPU
            if self.config.device_map == "auto":
                if torch.cuda.is_available():
                    device_map = "auto"
                    logger.info("GPU detectada, usando device_map='auto'")
                else:
                    device_map = None
                    logger.info("CPU detectada, usando device_map=None")
            else:
                device_map = self.config.device_map

            # Configurar argumentos del modelo
            model_kwargs = {
                "torch_dtype": torch_dtype,
                "cache_dir": self.config.cache_dir,
                "trust_remote_code": True,
                "use_multiprocessing": False,
            }

            # Solo agregar device_map si no es None
            if device_map is not None:
                model_kwargs["device_map"] = device_map

            # Agregar Flash Attention si está habilitado
            if self.config.use_flash_attention:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Flash Attention 2 habilitado")

            # Cargar tokenizador
            logger.info("Cargando tokenizador...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_id,
                cache_dir=self.config.cache_dir,
                trust_remote_code=True,
            )

            # Cargar modelo
            logger.info("Cargando modelo (esto puede tomar varios minutos)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                **model_kwargs,
            )

            self.is_loaded = True
            logger.info("Modelo cargado exitosamente")

            # Mover modelo a CPU si no se usó device_map
            if device_map is None:
                self.model = self.model.to("cpu")
                logger.info("Modelo movido a CPU")

            # Información del dispositivo
            if hasattr(self.model, "device"):
                logger.info("Modelo en dispositivo: %s", self.model.device)

            return True

        except Exception as e:
            logger.error("Error al cargar el modelo: %s", str(e))
            self.is_loaded = False
            raise

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        min_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Genera una respuesta a partir de una lista de mensajes.

        Args:
            messages: Lista de mensajes con formato {"role": str, "content": str}.
            max_new_tokens: Número máximo de tokens a generar.
            temperature: Temperatura para la generación.
            min_p: Probabilidad mínima para sampling.
            repetition_penalty: Penalización por repetición.
            system_prompt: Prompt del sistema personalizado.

        Returns:
            Texto generado por el modelo.

        Raises:
            RuntimeError: Si el modelo no está cargado.
            ValueError: Si los mensajes están vacíos.
        """
        if not self.is_loaded:
            raise RuntimeError("El modelo no está cargado. Llama a load_model() primero.")

        if not messages:
            raise ValueError("La lista de mensajes no puede estar vacía")

        # Usar valores por defecto si no se proporcionan
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature
        min_p = min_p or self.config.min_p
        repetition_penalty = repetition_penalty or self.config.repetition_penalty

        # Preparar mensajes con system prompt
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        else:
            full_messages.append({
                "role": "system",
                "content": "You are a helpful assistant trained by Liquid AI."
            })
        full_messages.extend(messages)

        # Aplicar template de chat
        input_ids = self.tokenizer.apply_chat_template(
            full_messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
        ).to(self.model.device)

        logger.debug(
            "Generando respuesta con %d tokens de entrada", input_ids.shape[1]
        )

        # Generar respuesta
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                do_sample=True,
                temperature=temperature,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decodificar solo los tokens nuevos
        generated_ids = output[0][input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        logger.debug("Respuesta generada con %d tokens", len(generated_ids))

        return response.strip()

    def generate_stream(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        min_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        Genera una respuesta en streaming.

        Args:
            messages: Lista de mensajes con formato {"role": str, "content": str}.
            max_new_tokens: Número máximo de tokens a generar.
            temperature: Temperatura para la generación.
            min_p: Probabilidad mínima para sampling.
            repetition_penalty: Penalización por repetición.
            system_prompt: Prompt del sistema personalizado.

        Yields:
            Fragmentos de texto generados por el modelo.

        Raises:
            RuntimeError: Si el modelo no está cargado.
            ValueError: Si los mensajes están vacíos.
        """
        if not self.is_loaded:
            raise RuntimeError("El modelo no está cargado. Llama a load_model() primero.")

        if not messages:
            raise ValueError("La lista de mensajes no puede estar vacía")

        # Usar valores por defecto si no se proporcionan
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature
        min_p = min_p or self.config.min_p
        repetition_penalty = repetition_penalty or self.config.repetition_penalty

        # Preparar mensajes con system prompt
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        else:
            full_messages.append({
                "role": "system",
                "content": "You are a helpful assistant trained by Liquid AI."
            })
        full_messages.extend(messages)

        # Aplicar template de chat
        input_ids = self.tokenizer.apply_chat_template(
            full_messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
        ).to(self.model.device)

        # Configurar streamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        # Argumentos de generación
        generation_kwargs = {
            "input_ids": input_ids,
            "streamer": streamer,
            "do_sample": True,
            "temperature": temperature,
            "min_p": min_p,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        # Ejecutar generación en thread separado
        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # Yield de tokens generados
        for text in streamer:
            yield text

        thread.join()

    def get_model_info(self) -> dict:
        """
        Obtiene información sobre el modelo.

        Returns:
            Diccionario con información del modelo.
        """
        info = {
            "model_id": self.config.model_id,
            "is_loaded": self.is_loaded,
            "config": self.config.to_dict(),
        }

        if self.is_loaded and self.model is not None:
            info["device"] = str(self.model.device)
            info["dtype"] = str(self.model.dtype)
            if hasattr(self.model, "num_parameters"):
                info["num_parameters"] = self.model.num_parameters()

        return info

    def unload_model(self) -> None:
        """
        Descarga el modelo de memoria.
        """
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        self.is_loaded = False

        # Limpiar caché de CUDA si está disponible
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Modelo descargado de memoria")

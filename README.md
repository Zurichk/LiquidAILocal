# LiquidAI LFM2-2.6B Chat Application

<p align="center">
  <img src="app/static/img/favicon.svg" alt="LiquidAI Chat Logo" width="80" height="80">
</p>

<p align="center">
  <strong>Una interfaz web moderna para interactuar con el modelo LiquidAI LFM2-2.6B</strong>
</p>

<p align="center">
  <a href="#características">Características</a> •
  <a href="#requisitos">Requisitos</a> •
  <a href="#instalación">Instalación</a> •
  <a href="#uso">Uso</a> •
  <a href="#api">API</a> •
  <a href="#despliegue">Despliegue</a>
</p>

---

## 🌟 Características

- **🤖 Modelo LFM2-2.6B**: Modelo híbrido de Liquid AI optimizado para conversaciones
- **💬 Chat Web Moderno**: Interfaz de usuario intuitiva y responsive
- **🔌 API REST**: Compatible con el formato de OpenAI
- **📡 Streaming**: Respuestas en tiempo real con Server-Sent Events
- **🔒 Autenticación**: Protección opcional por API key
- **🐳 Docker Ready**: Preparado para despliegue en contenedores
- **🌍 Multiidioma**: Soporte para 8 idiomas (ES, EN, FR, DE, AR, CN, JP, KR)

## 📋 Requisitos

### Hardware Mínimo
- **CPU**: 4 cores
- **RAM**: 16 GB (recomendado 32 GB)
- **Almacenamiento**: 20 GB libres (para el modelo)

### Hardware Recomendado (GPU)
- **GPU**: NVIDIA con 8+ GB VRAM (RTX 3080, RTX 4070, etc.)
- **CUDA**: 12.1 o superior

### Software
- Python 3.11+
- Docker & Docker Compose (para despliegue)
- NVIDIA Container Toolkit (para GPU)

## 🚀 Instalación

> **⚡ INICIO RÁPIDO**: Si tienes GPU, primero ejecuta `python check_gpu.py` para verificar compatibilidad.  
> Consulta [GETTING_STARTED.md](GETTING_STARTED.md) para instrucciones detalladas.

### Opción 1: Instalación Local (Recomendado para desarrollo)

1. **Clonar el repositorio**
   ```bash
   git clone https://github.com/tu-usuario/liquidai-chat.git
   cd liquidai-chat
   ```

2. **Crear y activar entorno virtual**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux/macOS
   python -m venv venv
   source venv/bin/activate
   ```

3. **Verificar compatibilidad GPU** (si tienes GPU)
   ```bash
   pip install psutil  # Para verificación completa
   python check_gpu.py
   ```

4. **Instalar dependencias**
   ```bash
   # CPU solamente
   pip install -r requirements.txt
   
   # Con soporte GPU (si tu GPU es compatible)
   pip install -r requirements-gpu.txt
   ```

5. **Configurar variables de entorno**
   ```bash
   # Windows
   Copy-Item .env.example .env
   
   # Linux/macOS
   cp .env.example .env
   
   # Editar .env según las recomendaciones de check_gpu.py
   ```

6. **Ejecutar la aplicación**
   ```bash
   # IMPORTANTE: Ejecutar desde el directorio raíz del proyecto
   python -m app.src.app
   ```

7. **Abrir en el navegador**
   ```
   http://localhost:5049
   ```

### Opción 2: Docker CPU (Para VPS sin GPU)

1. **Construir la imagen**
   ```bash
   docker-compose build
   ```

2. **Iniciar el contenedor**
   ```bash
   docker-compose up -d
   ```

3. **Ver logs**
   ```bash
   docker-compose logs -f liquidai-chat
   ```

### Opción 3: Docker con GPU (Recomendado para PC con GPU)

> **Prerequisitos:**
> - NVIDIA GPU con drivers actualizados
> - [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) instalado
> - Docker Desktop configurado para usar GPU

```bash
# Verificar que Docker puede acceder a la GPU
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi

# Ejecutar con GPU
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d

# Ver logs
docker-compose logs -f liquidai-chat
```

## 💻 Uso

### Primera Ejecución

**⚠️ IMPORTANTE**: La primera vez que ejecutes la aplicación:

1. El modelo se descargará automáticamente (~5.5 GB)
2. Puede tardar 5-15 minutos según tu conexión
3. El modelo se guarda en `./models` (o volumen Docker) para reutilizar

### Interfaz Web

1. Accede a `http://localhost:5049`
2. Haz clic en **"Cargar modelo"** en la barra lateral
3. Espera a que el modelo se cargue en memoria:
   - **GPU**: 30-60 segundos
   - **CPU**: 2-3 minutos
4. ¡Comienza a chatear!

### Configuración

Puedes personalizar la generación desde el botón de ⚙️ configuración:

| Parámetro | Descripción | Valor por defecto |
|-----------|-------------|-------------------|
| System Prompt | Instrucciones para el modelo | Asistente útil |
| Max Tokens | Longitud máxima de respuesta | 512 |
| Temperature | Creatividad (0-2) | 0.3 |
| Streaming | Respuestas en tiempo real | Habilitado |

## 🔌 API

La API es compatible con el formato de OpenAI. Consulta la documentación completa en:
```
http://localhost:5000/docs
```

### Endpoints Principales

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/api/v1/health` | Estado de la API |
| GET | `/api/v1/model/info` | Información del modelo |
| POST | `/api/v1/model/load` | Cargar modelo |
| POST | `/api/v1/model/unload` | Descargar modelo |
| POST | `/api/v1/chat/completions` | Chat (compatible OpenAI) |
| POST | `/api/v1/generate` | Generación simple |

### Ejemplo de Uso

```python
import requests

response = requests.post(
    "http://localhost:5049/api/v1/chat/completions",
    json={
        "messages": [
            {"role": "user", "content": "¿Qué es la inteligencia artificial?"}
        ],
        "max_tokens": 256
    }
)

print(response.json()["choices"][0]["message"]["content"])
```

### Autenticación (Opcional)

Si configuras `AEP_API_KEY`, incluye el header en tus peticiones:

```bash
curl -H "X-API-Key: tu-api-key" http://localhost:5049/api/v1/chat/completions ...
```

## 🐳 Despliegue en Coolify (VPS sin GPU)

### Configuración en Coolify

1. **Crear nuevo servicio** → Docker Compose

2. **Configurar repositorio Git** con tu código

3. **Dockerfile a usar**: `Dockerfile` (NO el Dockerfile.gpu)

4. **Docker Compose**: Solo `docker-compose.yml` (sin el gpu override)

5. **Variables de entorno recomendadas**:
   ```env
   AEP_SECRET_KEY=<generar-clave-segura-aqui>
   AEP_API_KEY=<opcional-para-proteger-endpoints>
   AEP_DEBUG=false
   AEP_DEVICE_MAP=cpu
   AEP_LOAD_MODEL_ON_STARTUP=false
   AEP_USE_FLASH_ATTENTION=false
   AEP_MAX_TOKENS=512
   ```

6. **Configurar volúmenes** para persistir el modelo:
   - `liquidai-models:/app/models` (Importante: evita re-descargar 5.5GB)

7. **Health Check**: Ya configurado automáticamente

### Recursos Recomendados VPS

| Configuración | CPU | RAM | Almacenamiento | Rendimiento |
|---------------|-----|-----|----------------|-------------|
| Mínima | 4 cores | 16 GB | 30 GB | Lento (~45s/100 tokens) |
| Recomendada | 8 cores | 32 GB | 50 GB | Aceptable (~20s/100 tokens) |
| Óptima | 16 cores | 64 GB | 100 GB | Rápido (~10s/100 tokens) |

**Para GPU** (si tu VPS tiene GPU):
- GPU: NVIDIA con 8+ GB VRAM
- Usa `docker-compose.gpu.yml`
- Rendimiento: ~2-4s/100 tokens

## ⚙️ Variables de Entorno

| Variable | Descripción | Por defecto |
|----------|-------------|-------------|
| `AEP_HOST` | Host del servidor | `0.0.0.0` |
| `AEP_PORT` | Puerto del servidor | `5049` |
| `AEP_DEBUG` | Modo debug | `false` |
| `AEP_SECRET_KEY` | Clave secreta Flask | Auto-generada |
| `AEP_API_KEY` | API key para autenticación | - |
| `HF_HOME` | Directorio del modelo (HuggingFace) | `./models` |
| `AEP_MODEL_CACHE_DIR` | Directorio del modelo (legacy) | `./models` |
| `AEP_LOAD_MODEL_ON_STARTUP` | Cargar modelo al inicio | `false` |
| `AEP_DEVICE_MAP` | Dispositivo (auto/cpu/cuda) | `auto` |
| `AEP_MAX_TOKENS` | Tokens máximos por defecto | `512` |
| `AEP_TEMPERATURE` | Temperatura por defecto | `0.3` |
| `AEP_USE_FLASH_ATTENTION` | Usar Flash Attention 2 | `false` |

## 📁 Estructura del Proyecto

```
liquidai-chat/
├── app/
│   ├── src/
│   │   ├── __init__.py
│   │   ├── app.py              # Aplicación Flask
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   └── routes.py       # Endpoints API
│   │   ├── config/
│   │   │   ├── __init__.py
│   │   │   └── settings.py     # Configuración
│   │   └── models/
│   │       ├── __init__.py
│   │       └── llm_service.py  # Servicio LLM
│   ├── static/
│   │   ├── css/
│   │   ├── js/
│   │   └── img/
│   ├── templates/
│   │   ├── index.html
│   │   └── api_docs.html
│   └── docs/
├── tests/
├── Dockerfile
├── Dockerfile.gpu
├── docker-compose.yml
├── docker-compose.gpu.yml
├── requirements.txt
├── requirements-gpu.txt
├── .env.example
├── .gitignore
└── README.md
```

## 🧪 Testing

```bash
# Instalar dependencias de desarrollo
pip install pytest pytest-cov

# Ejecutar tests
pytest tests/ -v

# Con cobertura
pytest tests/ --cov=app --cov-report=html
```

## 🤝 Contribuir

1. Fork el repositorio
2. Crea una rama (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver [LICENSE](LICENSE) para más detalles.

El modelo LiquidAI LFM2-2.6B tiene su propia licencia: [LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM2-2.6B)

## 🙏 Agradecimientos

- [Liquid AI](https://www.liquid.ai/) por el modelo LFM2-2.6B
- [Hugging Face](https://huggingface.co/) por la infraestructura de transformers

---

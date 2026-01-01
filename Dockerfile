# =============================================================================
# Dockerfile para LiquidAI LFM2-2.6B Chat Application
# Optimizado para despliegue en Coolify/VPS
# =============================================================================

# Etapa 1: Build stage
FROM python:3.11-slim as builder

# Argumentos de construcción
ARG DEBIAN_FRONTEND=noninteractive

# Instalar dependencias del sistema necesarias para compilación
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar requirements primero para aprovechar cache de Docker
COPY requirements.txt .

# Crear entorno virtual e instalar dependencias
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Instalar dependencias de Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Etapa 2: Runtime stage
# =============================================================================
FROM python:3.11-slim as runtime

# Metadata
LABEL maintainer="AEP Team"
LABEL description="LiquidAI LFM2-2.6B Chat Application"
LABEL version="1.0.0"

# Argumentos de construcción
ARG DEBIAN_FRONTEND=noninteractive

# Instalar dependencias mínimas del runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Crear usuario no-root para seguridad
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Crear directorios necesarios
RUN mkdir -p /app /app/models /app/logs && \
    chown -R appuser:appgroup /app

# Copiar entorno virtual desde builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Configurar directorio de trabajo
WORKDIR /app

# Copiar código de la aplicación
COPY --chown=appuser:appgroup app/ ./app/

# Variables de entorno por defecto
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    AEP_HOST=0.0.0.0 \
    AEP_PORT=5000 \
    AEP_DEBUG=false \
    HF_HOME=/app/models \
    AEP_LOAD_MODEL_ON_STARTUP=false \
    AEP_DEVICE_MAP=auto \
    AEP_MAX_TOKENS=512 \
    AEP_TEMPERATURE=0.3 \
    TRANSFORMERS_CACHE=/app/models \
    HF_HOME=/app/models

# Exponer puerto
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5049/api/v1/health || exit 1

# Cambiar a usuario no-root
USER appuser

# Script de inicio
COPY --chown=appuser:appgroup docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

# Comando por defecto
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "300", "app.src.app:app"]

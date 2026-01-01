#!/bin/bash
# =============================================================================
# Docker Entrypoint Script para LiquidAI Chat Application
# =============================================================================

set -e

echo "=============================================="
echo "  LiquidAI LFM2-2.6B Chat Application"
echo "=============================================="
echo ""

# Mostrar información del sistema
echo "📊 Información del sistema:"
echo "   - Python: $(python --version)"
echo "   - Host: ${AEP_HOST:-0.0.0.0}"
echo "   - Port: ${AEP_PORT:-5049}"
echo "   - Debug: ${AEP_DEBUG:-false}"
echo "   - Device Map: ${AEP_DEVICE_MAP:-auto}"
echo "   - Model Cache: ${HF_HOME:-${AEP_MODEL_CACHE_DIR:-/app/models}}"
echo ""

# Verificar si el directorio de modelos tiene permisos correctos
MODEL_CACHE_DIR="${HF_HOME:-${AEP_MODEL_CACHE_DIR:-/app/models}}"
if [ -d "$MODEL_CACHE_DIR" ]; then
    echo "📁 Directorio de modelos existe: $MODEL_CACHE_DIR"
else
    echo "📁 Creando directorio de modelos: $MODEL_CACHE_DIR"
    mkdir -p "$MODEL_CACHE_DIR"
fi

# Verificar CUDA si está disponible
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 GPU detectada:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
else
    echo "⚠️  No se detectó GPU, el modelo se ejecutará en CPU"
    echo ""
fi

# Información sobre el modelo
echo "🤖 Modelo configurado: LiquidAI/LFM2-2.6B"
echo "   El modelo se descargará automáticamente si no existe"
echo ""

# Pre-descargar el modelo si se especifica
if [ "${AEP_PRELOAD_MODEL:-false}" = "true" ]; then
    echo "⏳ Pre-descargando modelo..."
    python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

model_id = 'LiquidAI/LFM2-2.6B'
cache_dir = os.environ.get('HF_HOME', os.environ.get('AEP_MODEL_CACHE_DIR', '/app/models'))

print(f'Descargando tokenizador a {cache_dir}...')
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)

print(f'Descargando modelo a {cache_dir}...')
# Solo descargar, no cargar en memoria completamente
from huggingface_hub import snapshot_download
snapshot_download(repo_id=model_id, cache_dir=cache_dir)

print('✅ Modelo pre-descargado exitosamente')
"
    echo ""
fi

echo "🚀 Iniciando servidor..."
echo "=============================================="
echo ""

# Ejecutar comando pasado al entrypoint
exec "$@"

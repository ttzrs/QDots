#!/bin/bash
# Ejecutar VQE con aceleración GPU en Docker
# Requiere: Docker, nvidia-container-toolkit, GPU NVIDIA

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  QDots VQE - Ejecución con GPU (Docker + NVIDIA)              ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Verificar GPU
if ! nvidia-smi &>/dev/null; then
    echo "❌ Error: nvidia-smi no disponible. ¿Está instalado el driver NVIDIA?"
    exit 1
fi

echo "✓ GPU detectada:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

# Verificar Docker
if ! docker --version &>/dev/null; then
    echo "❌ Error: Docker no instalado"
    exit 1
fi

# Verificar nvidia-container-toolkit
if ! docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
    echo "⚠ Advertencia: nvidia runtime no detectado en Docker"
    echo "  Instalar: sudo dnf install nvidia-container-toolkit"
fi

# Construir imagen si no existe
if ! docker images | grep -q "qdots-vqe-gpu"; then
    echo "→ Construyendo imagen Docker con soporte GPU..."
    docker build -f Dockerfile.gpu -t qdots-vqe-gpu:latest .
fi

# Ejecutar
echo ""
echo "→ Ejecutando VQE con GPU..."
echo "─────────────────────────────────────────────────────────────────"

docker run --rm --gpus all \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e QULACS_USE_GPU=1 \
    -v "$(pwd)":/app:ro \
    qdots-vqe-gpu:latest qdot_vqe_gpu.py

echo "─────────────────────────────────────────────────────────────────"
echo "✓ Completado"

#!/bin/bash
#═══════════════════════════════════════════════════════════════════════════════
#  TEST DE SIMULACIÓN OPENFOAM - REACTOR DBD
#═══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASE_DIR="$(dirname "$SCRIPT_DIR")"
DOCKER_IMAGE="openfoam/openfoam11-paraview510"

echo "═══════════════════════════════════════════════════════════════════════════"
echo "  TEST DE OPENFOAM - REACTOR DBD"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""
echo "  Directorio del caso: $CASE_DIR"
echo ""

# Generar archivos del caso
echo "→ Generando archivos del caso..."
cd "$CASE_DIR"
python3 scripts/generate_mesh.py --params ../reactor_optimized.json

echo ""
echo "→ Ejecutando blockMesh en Docker..."
docker run --rm \
    -v "$CASE_DIR:/case:rw" \
    -w /case \
    $DOCKER_IMAGE \
    bash -c "source /opt/openfoam11/etc/bashrc && blockMesh"

if [ $? -ne 0 ]; then
    echo "✗ Error en blockMesh"
    exit 1
fi

echo ""
echo "→ Verificando malla..."
docker run --rm \
    -v "$CASE_DIR:/case:rw" \
    -w /case \
    $DOCKER_IMAGE \
    bash -c "source /opt/openfoam11/etc/bashrc && checkMesh"

echo ""
echo "→ Ejecutando simpleFoam (esto puede tardar)..."
docker run --rm \
    -v "$CASE_DIR:/case:rw" \
    -w /case \
    $DOCKER_IMAGE \
    bash -c "source /opt/openfoam11/etc/bashrc && simpleFoam"

echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo "  ✓ Simulación completada"
echo "  Resultados en: $CASE_DIR"
echo "═══════════════════════════════════════════════════════════════════════════"

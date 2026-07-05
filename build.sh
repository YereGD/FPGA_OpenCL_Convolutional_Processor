#!/bin/bash

# Salir inmediatamente si ocurre un error
set -e

echo "================================================="
echo "🏗️  Compilando programa Host C++ para DE10-Nano..."
echo "================================================="

# 1. Obtener los flags de OpenCL de Intel automáticamente
echo "[INFO] Cargando configuración de OpenCL..."
AOCL_COMPILE_CONFIG=$(aocl compile-config)
AOCL_LINK_CONFIG=$(aocl link-config)

# 2. Obtener los flags de OpenCV (intenta opencv4 primero, luego opencv clásico)
echo "[INFO] Cargando configuración de OpenCV..."
OPENCV_FLAGS=$(pkg-config --cflags --libs opencv4 2>/dev/null || pkg-config --cflags --libs opencv)

if [ -z "$OPENCV_FLAGS" ]; then
    echo "❌ ERROR: No se encontró OpenCV. Asegúrate de instalarlo con: sudo apt-get install libopencv-dev"
    exit 1
fi

# 3. Compilación final con Optimización Extrema (-O3)
echo "[INFO] Compilando con g++ (-O3)..."
g++ -O3 -Wall -std=c++17 main.cpp -o pipeonnx_hw \
    $AOCL_COMPILE_CONFIG \
    $AOCL_LINK_CONFIG \
    $OPENCV_FLAGS

echo "================================================="
echo "✅ ¡Compilación C++ exitosa!"
echo "🚀 Archivo generado: pipeonnx_hw"
echo "👉 Para ejecutar: ./pipeonnx_hw"
echo "================================================="

#!/bin/bash

# Script de inicialización del entorno de desarrollo
# Uso: ./setup_dev.sh

# Nombre del entorno virtual
VENV=".venv"

echo "[🚀] Creando entorno virtual en $VENV..."
python3 -m venv $VENV

echo "[✅] Activando entorno virtual..."
source $VENV/bin/activate

echo "[📦] Actualizando pip..."
pip install --upgrade pip

echo "[📦] Instalando dependencias principales (requirements.txt)..."
pip install -r requirements.txt

echo "[📦] Instalando dependencias de desarrollo (requirements-dev.txt)..."
pip install -r requirements-dev.txt

echo "[🎯] Entorno listo. Actívalo con:"
echo "source $VENV/bin/activate"

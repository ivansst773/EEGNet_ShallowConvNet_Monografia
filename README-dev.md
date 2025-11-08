# Guía de entorno de desarrollo – EEGNet_ShallowConvNet_Monografia

Este documento explica cómo usar las dependencias de desarrollo incluidas en `requirements-dev.txt`.  
El objetivo es mantener separado el entorno de investigación (pipeline EEGNet/ShallowConvNet) del entorno de desarrollo y documentación.

---

## 🚀 Instalación del entorno de desarrollo

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/iansst773/EEGNet-ShallowConvNet.git
   cd EEGNet-ShallowConvNet


Crear un entorno virtual (opcional pero recomendado):

bash
python3.11 -m venv .venv
source .venv/bin/activate
Instalar dependencias de desarrollo:

bash
pip install -r requirements-dev.txt
⚙️ Herramientas incluidas
Jupyter / JupyterLab → ejecución de notebooks interactivos.

Black, Flake8, Isort → formateo y estilo de código.

Pytest → pruebas unitarias y cobertura.

Sphinx + myst-parser → documentación técnica en formato HTML/PDF.

Pre-commit → hooks para mantener calidad de código antes de cada commit.

📌 Uso rápido
Ejecutar notebooks:

bash
jupyter lab
Formatear código automáticamente:

bash
black src/
Ejecutar pruebas:

bash
pytest
Generar documentación:

bash
cd docs/
make html
📝 Notas
requirements.txt → dependencias mínimas para correr el pipeline EEGNet/ShallowConvNet.

requirements-dev.txt → dependencias opcionales para desarrollo, pruebas y documentación.

Mantener ambos archivos en la raíz del proyecto facilita reproducibilidad y colaboración.
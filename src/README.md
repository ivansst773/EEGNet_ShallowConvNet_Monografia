# src

Este directorio contiene el **código fuente** del proyecto, incluyendo implementaciones de modelos, scripts de entrenamiento/prueba y utilidades compartidas.

## 📂 Subcarpetas

- `eegnet/`: Implementación y scripts relacionados con **EEGNet**.
  - `train_eegnet_bci_iv2a.py`: Entrenamiento de EEGNet sobre BCI IV 2a.
  - `test_eegnet_reducido.py`: Evaluación del modelo entrenado.
- `shallowconvnet/`: Implementación y scripts para **Shallow ConvNet**.
  - `train_shallowconvnet.py`: Entrenamiento de Shallow ConvNet.
  - `test_shallowconvnet.py`: Evaluación del modelo entrenado.
- `utils.py`: Funciones utilitarias (definición de modelos, carga de datos, preprocesamiento, métricas).
- `notebooks/`: Notebooks exploratorios y de análisis.

---

## ⚙️ Requisitos

- Python 3.9+
- PyTorch
- NumPy  
*(Opcional: MNE o SciPy si se desea cargar datos EEG reales en lugar de los datos dummy actuales).*

Instalación rápida:

```bash
pip install torch torchvision numpy


🚀 Uso
1. Entrenar EEGNet
bash
cd src/eegnet
python train_eegnet_bci_iv2a.py
Genera el modelo eegnet_bci_iv2a.pth.

2. Probar EEGNet
bash
cd src/eegnet
python test_eegnet_reducido.py
3. Entrenar ShallowConvNet

bash
cd src/shallowconvnet
python train_shallowconvnet.py
Genera el modelo shallowconvnet_bci_iv2a.pth.

4. Probar ShallowConvNet
bash
cd src/shallowconvnet
python test_shallowconvnet.py
📌 Notas
Actualmente, utils.py genera datos dummy para pruebas rápidas.

Para usar el dataset real BCI Competition IV 2a, reemplaza la función load_bci_iv2a() en utils.py con la lógica de carga de .gdf y etiquetas.

Mantén notebooks reproducibles y complementa con scripts .py para pipelines automatizados.

✍️ Autor: Edgar Iván Calpa Cuacialpud Universidad Nacional de Colombia – Sede Manizales
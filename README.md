# EEGNet_ShallowConvNet_Monografia

Repositorio del trabajo monográfico: comparación de EEGNet y Shallow ConvNet aplicados a señales EEG, con enfoque en biomarcadores de enfermedades neurodegenerativas.

---

## 📂 Estructura del repositorio

EEGNet-ShallowConvNet/
│── README.md
│── bitacora.md
│── referencias.bib
│── .gitignore
│
├── docs/
│   ├── monografia/
│   │   ├── README.md
│   │   └── figuras/
│   ├── presentacion/
│   │   ├── README.md
│   │   └── logos/
│   └── productos/
│       └── README.md
│
├── src/
│   ├── eegnet/
│   │   └── train_eegnet_bci_iv2a.py
│   ├── shallowconvnet/
│   │   └── train_shallowconvnet_bci_iv2a.py
│   ├── models.py
│   ├── utils.py
│   └── notebooks/
│
├── results/
│   ├── figuras/
│   ├── tablas/
│   ├── reportes/
│   └── README.md
│
└── data/
    ├── raw/
    │   └── BCI_IV_2a/   # colocar aquí los .gdf y true_labels
    ├── processed/
    └── README.md

---

## 🚀 Cómo usar el pipeline BCI IV-2a

1. **Preparar datos**  
   - Colocar los archivos `.gdf` y la carpeta `true_labels/` dentro de `data/raw/BCI_IV_2a/`.

2. **Probar el loader**  
   ```bash
   python src/utils.py

Esto imprime las dimensiones de los tensores cargados y confirma que el preprocesamiento funciona.
4Entrenar EEGNet (smoke test)

bash
python src/eegnet/train_eegnet_bci_iv2a.py
Entrenar ShallowConvNet (smoke test)

bash
python src/shallowconvnet/train_shallowconvnet_bci_iv2a.py

⚙️ Características implementadas hasta ahora
Loader refinado (utils.py) con:

Lectura de .gdf y .mat.

Normalización trial-wise (z-score).

Filtro bandpass opcional (4–40 Hz).

Segmentación opcional en ventanas (segment=True).

Modelos (models.py):

EEGNet.

ShallowConvNet.

Scripts de entrenamiento:

train_eegnet_bci_iv2a.py (smoke test).

train_shallowconvnet_bci_iv2a.py (smoke test).

Documentación inicial en README y bitácora.

📌 Próximos pasos
Entrenamiento completo en BCI IV-2a con todos los sujetos.

Uso de DataLoader y batches.

Guardado de métricas y modelos en results/.

Migración del pipeline al dataset clínico CN/MCI/AD + tau.
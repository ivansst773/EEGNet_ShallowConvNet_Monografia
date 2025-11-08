# Bitácora

Registro de cambios importantes del proyecto.

---

## 2025-09-05
- Edgar Calpa
- Inicio del proyecto monográfico. Definición del tema: comparación de EEGNet y ShallowConvNet aplicados a EEG con enfoque en biomarcadores de enfermedades neurodegenerativas.
- Creación de repositorio inicial en GitHub.

## 2025-09-10
- Edgar Calpa
- Revisión bibliográfica inicial sobre EEGNet y ShallowConvNet.
- Identificación de datasets públicos (BCI Competition IV-2a) como punto de partida.

## 2025-09-15
- Edgar Calpa
- Organización preliminar de carpetas (`src/`, `data/`, `results/`).
- Creación de `.gitignore` para excluir datasets y archivos temporales.

## 2025-09-20
- Edgar Calpa
- Configuración de entorno en WSL2 con Python, PyTorch, MNE.
- Pruebas iniciales de carga de datos EEG.

## 2025-09-25
- Edgar Calpa
- Implementación inicial del loader en `src/utils.py`.
- Lectura de archivos `.gdf` y `.mat` del BCI IV-2a.

## 2025-10-01
- Edgar Calpa
- Añadida normalización trial-wise (z-score).
- Split train/val con `train_test_split`.

## 2025-10-05
- Edgar Calpa
- Integrado filtro bandpass opcional (4–40 Hz).
- Validación con smoke test en `utils.py`.

## 2025-10-10
- Edgar Calpa
- Implementada segmentación opcional en ventanas (`segment=True`).
- Actualización de `utils.py`.

## 2025-10-15
- Edgar Calpa
- Añadidos modelos EEGNet y ShallowConvNet en `src/models.py`.

## 2025-10-20
- Edgar Calpa
- Creado script de entrenamiento `train_eegnet_bci_iv2a.py` (smoke test).
- Verificación de pérdida decreciente en subset de 50 trials.

## 2025-10-25
- Edgar Calpa
- Creado script de entrenamiento `train_shallowconvnet_bci_iv2a.py` (smoke test).
- Validación rápida en conjunto de validación.

## 2025-11-01
- Edgar Calpa
- Documentación en `README.md` con instrucciones de uso del pipeline BCI IV-2a.
- Actualización de bitácora con registros desde septiembre.

# Bitácora de desarrollo – EEGNet_ShallowConvNet_Monografia

---

## 🗓️ 05 de noviembre de 2025
- Se configuró el entorno en WSL con soporte CUDA.  
- PyTorch estaba inicialmente en versión 2.0.1+cu117.  
- Se verificó la detección de la GPU NVIDIA GeForce GTX 1050.  

## 🗓️ 06 de noviembre de 2025
- Se intentó migrar a PyTorch cu122, pero no se encontraron binarios compatibles.  
- Se instaló finalmente PyTorch 2.5.1+cu121 con CUDA 12.1 y cuDNN 9.1.  
- Se confirmó que el entorno reconoce la GPU y corre en CUDA.  

## 🗓️ 07 de noviembre de 2025
- Se instaló MNE-Python 1.10.2 para lectura de archivos `.gdf`.  
- Se detectó incompatibilidad con NumPy ≥1.24 (error `np.fromstring`).  
- Se resolvió bajando NumPy a 1.23.5.  
- Se ajustó SciPy a 1.10.1 para compatibilidad con NumPy 1.23.5.  
- Se probó el loader (`utils.py`) y se confirmó lectura correcta de los `.gdf`.  
- Se ejecutó entrenamiento de ShallowConvNet en GPU:  
  - Epoch 1 → Loss: 2.47  
  - Epoch 2 → Loss: 1.25  
  - Validación inicial → Accuracy: 50%.  

---

## 📌 Próximos pasos
- Optimizar hiperparámetros (learning rate, batch size, número de epochs).  
- Implementar `DataLoader` para batches.  
- Guardar métricas y modelos en `results/`.  
- Extender pipeline al dataset clínico CN/MCI/AD + tau.  

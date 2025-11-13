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


## 🗓️ 12 de noviembre de 2025
Se recreó el entorno virtual .venv con Python 3.11 para evitar inconsistencias con Python 3.10.

Se instaló exitosamente PyTorch 2.5.1+cu121, torchvision 0.20.1+cu121, torchaudio 2.5.1+cu121.

Se confirmaron versiones estables:

NumPy 1.26.4

SciPy 1.11.4

MNE 1.10.2

scikit-learn 1.7.2

matplotlib 3.10.7

pandas 2.3.3

Se verificó la detección de la GPU NVIDIA GeForce GTX 1050 con soporte CUDA disponible.

Se resolvió el problema de desincronización entre pip list y import torch al reinstalar todo en el .venv correcto.

El entorno quedó listo para correr los scripts de entrenamiento (train_eegnet_bci_iv2a.py, train_shallowconvnet_bci_iv2a.py) sin errores de librerías.


**Fecha:** 2025-11-12  
**Modelo:** EEGNet  
**Sujeto:** A01  
**Dispositivo:** GPU (NVIDIA GTX 1050)  
**Dataset:** BCI Competition IV-2a  
**Segmentación:** No  
**Filtro aplicado:** Band-pass 4–40 Hz  

### ⚙️ Configuración
- Epochs: 2  
- Batch size: 16  
- Learning rate: 0.001  
- Dropout: 0.25  
- Optimizer: Adam  

### 📊 Resultados
- **Entrenamiento**
  - Epoch 1 → Loss: 1.4420
  - Epoch 2 → Loss: 1.1628
- **Validación**
  - Loss final: 1.2149
  - Accuracy: 50.00 %

### 📝 Observaciones
- Estratificación: desactivada (clases con <2 muestras)  
- Loss decreciente, accuracy inicial moderada.  
- Pipeline estable, sin errores de ejecución.


**Fecha:** 2025-11-12 Modelo: 
**ShallowConvNet Sujeto:** A01 
**Dispositivo:** GPU (NVIDIA GTX 1050) 
**Dataset:** BCI Competition IV-2a 
**Segmentación:** No 
**Filtro aplicado:** Band-pass 4–40 Hz

⚙️ Configuración
Epochs: 2

Batch size: 16

Learning rate: 0.001

Dropout: 0.50

Optimizer: Adam

📊 Resultados
Entrenamiento

Epoch 1 → Loss: 1.6835

Epoch 2 → Loss: 0.8931

Validación

Loss final: 6.1624

Accuracy: 25.00 %


## 📊 Comparativa inicial – Smoke tests (BCI IV‑2a, sujeto A01)

| Modelo           | Epochs | Batch Size | Learning Rate | Dropout | Train Loss Final | Val Loss Final | Val Accuracy |
|------------------|--------|------------|---------------|---------|------------------|----------------|--------------|
| **EEGNet**       | 2      | 16         | 0.001         | 0.25    | 1.1628           | 1.2149         | 50.00 %      |
| **ShallowConvNet** | 2    | 16         | 0.001         | 0.50    | 0.8931           | 6.1624         | 25.00 %      |

### 📝 Observaciones rápidas
- **EEGNet**: más estable entre entrenamiento y validación, accuracy inicial moderada.  
- **ShallowConvNet**: entrenó bien en train, pero se sobreajustó y no generalizó (val_loss muy alto, accuracy baja).  

### 📈 Gráficas asociadas
![Loss Comparison](results/figuras/loss_comparison_2025-11-12.png)  
![Accuracy Comparison](results/figuras/accuracy_comparison_2025-11-12.png)
 

## 📊 Comparativa – Smoke tests con segmentación (BCI IV‑2a, sujeto A01)

<table>
  <thead>
    <tr>
      <th>Modelo</th>
      <th>Epochs</th>
      <th>Batch Size</th>
      <th>Learning Rate</th>
      <th>Dropout</th>
      <th>Segmentación</th>
      <th>Train Loss Final</th>
      <th>Val Loss Final</th>
      <th>Val Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>EEGNet</b></td>
      <td>2</td>
      <td>16</td>
      <td>0.001</td>
      <td>0.25</td>
      <td style="color:green;">True ✅</td>
      <td>1.1655</td>
      <td>1.3394</td>
      <td style="color:green;"><b>50 %</b></td>
    </tr>
    <tr>
      <td><b>ShallowConvNet</b></td>
      <td>2</td>
      <td>16</td>
      <td>0.001</td>
      <td>0.50</td>
      <td style="color:green;">True ✅</td>
      <td>1.1225</td>
      <td style="color:red;"><b>6.8614</b></td>
      <td style="color:orange;"><b>50 %</b></td>
    </tr>
  </tbody>
</table>

### 📝 Observaciones rápidas
- <b style="color:green;">EEGNet</b>: estable entre entrenamiento y validación, accuracy inicial moderada.  
- <b style="color:red;">ShallowConvNet</b>: train loss bajó, pero validación muy alta → sobreajuste evidente.  
- Estratificación desactivada (clases con <2 muestras).  
- Segmentación activada generó más muestras, pero con pocas epochs aún no se observa mejora clara.  


### 📝 Observaciones rápidas
- **EEGNet**: se mantiene estable entre entrenamiento y validación, con accuracy inicial moderada.  
- **ShallowConvNet**: aunque el train loss bajó, la validación sigue muy alta → sobreajuste evidente.  
- Estratificación desactivada (clases con <2 muestras).  
- Segmentación activada generó más muestras, pero con pocas epochs aún no se observa mejora clara.  


📝 Observaciones
Estratificación: desactivada (clases con <2 muestras).

Train loss decreciente, pero validación muy alta → indica sobreajuste o desbalance en el split.

Accuracy inicial baja, requiere más datos y epochs para estabilizar.

Pipeline estable, sin errores de ejecución.


📌 Próximos pasos
Ejecutar smoke tests en ambos modelos (EEGNet y ShallowConvNet) con el dataset BCI IV‑2a.

Documentar métricas iniciales en results/.

Ajustar hiperparámetros y preparar entrenamiento completo en todos los sujetos.

Migrar pipeline al dataset clínico CN/MCI/AD + tau.

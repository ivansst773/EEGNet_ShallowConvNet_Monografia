import os
import torch
import numpy as np
import mne
import scipy.io as sio
from sklearn.model_selection import train_test_split
from mne.filter import filter_data

def segment_trials(X, y, window_size=256, step=128):
    """
    Divide cada trial en ventanas más pequeñas.
    window_size: número de muestras por ventana
    step: desplazamiento entre ventanas
    """
    X_segments, y_segments = [], []
    for trial, label in zip(X, y):
        for start in range(0, trial.shape[-1] - window_size + 1, step):
            end = start + window_size
            X_segments.append(trial[:, start:end])
            y_segments.append(label)
    return np.array(X_segments), np.array(y_segments)

def load_bci_iv2a(subject=1, base_path="data/raw/BCI_IV_2a",
                  test_size=0.2, preload=True,
                  apply_filter=True, segment=False,
                  window_size=256, step=128,
                  stratify=True):
    """
    Carga datos de un sujeto del BCI Competition IV-2a.
    
    Parámetros:
    -----------
    subject : int
        Número de sujeto (1–9).
    base_path : str
        Carpeta base donde están los archivos .gdf y true_labels.
    test_size : float
        Proporción de validación en el split de entrenamiento.
    preload : bool
        Si True, carga todo en memoria.
    apply_filter : bool
        Si True, aplica filtro bandpass 4–40 Hz.
    segment : bool
        Si True, divide cada trial en ventanas más pequeñas.
    window_size : int
        Número de muestras por ventana (solo si segment=True).
    step : int
        Desplazamiento entre ventanas (solo si segment=True).
    stratify : bool
        Si True, usa stratify en el split (requiere >=2 muestras por clase).
        Si False, hace split aleatorio sin balance.

    Devuelve:
    ---------
    X_tr, y_tr, X_val, y_val, X_eval, y_eval : torch.Tensor
        Tensores listos para PyTorch con shape [N,1,chans,samples].
    chans, samples : int
        Número de canales y muestras detectados automáticamente.
    """

    # Archivos
    train_file = os.path.join(base_path, f"A0{subject}T.gdf")
    eval_file  = os.path.join(base_path, f"A0{subject}E.gdf")
    label_file = os.path.join(base_path, "true_labels", f"A0{subject}E.mat")

    for f in [train_file, eval_file, label_file]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Archivo no encontrado: {f}")

    # Entrenamiento
    raw_train = mne.io.read_raw_gdf(train_file, preload=preload)
    events_train, _ = mne.events_from_annotations(raw_train)
    picks = mne.pick_types(raw_train.info, eeg=True, exclude="bads")

    epochs_train = mne.Epochs(raw_train, events_train,
                              event_id=dict(left=1, right=2, foot=3, tongue=4),
                              tmin=0, tmax=4, proj=True, picks=picks,
                              baseline=None, preload=True)

    X_train = epochs_train.get_data()
    y_train = epochs_train.events[:, -1] - 1

    # Evaluación
    raw_eval = mne.io.read_raw_gdf(eval_file, preload=preload)
    labels = sio.loadmat(label_file)
    y_eval = labels["classlabel"].squeeze()

    events_eval, _ = mne.events_from_annotations(raw_eval)
    epochs_eval = mne.Epochs(raw_eval, events_eval,
                             event_id=dict(left=1, right=2, foot=3, tongue=4),
                             tmin=0, tmax=4, proj=True, picks=picks,
                             baseline=None, preload=True)

    X_eval = epochs_eval.get_data()

    # Detectar chans y samples
    chans   = X_train.shape[1]
    samples = X_train.shape[2]

    # Filtro bandpass opcional
    if apply_filter:
        X_train = filter_data(X_train, sfreq=raw_train.info['sfreq'], l_freq=4., h_freq=40.)
        X_eval  = filter_data(X_eval,  sfreq=raw_eval.info['sfreq'],  l_freq=4., h_freq=40.)

    # Normalización
    X_train = (X_train - X_train.mean(axis=-1, keepdims=True)) / X_train.std(axis=-1, keepdims=True)
    X_eval  = (X_eval  - X_eval.mean(axis=-1, keepdims=True))  / X_eval.std(axis=-1, keepdims=True)

    # Segmentación opcional
    if segment:
        X_train, y_train = segment_trials(X_train, y_train, window_size=window_size, step=step)
        X_eval,  y_eval  = segment_trials(X_eval,  y_eval,  window_size=window_size, step=step)
        samples = window_size  # ahora cada ventana tiene tamaño fijo

    # Split train/val
    if stratify and len(np.unique(y_train)) > 1:
        try:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=test_size, random_state=42, stratify=y_train
            )
        except ValueError:
            # fallback si alguna clase tiene <2 muestras
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=test_size, random_state=42
            )
    else:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=test_size, random_state=42
        )

    # Tensores
    def to_tensor(X, y):
        return torch.tensor(X, dtype=torch.float32).unsqueeze(1), torch.tensor(y, dtype=torch.long)

    X_tr, y_tr   = to_tensor(X_tr, y_tr)
    X_val, y_val = to_tensor(X_val, y_val)
    X_eval, y_eval = to_tensor(X_eval, y_eval)

    return X_tr, y_tr, X_val, y_val, X_eval, y_eval, chans, samples

# Smoke test
if __name__ == "__main__":
    X_tr, y_tr, X_val, y_val, X_eval, y_eval, chans, samples = load_bci_iv2a(
        subject=1, apply_filter=True, segment=True, stratify=False
    )
    print("Train:", X_tr.shape, y_tr.shape)
    print("Val:", X_val.shape, y_val.shape)
    print("Eval:", X_eval.shape, y_eval.shape)
    print("Chans:", chans, "Samples:", samples)

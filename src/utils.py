import os, random
import torch
import numpy as np
import mne
import scipy.io as sio
from sklearn.model_selection import train_test_split
from mne.filter import filter_data

def set_seed(seed: int = 42):
    """Fijar semillas para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def segment_trials(X, y, window_size=256, step=128):
    """Segmentar cada trial en ventanas más pequeñas."""
    X_segments, y_segments = [], []
    for trial, label in zip(X, y):
        for start in range(0, trial.shape[-1] - window_size + 1, step):
            end = start + window_size
            X_segments.append(trial[:, start:end])
            y_segments.append(label)
    return np.array(X_segments), np.array(y_segments)

def load_bci_iv2a_full(subject=1, base_path="data/raw/BCI_IV_2a",
                       test_size=0.2, preload=True,
                       apply_filter=True, segment=False,
                       window_size=256, step=128,
                       stratify=True):
    """Carga completa del dataset BCI Competition IV-2a para un sujeto."""
    train_file = os.path.join(base_path, f"A0{subject}T.gdf")
    eval_file  = os.path.join(base_path, f"A0{subject}E.gdf")
    label_file = os.path.join(base_path, "true_labels", f"A0{subject}E.mat")

    # === Entrenamiento ===
    raw_train = mne.io.read_raw_gdf(train_file, preload=preload)
    events_train, _ = mne.events_from_annotations(raw_train)
    picks = mne.pick_types(raw_train.info, eeg=True, exclude="bads")

    epochs_train = mne.Epochs(
        raw_train, events_train,
        event_id=dict(left=1, right=2, foot=3, tongue=4),
        tmin=0, tmax=4, proj=True, picks=picks,
        baseline=None, preload=True,
        event_repeated="merge"   # 👈 clave para evitar error en A04
    )
    X_train = epochs_train.get_data()
    y_train = epochs_train.events[:, -1] - 1

    # Filtrar etiquetas fuera de rango (0–3)
    mask_train = (y_train >= 0) & (y_train < 4)
    X_train, y_train = X_train[mask_train], y_train[mask_train]

    # === Evaluación ===
    raw_eval = mne.io.read_raw_gdf(eval_file, preload=preload)
    labels = sio.loadmat(label_file)
    y_eval = labels["classlabel"].squeeze()

    events_eval, _ = mne.events_from_annotations(raw_eval)
    epochs_eval = mne.Epochs(
        raw_eval, events_eval,
        event_id=dict(left=1, right=2, foot=3, tongue=4),
        tmin=0, tmax=4, proj=True, picks=picks,
        baseline=None, preload=True,
        event_repeated="merge"
    )
    X_eval = epochs_eval.get_data()

    # ⚠️ Alinear etiquetas con número de trials detectados
    min_len = min(len(X_eval), len(y_eval))
    X_eval, y_eval = X_eval[:min_len], y_eval[:min_len]

    # Filtrar etiquetas fuera de rango (0–3)
    mask_eval = (y_eval >= 0) & (y_eval < 4)
    X_eval, y_eval = X_eval[mask_eval], y_eval[mask_eval]

    # === Filtro band-pass opcional ===
    if apply_filter:
        X_train = filter_data(X_train, sfreq=raw_train.info['sfreq'], l_freq=4., h_freq=40.)
        X_eval  = filter_data(X_eval,  sfreq=raw_eval.info['sfreq'],  l_freq=4., h_freq=40.)

    # === Normalización trial-wise ===
    X_train = (X_train - X_train.mean(axis=-1, keepdims=True)) / X_train.std(axis=-1, keepdims=True)
    X_eval  = (X_eval  - X_eval.mean(axis=-1, keepdims=True))  / X_eval.std(axis=-1, keepdims=True)

    # === Segmentación opcional ===
    if segment:
        X_train, y_train = segment_trials(X_train, y_train, window_size, step)
        X_eval,  y_eval  = segment_trials(X_eval,  y_eval,  window_size, step)
        samples = window_size
    else:
        samples = X_train.shape[-1]

    # === División train/val con verificación de clases mínimas ===
    unique, counts = np.unique(y_train, return_counts=True)
    if stratify and np.all(counts >= 2):
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=test_size,
            random_state=42, stratify=y_train
        )
    else:
        print("[⚠️] Estratificación desactivada: al menos una clase tiene menos de 2 muestras.")
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=test_size,
            random_state=42
        )

    def to_tensor(X, y):
        return torch.tensor(X, dtype=torch.float32).unsqueeze(1), torch.tensor(y, dtype=torch.long)

    # ⚠️ Recalcular dimensiones finales
    chans = X_tr.shape[-2]

    return *to_tensor(X_tr, y_tr), *to_tensor(X_val, y_val), *to_tensor(X_eval, y_eval), chans, samples

# === Wrapper simple para el script de entrenamiento ===
def load_bci_iv2a(split="train", subject=1, **kwargs):
    X_tr, y_tr, X_val, y_val, X_eval, y_eval, chans, samples = load_bci_iv2a_full(subject=subject, **kwargs)
    if split == "train":
        return X_tr, y_tr
    elif split == "val":
        return X_val, y_val
    elif split == "eval":
        return X_eval, y_eval
    else:
        raise ValueError("split debe ser 'train', 'val' o 'eval'")

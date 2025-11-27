import os
import torch
import numpy as np
import mne
import pandas as pd
from sklearn.model_selection import train_test_split

def _normalize_trialwise(X: np.ndarray):
    """
    Normalización por trial (canal-wise): z-score en el eje temporal.
    Espera X con forma (n_trials, n_channels, n_times).
    """
    mean = X.mean(axis=-1, keepdims=True)
    std = X.std(axis=-1, keepdims=True)
    std[std == 0] = 1.0
    return (X - mean) / std

def _segment_trials(X: np.ndarray, y: np.ndarray, window_size=256, step=128):
    """
    Segmenta cada trial/registro en ventanas.
    X: (n_trials, n_channels, n_times)
    y: (n_trials,)
    """
    X_segments, y_segments = [], []
    for trial, label in zip(X, y):
        n_times = trial.shape[-1]
        for start in range(0, n_times - window_size + 1, step):
            end = start + window_size
            X_segments.append(trial[:, start:end])
            y_segments.append(label)
    return np.array(X_segments), np.array(y_segments)

def load_clinico_full(base_path="data/raw/CLINICO",
                      test_size=0.2, preload=True,
                      apply_filter=True, segment=False,
                      window_size=256, step=128,
                      stratify=True,
                      l_freq=1.0, h_freq=40.0):
    """
    Carga completa del dataset clínico OpenNeuro (ds004504) en formato BIDS.

    Soporta archivos .edf y .set (EEGLAB).
    """

    participants_file = os.path.join(base_path, "participants.tsv")
    if not os.path.isfile(participants_file):
        raise FileNotFoundError(f"No se encontró {participants_file}. Asegúrate de descargar y ubicar el dataset BIDS clínico.")

    df = pd.read_csv(participants_file, sep="\t")

    # Detectar columnas de ID y diagnóstico
    id_cols = ["participant_id", "participant", "subject_id", "sub"]
    diag_cols = ["diagnosis", "group", "Group", "label", "dx"]
    id_col = next((c for c in id_cols if c in df.columns), None)
    diag_col = next((c for c in diag_cols if c in df.columns), None)

    if id_col is None or diag_col is None:
        raise ValueError(f"participants.tsv debe contener columnas de ID y diagnóstico. Encontradas: {list(df.columns)}")

    # Mapeo diagnóstico -> índice
    diagnosis_map = {"CN": 0, "AD": 1, "FTD": 2}
    unique_diags = df[diag_col].astype(str).str.upper().unique()
    for d in unique_diags:
        if d not in diagnosis_map:
            diagnosis_map[d] = len(diagnosis_map)

    X_all, y_all = [], []
    sfreq_ref = None
    chans_ref = None

    for _, row in df.iterrows():
        subj_id = str(row[id_col])
        diagnosis = str(row[diag_col]).upper()

        # Buscar carpeta del sujeto
        subj_dir = os.path.join(base_path, f"sub-{subj_id}")
        eeg_dir = os.path.join(subj_dir, "eeg")

        # Buscar archivos .edf o .set
        eeg_files = []
        if os.path.isdir(eeg_dir):
            eeg_files = [f for f in os.listdir(eeg_dir) if f.lower().endswith((".edf", ".set"))]
        else:
            # Buscar en derivatives
            der_dir = os.path.join(base_path, "derivatives", f"sub-{subj_id}", "eeg")
            if os.path.isdir(der_dir):
                eeg_files = [f for f in os.listdir(der_dir) if f.lower().endswith((".edf", ".set"))]
                eeg_dir = der_dir

        if not eeg_files:
            continue

        eeg_file = os.path.join(eeg_dir, eeg_files[0])

        try:
            if eeg_file.endswith(".edf"):
                raw = mne.io.read_raw_edf(eeg_file, preload=preload, verbose="ERROR")
            elif eeg_file.endswith(".set"):
                raw = mne.io.read_raw_eeglab(eeg_file, preload=preload, verbose="ERROR")
        except Exception as e:
            print(f"[⚠️] Error leyendo {eeg_file}: {e}")
            continue

        # Filtrado opcional
        if apply_filter:
            try:
                raw.filter(l_freq, h_freq, fir_design="firwin", verbose="ERROR")
            except Exception as e:
                print(f"[⚠️] Error filtrando {eeg_file}: {e}")

        # Selección EEG
        picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
        data = raw.get_data(picks=picks)  # (n_channels, n_times)

        if sfreq_ref is None:
            sfreq_ref = raw.info.get("sfreq", None)
        if chans_ref is None:
            chans_ref = data.shape[0]

        X_all.append(data)
        y_all.append(diagnosis_map.get(diagnosis, -1))

    if len(X_all) == 0:
        raise RuntimeError("No se cargaron registros EEG (.edf o .set). Verifica rutas.")

    X_all = np.stack(X_all, axis=0)
    y_all = np.array(y_all)

    # Normalización por trial
    X_all = _normalize_trialwise(X_all)

    # Segmentación opcional
    if segment:
        X_all, y_all = _segment_trials(X_all, y_all, window_size=window_size, step=step)
        samples = window_size
    else:
        samples = X_all.shape[-1]

    # División train/val
    ok_strat = False
    if stratify:
        unique, counts = np.unique(y_all, return_counts=True)
        ok_strat = np.all(counts >= 2) and (len(unique) > 1)

    if ok_strat:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_all, y_all, test_size=test_size, random_state=42, stratify=y_all
        )
    else:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_all, y_all, test_size=test_size, random_state=42
        )

    # Eval vacío por ahora
    X_eval = np.empty((0, chans_ref, samples), dtype=np.float32)
    y_eval = np.empty((0,), dtype=np.int64)

    def to_tensor(X, y):
        return torch.tensor(X, dtype=torch.float32).unsqueeze(1), torch.tensor(y, dtype=torch.long)

    X_tr_t, y_tr_t = to_tensor(X_tr, y_tr)
    X_val_t, y_val_t = to_tensor(X_val, y_val)
    X_eval_t, y_eval_t = to_tensor(X_eval, y_eval)

    chans = chans_ref

    return X_tr_t, y_tr_t, X_val_t, y_val_t, X_eval_t, y_eval_t, chans, samples

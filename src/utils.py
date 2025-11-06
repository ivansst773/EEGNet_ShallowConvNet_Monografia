import os
import torch
import torch.nn as nn
import numpy as np
import mne
import scipy.io as sio
from sklearn.model_selection import train_test_split

# ============================================================
# 1. Loader para BCI Competition IV-2a
# ============================================================

def load_bci_iv2a(subject=1, base_path="data/raw/BCI_IV_2a", test_size=0.2, preload=True):
    """
    Carga datos de un sujeto del BCI Competition IV-2a.
    Devuelve X_train, y_train, X_val, y_val listos para PyTorch.
    """
    # Archivos
    train_file = os.path.join(base_path, f"A0{subject}T.gdf")
    eval_file  = os.path.join(base_path, f"A0{subject}E.gdf")
    label_file = os.path.join(base_path, "true_labels", f"A0{subject}E.mat")

    # -----------------------------
    # Entrenamiento
    # -----------------------------
    raw_train = mne.io.read_raw_gdf(train_file, preload=preload)
    events_train, _ = mne.events_from_annotations(raw_train)
    picks = mne.pick_types(raw_train.info, eeg=True, exclude="bads")

    epochs_train = mne.Epochs(raw_train, events_train,
                              event_id=dict(left=1, right=2, foot=3, tongue=4),
                              tmin=0, tmax=4, proj=True, picks=picks,
                              baseline=None, preload=True)

    X_train = epochs_train.get_data()  # [n_trials, n_channels, n_times]
    y_train = epochs_train.events[:, -1] - 1  # etiquetas 0–3

    # -----------------------------
    # Evaluación
    # -----------------------------
    raw_eval = mne.io.read_raw_gdf(eval_file, preload=preload)
    labels = sio.loadmat(label_file)
    y_eval = labels["classlabel"].squeeze()

    events_eval, _ = mne.events_from_annotations(raw_eval)
    epochs_eval = mne.Epochs(raw_eval, events_eval,
                             event_id=dict(left=1, right=2, foot=3, tongue=4),
                             tmin=0, tmax=4, proj=True, picks=picks,
                             baseline=None, preload=True)

    X_eval = epochs_eval.get_data()

    # -----------------------------
    # Normalización por trial
    # -----------------------------
    X_train = (X_train - X_train.mean(axis=-1, keepdims=True)) / X_train.std(axis=-1, keepdims=True)
    X_eval  = (X_eval  - X_eval.mean(axis=-1, keepdims=True))  / X_eval.std(axis=-1, keepdims=True)

    # -----------------------------
    # Split train/val
    # -----------------------------
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=test_size, random_state=42, stratify=y_train
    )

    # -----------------------------
    # Convertir a tensores
    # -----------------------------
    X_tr   = torch.tensor(X_tr, dtype=torch.float32).unsqueeze(1)  # [N,1,chans,samples]
    y_tr   = torch.tensor(y_tr, dtype=torch.long)
    X_val  = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
    y_val  = torch.tensor(y_val, dtype=torch.long)
    X_eval = torch.tensor(X_eval, dtype=torch.float32).unsqueeze(1)
    y_eval = torch.tensor(y_eval, dtype=torch.long)

    return X_tr, y_tr, X_val, y_val, X_eval, y_eval


# ============================================================
# 2. EEGNet
# ============================================================

class EEGNet(nn.Module):
    def __init__(self, nb_classes=4, Chans=22, Samples=256):
        super(EEGNet, self).__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(16)
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, (Chans, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.25)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.25)
        )
        # Capa final se define con dummy forward
        dummy = torch.zeros(1, 1, Chans, Samples)
        out = self._forward_features(dummy)
        in_feats = out.view(1, -1).size(1)
        self.classify = nn.Linear(in_feats, nb_classes)

    def _forward_features(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        return self.classify(x)


# ============================================================
# 3. ShallowConvNet
# ============================================================

class ShallowConvNet(nn.Module):
    def __init__(self, nb_classes=4, Chans=22, Samples=256):
        super(ShallowConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 40, (1, 25), bias=False)
        self.conv2 = nn.Conv2d(40, 40, (Chans, 1), bias=False)
        self.batchnorm = nn.BatchNorm2d(40)
        self.pool = nn.AvgPool2d((1, 75), stride=(1, 15))
        self.dropout = nn.Dropout(0.5)
        # Capa final con dummy forward
        dummy = torch.zeros(1, 1, Chans, Samples)
        out = self._forward_features(dummy)
        in_feats = out.view(1, -1).size(1)
        self.fc = nn.Linear(in_feats, nb_classes)

    def _forward_features(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batchnorm(x)
        x = torch.square(x)
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        x = self.dropout(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

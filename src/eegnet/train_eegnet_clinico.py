import argparse
import yaml
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import time   # ⏱️ añadido para medir tiempos

from src.clinical_loader import ClinicalEEGDataset
from src.models import EEGNet
from src.metrics import log_metrics

# -----------------------------
# Argumentos y configuración
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/clinico.yaml")
args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

epochs = config["training"]["epochs"]
batch_size = config["training"]["batch_size"]
learning_rate = config["training"]["learning_rate"]
dropout = config["eegnet"]["dropout"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Usando dispositivo: {device}")

# -----------------------------
# Cargar datos clínicos desde index.csv
# -----------------------------
index_file = "data/raw/CLINICO/processed/index.csv"
dataset = ClinicalEEGDataset(index_file=index_file)
print(f"[INFO] Dataset cargado con {len(dataset)} segmentos")

# Split train/val
train_size = int((1 - config["general"]["test_size"]) * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)
val_loader   = DataLoader(val_dataset, batch_size=batch_size,
                          num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)

# Obtener dimensiones de un ejemplo
sample_data, _ = dataset[0]
chans, samples = sample_data.shape

# -----------------------------
# Definir modelo
# -----------------------------
model = EEGNet(
    n_channels=chans,
    n_times=samples,
    n_classes=len(dataset.label_to_int),
    dropout=dropout
).to(device)

print(f"[INFO] Modelo en: {next(model.parameters()).device}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# AMP scaler
scaler = torch.amp.GradScaler(device="cuda")

# -----------------------------
# Entrenamiento + Validación por época
# -----------------------------
epoch_times = []  # lista para guardar tiempos por época

for epoch in range(epochs):
    start_time = time.time()   # ⏱️ inicio época

    model.train()
    train_losses = []
    for batch_idx, (xb, yb) in enumerate(train_loader):
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

        if batch_idx == 0 and epoch == 0:
            print(f"[DEBUG] xb: {xb.device}, yb: {yb.device}, modelo: {next(model.parameters()).device}")

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            outputs = model(xb.unsqueeze(1))
            loss = criterion(outputs, yb)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_losses.append(loss.item())

        if batch_idx % 50 == 0:
            print(f"[Epoch {epoch+1}] Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

    # ---- Validación ----
    model.eval()
    val_losses, val_preds, val_targets = [], [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                val_outputs = model(xb.unsqueeze(1))
                val_loss = criterion(val_outputs, yb)
            val_losses.append(val_loss.item())
            _, preds = torch.max(val_outputs, 1)
            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(yb.cpu().numpy())

    val_loss_mean = np.mean(val_losses)
    val_acc = (np.array(val_preds) == np.array(val_targets)).mean()

    end_time = time.time()   # ⏱️ fin época
    epoch_duration = end_time - start_time
    epoch_times.append(epoch_duration)

    print(
        f"[EPOCH {epoch+1}/{epochs}] "
        f"Train Loss: {np.mean(train_losses):.4f} | "
        f"Val Loss: {val_loss_mean:.4f} | "
        f"Val Acc: {val_acc*100:.2f}% | "
        f"Tiempo: {epoch_duration:.2f} segundos"
    )

# -----------------------------
# Log de métricas finales
# -----------------------------
log_metrics(
    modelo="EEGNet-Clinico",
    sujeto="CLINICO",
    epochs=epochs,
    batch_size=batch_size,
    learning_rate=learning_rate,
    dropout=dropout,
    train_loss=np.mean(train_losses),
    val_loss=val_loss_mean,
    val_accuracy=val_acc * 100,
    dispositivo=str(device),
    observaciones=f"Entrenamiento clínico con AMP (mixed precision). "
                  f"Tiempos por época: {epoch_times}, promedio: {np.mean(epoch_times):.2f} s"
)

if config["training"].get("save_model", False):
    os.makedirs("results/modelos", exist_ok=True)
    model_path = "results/modelos/EEGNet_Clinico.pth"
    torch.save(model.state_dict(), model_path)
    print(f"[INFO] Modelo guardado en {model_path}")

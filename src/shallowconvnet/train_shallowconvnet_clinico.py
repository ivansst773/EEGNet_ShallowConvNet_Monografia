import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from src.utils import load_clinico_full
from src.models import ShallowConvNet
from src.metrics import log_metrics

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/clinico.yaml")
args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

epochs = config["training"]["epochs"]
batch_size = config["training"]["batch_size"]
learning_rate = config["training"]["learning_rate"]
dropout = config["shallowconvnet"]["dropout"]
segment = config["general"]["segment"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar datos clínicos
X_tr, y_tr, X_val, y_val, chans, samples = load_clinico_full(
    base_path="data/raw/CLINICO",
    apply_filter=config["general"]["apply_filter"],
    segment=segment
)
X_tr, y_tr = X_tr.to(device), y_tr.to(device)
X_val, y_val = X_val.to(device), y_val.to(device)

# Crear DataLoaders
train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

# Definir modelo
model = ShallowConvNet(n_channels=chans, n_times=samples, n_classes=3, dropout=dropout).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Entrenamiento
for epoch in range(epochs):
    model.train()
    train_losses = []
    for xb, yb in train_loader:
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    print(f"Epoch {epoch+1}, Train Loss: {np.mean(train_losses):.4f}")

# Validación
model.eval()
val_losses, val_preds, val_targets = [], [], []
with torch.no_grad():
    for xb, yb in val_loader:
        val_outputs = model(xb)
        val_loss = criterion(val_outputs, yb)
        val_losses.append(val_loss.item())
        _, preds = torch.max(val_outputs, 1)
        val_preds.extend(preds.cpu().numpy())
        val_targets.extend(yb.cpu().numpy())

val_acc = (np.array(val_preds) == np.array(val_targets)).mean()
print(f"Validation Loss: {np.mean(val_losses):.4f}, Accuracy: {val_acc*100:.2f}%")

# Log de métricas
log_metrics(
    modelo="ShallowConvNet-Clinico",
    sujeto="CLINICO",
    epochs=epochs,
    batch_size=batch_size,
    learning_rate=learning_rate,
    dropout=dropout,
    train_loss=np.mean(train_losses),
    val_loss=np.mean(val_losses),
    val_accuracy=val_acc * 100,
    dispositivo=str(device),
    observaciones="Entrenamiento clínico con segmentación" if segment else "Entrenamiento clínico sin segmentación"
)

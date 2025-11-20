import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from src.utils import load_bci_iv2a_full
from src.models import EEGNet
from src.metrics import log_metrics

# 1. Argumentos
parser = argparse.ArgumentParser()
parser.add_argument("--subject", type=int, required=True)
parser.add_argument("--config", type=str, default="configs/bci_iv2a.yaml")
args = parser.parse_args()

# 2. Cargar configuración YAML
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

epochs = config["training"]["epochs"]
batch_size = config["training"]["batch_size"]
learning_rate = config["training"]["learning_rate"]
dropout = config["eegnet"]["dropout"]
segment = config["general"]["segment"]

# 3. Dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando dispositivo:", device)

# 4. Datos
X_tr, y_tr, X_val, y_val, X_eval, y_eval, chans, samples = load_bci_iv2a_full(
    subject=args.subject,
    apply_filter=config["general"]["apply_filter"],
    segment=segment
)
X_tr, y_tr = X_tr.to(device), y_tr.to(device)
X_val, y_val = X_val.to(device), y_val.to(device)

# 5. Modelo
model = EEGNet(n_channels=chans, n_times=samples, n_classes=4, dropout=dropout).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 6. Entrenamiento
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_tr[:batch_size])
    loss = criterion(outputs, y_tr[:batch_size])
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 7. Validación
with torch.no_grad():
    val_outputs = model(X_val[:batch_size])
    val_loss = criterion(val_outputs, y_val[:batch_size])
    _, preds = torch.max(val_outputs, 1)
    acc = (preds == y_val[:batch_size]).float().mean()
    print(f"Validation Loss: {val_loss.item():.4f}, Accuracy: {acc.item()*100:.2f}%")

# 8. Logging
log_metrics(
    modelo="EEGNet",
    sujeto=f"A0{args.subject}",
    epochs=epochs,
    batch_size=batch_size,
    learning_rate=learning_rate,
    dropout=dropout,
    train_loss=loss.item(),
    val_loss=val_loss.item(),
    val_accuracy=acc.item() * 100,
    dispositivo=str(device),
    observaciones="Entrenamiento automático con segmentación" if segment else "Entrenamiento automático sin segmentación"
)

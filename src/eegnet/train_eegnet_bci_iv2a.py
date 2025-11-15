import torch
import torch.nn as nn
import torch.optim as optim
from src.utils import load_bci_iv2a_full
from src.models import EEGNet
from src.metrics import log_metrics

# 1. Selección de dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando dispositivo:", device)

# 2. Cargar datos
X_tr, y_tr, X_val, y_val, X_eval, y_eval, chans, samples = load_bci_iv2a_full(
    subject=1, apply_filter=True
)

# Mover datos al dispositivo
X_tr, y_tr = X_tr.to(device), y_tr.to(device)
X_val, y_val = X_val.to(device), y_val.to(device)
X_eval, y_eval = X_eval.to(device), y_eval.to(device)

# 3. Definir modelo (usa los nombres correctos)
model = EEGNet(n_channels=chans, n_times=samples, n_classes=4).to(device)

# 4. Configuración de entrenamiento
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Entrenamiento rápido (smoke test)
for epoch in range(2):
    optimizer.zero_grad()
    outputs = model(X_tr[:50])
    loss = criterion(outputs, y_tr[:50])
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 6. Validación rápida
with torch.no_grad():
    val_outputs = model(X_val[:50])
    val_loss = criterion(val_outputs, y_val[:50])
    _, preds = torch.max(val_outputs, 1)
    acc = (preds == y_val[:50]).float().mean()
    print(f"Validation Loss: {val_loss.item():.4f}, Accuracy: {acc.item()*100:.2f}%")


# Después de calcular val_loss y acc
log_metrics(
    modelo="EEGNet",
    sujeto="A01",
    epochs=2,
    batch_size=16,
    learning_rate=0.001,
    dropout=0.25,
    train_loss=loss.item(),
    val_loss=val_loss.item(),
    val_accuracy=acc.item() * 100,
    dispositivo=str(device),
    observaciones="Smoke test con 50 trials, estratificación desactivada"
)

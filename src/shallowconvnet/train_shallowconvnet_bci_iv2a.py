import torch
import torch.nn as nn
import torch.optim as optim
from src.utils import load_bci_iv2a
from src.models import ShallowConvNet

# 1. Selección de dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando dispositivo:", device)

# 2. Cargar datos
X_tr, y_tr, X_val, y_val, X_eval, y_eval, chans, samples = load_bci_iv2a(subject=1, apply_filter=True)

# Mover datos al dispositivo
X_tr, y_tr = X_tr.to(device), y_tr.to(device)
X_val, y_val = X_val.to(device), y_val.to(device)
X_eval, y_eval = X_eval.to(device), y_eval.to(device)

# 3. Definir modelo
model = ShallowConvNet(nb_classes=4, Chans=chans, Samples=samples).to(device)

# 4. Configuración
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Entrenamiento rápido
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

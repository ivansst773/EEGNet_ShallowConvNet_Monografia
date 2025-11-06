import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import load_bci_iv2a, EEGNet

def train():
    # Hiperparámetros
    batch_size = 64
    lr = 1e-3
    epochs = 50

    # Dataset
    X_train, y_train, X_val, y_val = load_bci_iv2a()
    train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(list(zip(X_val, y_val)), batch_size=batch_size)

    # Modelo
    model = EEGNet(nb_classes=len(set(y_train)))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Entrenamiento
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

        # Validación
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                outputs = model(X)
                _, predicted = torch.max(outputs, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        print(f"Validation Accuracy: {100*correct/total:.2f}%")

    torch.save(model.state_dict(), "eegnet_bci_iv2a.pth")

if __name__ == "__main__":
    train()

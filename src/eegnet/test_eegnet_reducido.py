import torch
from utils import load_bci_iv2a, EEGNet

def test():
    # Cargar datos de validación
    _, _, X_val, y_val = load_bci_iv2a()

    # Cargar modelo
    model = EEGNet(nb_classes=len(set(y_val)))
    model.load_state_dict(torch.load("eegnet_bci_iv2a.pth"))
    model.eval()

    # Evaluación
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in zip(X_val, y_val):
            X = X.unsqueeze(0)  # batch de 1
            outputs = model(X)
            _, predicted = torch.max(outputs, 1)
            total += 1
            correct += (predicted == y).item()

    print(f"Test Accuracy: {100*correct/total:.2f}%")

if __name__ == "__main__":
    test()

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from utils import EEGNet  # reutilizamos tu EEGNet

# Configuración
batch_size = 64
epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(root="../data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="../data", train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Adaptar EEGNet: entrada [batch, 1, chans, samples]
# Para MNIST: chans=28, samples=28
model = EEGNet(nb_classes=10, Chans=28, Samples=28).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenamiento
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        # Ajustar forma: [batch, 1, chans, samples]
        data = data.view(-1, 1, 28, 28)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}")

    # Evaluación
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(-1, 1, 28, 28)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    acc = 100. * correct / len(test_loader.dataset)
    print(f"Test Accuracy: {acc:.2f}%")

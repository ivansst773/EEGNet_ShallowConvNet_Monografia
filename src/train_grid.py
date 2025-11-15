# src/train_grid.py
import os
import time
import yaml
import torch
import random
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim

from models import EEGNet, ShallowConvNet
from utils import load_bci_iv2a, set_seed  # Ajusta si tus funciones tienen otro nombre

def make_dataloaders(batch_size: int):
    # Carga datos ya preprocesados (X: [N, C, T], y: [N])
    X_train, y_train = load_bci_iv2a(split='train')
    X_eval, y_eval = load_bci_iv2a(split='eval')

    # Tensores
    X_train = torch.as_tensor(X_train, dtype=torch.float32)
    y_train = torch.as_tensor(y_train, dtype=torch.long)
    X_eval = torch.as_tensor(X_eval, dtype=torch.float32)
    y_eval = torch.as_tensor(y_eval, dtype=torch.long)

    # Datasets y loaders
    train_ds = TensorDataset(X_train, y_train)
    eval_ds = TensorDataset(X_eval, y_eval)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, eval_loader, X_train.shape[1], X_train.shape[2]  # C, T

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_model(model_name: str, n_channels: int, n_times: int, n_classes: int, dropout: float):
    if model_name.lower() == 'eegnet':
        model = EEGNet(n_channels=n_channels, n_times=n_times, n_classes=n_classes, dropout=dropout)
    elif model_name.lower() == 'shallowconvnet':
        model = ShallowConvNet(n_channels=n_channels, n_times=n_times, n_classes=n_classes, dropout=dropout)
    else:
        raise ValueError(f"Modelo desconocido: {model_name}")
    return model

def train_one_config(cfg, results_dir):
    set_seed(cfg['seed'])
    device = get_device()

    train_loader, eval_loader, n_channels, n_times = make_dataloaders(cfg['batch_size'])
    n_classes = cfg.get('n_classes', 2)  # Ajusta según tu dataset

    model = build_model(cfg['model'], n_channels, n_times, n_classes, cfg['dropout']).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

    # Scheduler opcional
    scheduler = None
    if cfg['scheduler'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg['step_size'], gamma=cfg['gamma'])
    elif cfg['scheduler'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=cfg['gamma'], patience=cfg['patience'])

    metrics = {'train_loss': [], 'eval_acc': []}
    best_acc = 0.0

    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        epoch_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        metrics['train_loss'].append(avg_loss)

        # Evaluación
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in eval_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        eval_acc = correct / total if total > 0 else 0.0
        metrics['eval_acc'].append(eval_acc)

        # Actualizar scheduler
        if scheduler is not None:
            if cfg['scheduler'] == 'ReduceLROnPlateau':
                scheduler.step(eval_acc)
            else:
                scheduler.step()

        print(f"[{cfg['model']}] Epoch {epoch}/{cfg['epochs']} | loss={avg_loss:.4f} | acc={eval_acc:.3f}")

        # Guardar mejor modelo
        if eval_acc > best_acc:
            best_acc = eval_acc
            ckpt_path = os.path.join(results_dir, 'models', f"{cfg['run_name']}_best.pth")
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save({'model_state': model.state_dict(),
                        'cfg': cfg,
                        'best_acc': best_acc}, ckpt_path)

    # Guardar métricas en YAML
    out_path = os.path.join(results_dir, 'metrics', f"{cfg['run_name']}.yaml")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        yaml.dump({'cfg': cfg, 'metrics': metrics, 'best_acc': best_acc}, f)

    return best_acc, metrics

def main():
    # Reproducibilidad general
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    results_dir = 'results'  # respetando tu .gitignore
    # Define tu grid de hiperparámetros
    grids = [
        {
            'run_name': 'eegnet_bs32_lr1e-3_ep20_wd1e-4',
            'model': 'EEGNet',
            'batch_size': 32,
            'lr': 1e-3,
            'epochs': 20,
            'weight_decay': 1e-4,
            'dropout': 0.25,
            'scheduler': 'StepLR',  # 'None' | 'StepLR' | 'ReduceLROnPlateau'
            'step_size': 5,
            'gamma': 0.5,
            'patience': 3,
            'seed': 42,
            'n_classes': 2
        },
        {
            'run_name': 'shallow_bs64_lr5e-4_ep30_wd5e-5',
            'model': 'ShallowConvNet',
            'batch_size': 64,
            'lr': 5e-4,
            'epochs': 30,
            'weight_decay': 5e-5,
            'dropout': 0.5,
            'scheduler': 'ReduceLROnPlateau',
            'step_size': 0,  # no aplica
            'gamma': 0.5,
            'patience': 2,
            'seed': 123,
            'n_classes': 2
        },
    ]

    summary = []
    start = time.time()
    for cfg in grids:
        acc, metrics = train_one_config(cfg, results_dir)
        summary.append({'run_name': cfg['run_name'], 'best_acc': acc})
    dur = time.time() - start

    # Guardar resumen
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'summary.yaml'), 'w') as f:
        yaml.dump({'summary': summary, 'duration_sec': dur}, f)
    print("Resumen:", summary)

if __name__ == '__main__':
    main()

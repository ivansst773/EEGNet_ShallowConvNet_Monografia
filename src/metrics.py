import csv
import os
from datetime import datetime

def log_metrics(
    modelo: str,
    sujeto: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    dropout: float,
    train_loss: float,
    val_loss: float,
    val_accuracy: float,
    dispositivo: str,
    observaciones: str = ""
):
    # Ruta al archivo CSV
    csv_path = os.path.join("results", "tablas", "metrics.csv")

    # Crear carpeta si no existe
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Encabezados
    headers = [
        "fecha", "modelo", "sujeto", "epochs", "batch_size", "learning_rate",
        "dropout", "train_loss", "val_loss", "val_accuracy", "dispositivo", "observaciones"
    ]

    # Fila de datos
    row = [
        datetime.now().strftime("%Y-%m-%d"),
        modelo,
        sujeto,
        epochs,
        batch_size,
        learning_rate,
        dropout,
        f"{train_loss:.4f}",
        f"{val_loss:.4f}",
        f"{val_accuracy:.2f}",
        dispositivo,
        observaciones
    ]

    # Escribir en el CSV (crear si no existe)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(row)

    print(f"[✅] Métricas registradas en {csv_path}")

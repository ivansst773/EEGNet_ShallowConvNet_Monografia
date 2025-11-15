import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

def plot_metrics(csv_path="results/tablas/metrics.csv"):
    # Leer el archivo CSV
    df = pd.read_csv(csv_path)

    # Crear carpeta de salida si no existe
    os.makedirs("results/figuras", exist_ok=True)

    # Fecha actual para nombrar archivos
    fecha = datetime.now().strftime("%Y-%m-%d")

    # Iterar por cada modelo registrado en metrics.csv
    for modelo in df["modelo"].unique():
        subset = df[df["modelo"] == modelo]

        # --- Gráfica de Loss ---
        plt.figure(figsize=(8,5))
        plt.bar(["Train Loss"], subset["train_loss"].mean(), label="Train Loss")
        plt.bar(["Val Loss"], subset["val_loss"].mean(), label="Val Loss")
        plt.ylabel("Loss")
        plt.title(f"{modelo} - Train vs Val Loss")
        plt.legend()
        plt.grid(True, axis="y")
        filename_loss = f"results/figuras/{modelo}_{fecha}_loss.png"
        plt.savefig(filename_loss)
        plt.close()

        # --- Gráfica de Accuracy ---
        plt.figure(figsize=(8,5))
        plt.bar([modelo], subset["val_accuracy"].mean())
        plt.ylabel("Accuracy (%)")
        plt.title(f"{modelo} - Validation Accuracy")
        plt.grid(True, axis="y")
        filename_acc = f"results/figuras/{modelo}_{fecha}_accuracy.png"
        plt.savefig(filename_acc)
        plt.close()

        print(f"[✅] Gráficas guardadas: {filename_loss}, {filename_acc}")

if __name__ == "__main__":
    plot_metrics()

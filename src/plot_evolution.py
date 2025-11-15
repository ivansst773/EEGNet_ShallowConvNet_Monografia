import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_evolution(csv_path="results/tablas/metrics.csv"):
    df = pd.read_csv(csv_path)

    os.makedirs("results/figuras", exist_ok=True)

    # Convertir fecha a datetime para graficar en orden
    df["fecha"] = pd.to_datetime(df["fecha"])

    # --- Evolución de Loss ---
    plt.figure(figsize=(10,5))
    for model in df["modelo"].unique():
        subset = df[df["modelo"] == model]
        plt.plot(subset["fecha"], subset["train_loss"], marker="o", label=f"{model} Train Loss")
        plt.plot(subset["fecha"], subset["val_loss"], marker="x", label=f"{model} Val Loss")
    plt.title("Evolución de Loss por modelo")
    plt.xlabel("Fecha")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/figuras/evolucion_loss.png")
    plt.close()

    # --- Evolución de Accuracy ---
    plt.figure(figsize=(10,5))
    for model in df["modelo"].unique():
        subset = df[df["modelo"] == model]
        plt.plot(subset["fecha"], subset["val_accuracy"], marker="s", label=f"{model} Val Accuracy")
    plt.title("Evolución de Accuracy por modelo")
    plt.xlabel("Fecha")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/figuras/evolucion_accuracy.png")
    plt.close()

    print("[✅] Gráficas de evolución guardadas en results/figuras/")

if __name__ == "__main__":
    plot_evolution()

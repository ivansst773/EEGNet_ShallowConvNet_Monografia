import pandas as pd
import matplotlib.pyplot as plt
import os

# Ruta de resultados
metrics_path = "results/tablas/metrics.csv"
figuras_path = "results/figuras"

# Crear carpeta si no existe
os.makedirs(figuras_path, exist_ok=True)

# Cargar métricas
df = pd.read_csv(metrics_path)

# Lista de sujetos
sujetos = df["sujeto"].unique()

# =========================
# 🔹 Gráficas por sujeto
# =========================
for sujeto in sujetos:
    df_sujeto = df[df["sujeto"] == sujeto]

    # Graficar pérdidas
    plt.figure(figsize=(10,5))
    for modelo in df_sujeto["modelo"].unique():
        subset = df_sujeto[df_sujeto["modelo"] == modelo]
        plt.plot(subset.index, subset["train_loss"], marker="o", label=f"{modelo} - Train Loss")
        plt.plot(subset.index, subset["val_loss"], marker="x", label=f"{modelo} - Val Loss")

    plt.title(f"Evolución de Loss - Sujeto {sujeto}")
    plt.xlabel("Experimento")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(figuras_path, f"{sujeto}_loss.png"))
    plt.close()

    # Graficar accuracy
    plt.figure(figsize=(10,5))
    for modelo in df_sujeto["modelo"].unique():
        subset = df_sujeto[df_sujeto["modelo"] == modelo]
        plt.plot(subset.index, subset["val_accuracy"], marker="s", label=f"{modelo} - Val Accuracy")

    plt.title(f"Evolución de Accuracy - Sujeto {sujeto}")
    plt.xlabel("Experimento")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(figuras_path, f"{sujeto}_accuracy.png"))
    plt.close()

    print(f"[✅] Gráficas guardadas para sujeto {sujeto} en {figuras_path}")

# =========================
# 🔹 Gráficas globales
# =========================

# Promedio de accuracy por modelo en todos los sujetos
df_global = df.groupby("modelo")["val_accuracy"].mean().reset_index()

plt.figure(figsize=(8,6))
plt.bar(df_global["modelo"], df_global["val_accuracy"], color=["#1f77b4", "#ff7f0e"])
plt.title("Promedio de Accuracy por Modelo (todos los sujetos)")
plt.ylabel("Accuracy promedio (%)")
plt.grid(axis="y")
plt.savefig(os.path.join(figuras_path, "global_accuracy.png"))
plt.close()

print("[🎯] Gráfica global de accuracy guardada en results/figuras/global_accuracy.png")

# Promedio de loss por modelo
df_global_loss = df.groupby("modelo")["val_loss"].mean().reset_index()

plt.figure(figsize=(8,6))
plt.bar(df_global_loss["modelo"], df_global_loss["val_loss"], color=["#2ca02c", "#d62728"])
plt.title("Promedio de Validation Loss por Modelo (todos los sujetos)")
plt.ylabel("Validation Loss promedio")
plt.grid(axis="y")
plt.savefig(os.path.join(figuras_path, "global_val_loss.png"))
plt.close()

print("[🎯] Gráfica global de validation loss guardada en results/figuras/global_val_loss.png")

print("[🚀] Todas las gráficas por sujeto y globales generadas correctamente.")

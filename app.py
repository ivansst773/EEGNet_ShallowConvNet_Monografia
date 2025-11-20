import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

# =========================
# 🔹 Rutas
# =========================
metrics_path = "results/tablas/metrics.csv"
figuras_path = "results/figuras"

# =========================
# 🔹 Cargar métricas
# =========================
df = pd.read_csv(metrics_path)

st.title("🧠 Proyecto EEGNet & ShallowConvNet - BCI IV 2a")

# =========================
# 🔹 Selector de sujeto y modelo
# =========================
sujeto = st.selectbox("Selecciona sujeto:", df["sujeto"].unique())
modelo = st.selectbox("Selecciona modelo:", df["modelo"].unique())

# Filtrar datos
df_filtrado = df[(df["sujeto"] == sujeto) & (df["modelo"] == modelo)]

st.subheader(f"📊 Métricas - Sujeto {sujeto}, Modelo {modelo}")
st.dataframe(df_filtrado)

# =========================
# 🔹 Mostrar gráficas por sujeto
# =========================
loss_file = os.path.join(figuras_path, f"{sujeto}_loss.png")
acc_file = os.path.join(figuras_path, f"{sujeto}_accuracy.png")

if os.path.exists(loss_file):
    st.subheader("📉 Evolución de Loss")
    st.image(loss_file)

if os.path.exists(acc_file):
    st.subheader("📈 Evolución de Accuracy")
    st.image(acc_file)

# =========================
# 🔹 Gráficas globales
# =========================
st.subheader("🌍 Comparativas globales")

global_acc_file = os.path.join(figuras_path, "global_accuracy.png")
global_loss_file = os.path.join(figuras_path, "global_val_loss.png")

if os.path.exists(global_acc_file):
    st.image(global_acc_file, caption="Promedio de Accuracy por modelo")

if os.path.exists(global_loss_file):
    st.image(global_loss_file, caption="Promedio de Validation Loss por modelo")

# =========================
# 🔹 Gráfico comparativo global (EEGNet vs ShallowConvNet)
# =========================
st.subheader("📊 Accuracy comparativo por sujeto")

# Filtrar solo entrenamientos completos (ej. 50 epochs con segmentación)
df_completo = df[(df["epochs"] >= 50)]

# Pivotear para tener columnas EEGNet y ShallowConvNet
df_pivot = df_completo.pivot_table(
    index="sujeto",
    columns="modelo",
    values="val_accuracy",
    aggfunc="mean"
).reset_index()

# Crear gráfico de barras agrupadas
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.35
index = range(len(df_pivot))

ax.bar([i - bar_width/2 for i in index], df_pivot["EEGNet"], bar_width, label="EEGNet", color="#1f77b4")
ax.bar([i + bar_width/2 for i in index], df_pivot["ShallowConvNet"], bar_width, label="ShallowConvNet", color="#ff7f0e")

ax.set_xlabel("Sujeto", fontsize=12)
ax.set_ylabel("Accuracy de Validación (%)", fontsize=12)
ax.set_title("Comparación Global de Accuracy por Sujeto", fontsize=14, weight='bold')
ax.set_xticks(index)
ax.set_xticklabels(df_pivot["sujeto"])
ax.legend()

st.pyplot(fig)

# =========================
# 🔹 Evolución dinámica de métricas por epochs
# =========================
st.subheader("📈 Evolución dinámica de Accuracy y Loss")

# Filtrar datos del sujeto y modelo seleccionados
df_epochs = df[(df["sujeto"] == sujeto) & (df["modelo"] == modelo)]

if not df_epochs.empty:
    # Crear gráfico de evolución
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax[0].plot(df_epochs["epochs"], df_epochs["val_accuracy"], marker="o", color="#1f77b4")
    ax[0].set_title(f"Evolución de Accuracy - {modelo} {sujeto}")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Validation Accuracy (%)")

    # Loss
    ax[1].plot(df_epochs["epochs"], df_epochs["val_loss"], marker="o", color="#ff7f0e")
    ax[1].set_title(f"Evolución de Loss - {modelo} {sujeto}")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Validation Loss")

    st.pyplot(fig)
else:
    st.info("No hay datos de evolución disponibles para este sujeto/modelo.")

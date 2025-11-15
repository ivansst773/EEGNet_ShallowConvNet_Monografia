import streamlit as st
import pandas as pd
import os

# Rutas
metrics_path = "results/tablas/metrics.csv"
figuras_path = "results/figuras"

# Cargar métricas
df = pd.read_csv(metrics_path)

st.title("🧠 Proyecto EEGNet & ShallowConvNet - BCI IV 2a")

# Selector de sujeto y modelo
sujeto = st.selectbox("Selecciona sujeto:", df["sujeto"].unique())
modelo = st.selectbox("Selecciona modelo:", df["modelo"].unique())

# Filtrar datos
df_filtrado = df[(df["sujeto"] == sujeto) & (df["modelo"] == modelo)]

st.subheader(f"📊 Métricas - Sujeto {sujeto}, Modelo {modelo}")
st.dataframe(df_filtrado)

# Mostrar gráficas por sujeto
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

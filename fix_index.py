import pandas as pd
import os

# Ruta al index.csv original
index_path = "data/raw/CLINICO/processed/index.csv"

# Leer el archivo
df = pd.read_csv(index_path)

# Forzar etiquetas a string
df["label"] = df["label"].astype(str)

# Corregir rutas: anteponer carpeta base si no está incluida
def fix_path(path):
    if not path.startswith("data/raw/CLINICO/processed/"):
        return os.path.join("data/raw/CLINICO", path)
    return path

df["file"] = df["file"].apply(fix_path)

# Guardar nuevo index.csv corregido
df.to_csv(index_path, index=False)

print("[INFO] index.csv corregido y guardado en:", index_path)
print(df.head())

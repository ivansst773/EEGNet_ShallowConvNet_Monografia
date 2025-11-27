import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os

class ClinicalEEGDataset(Dataset):
    def __init__(self, index_file, base_path="data/raw/CLINICO",
                 label_map=None, transform=None,
                 segment=True, window_size=256, step=128):
        # Leer index.csv
        self.index = pd.read_csv(index_file)
        self.base_path = os.path.normpath(base_path)
        self.label_map = label_map
        self.transform = transform
        self.segment = segment
        self.window_size = window_size
        self.step = step

        # Forzar etiquetas a string y limpiar NaN
        self.index["label"] = self.index["label"].astype(str)
        self.index["label"] = self.index["label"].replace({"nan": "Unknown"})

        # Mapear etiquetas si se pasa diccionario
        if self.label_map:
            mapped = self.index["label"].map(self.label_map)
            self.index["label"] = mapped.fillna(self.index["label"])

        # Convertir etiquetas a enteros
        unique_labels = sorted(self.index["label"].unique().tolist())
        self.label_to_int = {label: i for i, label in enumerate(unique_labels)}
        self.index["label_id"] = self.index["label"].map(self.label_to_int)

        # Precomputar lista de muestras
        self.samples = []
        for _, row in self.index.iterrows():
            file_path = self._resolve_path(row["file"])
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"[ERROR] No se encontró el archivo: {file_path}")

            # 👇 usamos memmap para no cargar todo en RAM
            data = np.load(file_path, mmap_mode="r")  # (n_channels, n_samples)
            label_id = int(row["label_id"])

            if self.segment:
                n = data.shape[1]
                for start in range(0, n - self.window_size + 1, self.step):
                    self.samples.append((file_path, label_id, start))
            else:
                self.samples.append((file_path, label_id, None))

    def _resolve_path(self, path_str: str) -> str:
        p = os.path.normpath(path_str)
        if p.startswith(self.base_path):
            return p
        if os.path.isabs(p):
            return p
        return os.path.normpath(os.path.join(self.base_path, p))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label_id, start = self.samples[idx]
        # 👇 memmap también aquí
        data = np.load(file_path, mmap_mode="r")

        if start is not None:
            data = data[:, start:start + self.window_size]

        if self.transform:
            data = self.transform(data)

        x = torch.tensor(data, dtype=torch.float32)
        y = torch.tensor(label_id, dtype=torch.long)
        return x, y

from src.clinical_loader import ClinicalEEGDataset
from collections import Counter

label_map = {"A": "CN", "F": "AD"}  # ajusta según tu dataset

dataset = ClinicalEEGDataset(
    index_file="data/raw/CLINICO/processed/index.csv",
    base_path="data/raw/CLINICO",
    label_map=label_map,
    segment=True,
    window_size=256,
    step=128
)

print("Número total de segmentos:", len(dataset))

# Usar directamente la lista precomputada de samples
labels = [sample[1] for sample in dataset.samples]  # sample = (file_path, label_id, start)
conteo = Counter(labels)

print("\nSegmentos por clase:")
for clase_id, cantidad in conteo.items():
    print(f"Clase {clase_id}: {cantidad} segmentos")

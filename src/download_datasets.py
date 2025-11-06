import os
import mne

def download_eegbci():
    """
    Descarga EEGBCI (PhysioNet) automáticamente con MNE.
    """
    print("Descargando EEGBCI (PhysioNet)...")
    # Ejemplo: sujeto 1, runs de motor imagery (3=izquierda, 7=derecha, 11=manos/pies)
    files = mne.datasets.eegbci.load_data(1, runs=[3, 7, 11])
    print("Archivos descargados:", files)

def prepare_bci_iv2a():
    """
    Prepara la carpeta esperada para BCI Competition IV-2a.
    Los archivos .gdf deben colocarse manualmente aquí.
    """
    path = os.path.expanduser("~/.mne/datasets/BCI_IV_2a/")
    os.makedirs(path, exist_ok=True)
    print(f"Carpeta creada/preparada: {path}")
    print("👉 Ahora descarga manualmente los .gdf desde PhysioNet o la página oficial")
    print("   y colócalos en esa carpeta (ej: A01T.gdf, A01E.gdf, ...).")

if __name__ == "__main__":
    download_eegbci()
    prepare_bci_iv2a()

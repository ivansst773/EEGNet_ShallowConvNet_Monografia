import torch
import time

print("CUDA disponible:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

# Mantener la GPU ocupada con varias operaciones
for i in range(10):
    x = torch.rand((8000, 8000), device="cuda")
    y = torch.mm(x, x)
    torch.cuda.synchronize()  # esperar a que termine en GPU
    print(f"Iteración {i+1}, resultado en {y.device}")
    time.sleep(1)  # pausa para que puedas ver el uso en nvidia-smi

#!/bin/bash

# Parámetros comunes
EPOCHS=50
BATCH_SIZE=32
SEGMENT=True

# Lista de sujetos
SUBJECTS=(A01 A02 A03 A04 A05 A06 A07 A08 A09)

echo "[🚀] Iniciando entrenamiento completo para EEGNet y ShallowConvNet..."

for SUBJECT in "${SUBJECTS[@]}"; do
  echo "[🧠] EEGNet - Sujeto $SUBJECT"
  python -m src.eegnet.train_eegnet_bci_iv2a --subject $SUBJECT --epochs $EPOCHS --batch_size $BATCH_SIZE --dropout 0.25 --segment $SEGMENT

  echo "[🧠] ShallowConvNet - Sujeto $SUBJECT"
  python -m src.shallowconvnet.train_shallowconvnet_bci_iv2a --subject $SUBJECT --epochs $EPOCHS --batch_size $BATCH_SIZE --dropout 0.50 --segment $SEGMENT
done

echo "[✅] Entrenamiento completo finalizado."

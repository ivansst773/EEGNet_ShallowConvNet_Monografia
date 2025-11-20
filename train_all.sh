#!/bin/bash

# Parámetros comunes
EPOCHS=50
BATCH_SIZE=32
SEGMENT=True

# Lista de sujetos
SUBJECTS=(1 2 3 4 5 6 7 8 9)

# Crear carpeta de logs si no existe
mkdir -p logs

echo "[🚀] Iniciando entrenamiento completo para EEGNet y ShallowConvNet..."

for SUBJECT in "${SUBJECTS[@]}"; do
  echo "[🧠] EEGNet - Sujeto A0$SUBJECT"
  python -m src.eegnet.train_eegnet_bci_iv2a \
    --subject $SUBJECT --epochs $EPOCHS --batch_size $BATCH_SIZE --dropout 0.25 --segment $SEGMENT \
    > logs/A0${SUBJECT}_eegnet.log 2>&1

  echo "[🧠] ShallowConvNet - Sujeto A0$SUBJECT"
  python -m src.shallowconvnet.train_shallowconvnet_bci_iv2a \
    --subject $SUBJECT --epochs $EPOCHS --batch_size $BATCH_SIZE --dropout 0.50 --segment $SEGMENT \
    > logs/A0${SUBJECT}_shallowconvnet.log 2>&1
done

echo "[✅] Entrenamiento completo finalizado."

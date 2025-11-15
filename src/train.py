import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Wrapper para entrenar EEGNet o ShallowConvNet en BCI IV-2a")
    parser.add_argument("--model", type=str, required=True, choices=["EEGNet", "ShallowConvNet"], help="Modelo a entrenar")
    parser.add_argument("--subject", type=str, default="A01", help="Sujeto (ej. A01)")
    parser.add_argument("--epochs", type=int, default=2, help="Número de epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout")
    parser.add_argument("--segment", type=bool, default=False, help="Activar segmentación en ventanas")
    args = parser.parse_args()

    if args.model == "EEGNet":
        script = "src/eegnet/train_eegnet_bci_iv2a.py"
    elif args.model == "ShallowConvNet":
        script = "src/shallowconvnet/train_shallowconvnet_bci_iv2a.py"

    cmd = [
        "python", script,
        "--subject", args.subject,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--dropout", str(args.dropout),
        "--segment", str(args.segment)
    ]

    print(f"[🚀] Ejecutando: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()

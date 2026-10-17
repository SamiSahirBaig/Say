#!/usr/bin/env python3
"""Run model training standalone script."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_pipeline import load_config, run_train_mode

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO-OBB Model")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    args = parser.parse_args()

    cfg = load_config()
    run_train_mode(cfg, epochs=args.epochs, batch_size=args.batch_size)

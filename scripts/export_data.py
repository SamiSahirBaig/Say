#!/usr/bin/env python3
"""Run YOLO-OBB dataset export standalone script."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_pipeline import load_config, run_prepare_mode

if __name__ == "__main__":
    cfg = load_config()
    run_prepare_mode(cfg, train_n=2000, val_n=500, test_n=200)

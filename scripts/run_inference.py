#!/usr/bin/env python3
"""Run test inference and create competition submission CSV."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_pipeline import load_config, run_predict_mode

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Inference on Test Tiles")
    parser.add_argument("--test-dir", default=None, help="Directory containing test images")
    parser.add_argument("--weights", default=None, help="Weights path")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold")
    args = parser.parse_args()

    cfg = load_config()
    run_predict_mode(cfg, test_dir=args.test_dir, weights_path=args.weights, conf_threshold=args.conf)

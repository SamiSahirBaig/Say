#!/usr/bin/env python3
"""Run validation evaluation with DOTA rotated mAP@0.5 standalone script."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_pipeline import load_config, run_eval_mode

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Model on Rotated mAP@0.5")
    parser.add_argument("--weights", default=None, help="Weights path")
    args = parser.parse_args()

    cfg = load_config()
    run_eval_mode(cfg, weights_path=args.weights)

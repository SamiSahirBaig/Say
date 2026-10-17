#!/usr/bin/env python3
"""Run Demo District geospatial analysis, deduplication, and reporting."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_pipeline import load_config, run_report_mode

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Demo District Geospatial Analysis")
    parser.add_argument("--weights", default=None, help="Weights path")
    args = parser.parse_args()

    cfg = load_config()
    run_report_mode(cfg, weights_path=args.weights)

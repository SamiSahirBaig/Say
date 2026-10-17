#!/usr/bin/env python3
"""Run dataset audit standalone script."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_pipeline import load_config, run_audit_mode

if __name__ == "__main__":
    cfg = load_config()
    run_audit_mode(cfg)

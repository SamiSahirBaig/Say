#!/usr/bin/env bash
# Automated Setup Script for Raspberry Pi 4 / 5 Satellite Edge AI Payload
# Target OS: Raspberry Pi OS (Debian 12 Bookworm, 64-bit ARM)

set -e

echo "=========================================================="
echo "  Deploying Chimney Census AI on Raspberry Pi Satellite   "
echo "=========================================================="

# 1. System packages
echo "[*] Updating apt repositories and installing system vision libraries..."
sudo apt update
sudo apt install -y python3-pip python3-venv libgl1-mesa-glx libglib2.0-0 v4l-utils

# 2. Virtual Environment
echo "[*] Setting up Python virtual environment..."
python3 -m venv ~/satellite_env
source ~/satellite_env/bin/activate

# 3. Pip dependencies
echo "[*] Installing lightweight edge dependencies..."
pip install --upgrade pip
pip install -r requirements_rpi.txt

# 4. Self-test & Benchmark
echo "[*] Running onboard flight readiness self-test..."
python test_rpi_setup.py

echo ""
echo "=========================================================="
echo "  [✓] RASPBERRY PI EDGE AI ENVIRONMENT IS FLIGHT READY!   "
echo "=========================================================="
echo "Test on sample tile:  python raspberry_pi_satellite_edge.py --source sample_data/sample_zigzag.png"
echo "Run live video feed:  python raspberry_pi_satellite_edge.py --source /dev/video0"
echo "=========================================================="

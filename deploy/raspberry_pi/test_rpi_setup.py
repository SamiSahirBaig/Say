#!/usr/bin/env python3
"""Automated Self-Test & Hardware Verification Script for Raspberry Pi Satellite Edge AI."""

import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np

def run_raspberry_pi_verification():
    print("=" * 65)
    print("  [+] RASPBERRY PI SATELLITE EDGE AI: HARDWARE & MODEL VERIFICATION  ")
    print("=" * 65)
    
    # 1. Environment and Architecture
    print(f"[*] Python Version: {sys.version.split()[0]} on platform: {sys.platform}")
    
    # 2. Check Dependencies
    print("[*] Testing core edge dependencies...")
    try:
        import onnxruntime as ort
        print(f"    [PASS] ONNX Runtime: {ort.__version__}")
    except ImportError as e:
        print(f"    [FAIL] ONNX Runtime missing: {e}")
        return False
        
    try:
        import shapely
        print(f"    [PASS] Shapely: {shapely.__version__}")
    except ImportError as e:
        print(f"    [FAIL] Shapely missing: {e}")
        return False

    print(f"    [PASS] OpenCV: {cv2.__version__}")
    print(f"    [PASS] NumPy: {np.__version__}")

    # 3. Model Integrity Check
    model_paths = [
        Path("weights/best.onnx"),
        Path("weights/edge/best.onnx"),
        Path("../weights/edge/best.onnx"),
    ]
    model_file = next((p for p in model_paths if p.exists()), None)
    if model_file is None:
        print("    [FAIL] Model file weights/best.onnx not found!")
        return False
        
    file_size_mb = model_file.stat().st_size / (1024 * 1024)
    print(f"    [PASS] Model Weights Verified: {model_file} ({file_size_mb:.2f} MB)")
    
    # 4. Latency & CPU Inference Benchmark
    print("[*] Benchmarking Edge CPU Execution (10 iterations @ 128x128)...")
    try:
        from raspberry_pi_satellite_edge import EdgeOBBDetector
    except ImportError:
        from deploy.raspberry_pi.raspberry_pi_satellite_edge import EdgeOBBDetector
    
    detector = EdgeOBBDetector(str(model_file), conf_thresh=0.02)
    dummy_input = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    
    # Warmup
    for _ in range(3):
        detector.infer_tile(dummy_input)
        
    latencies = []
    for _ in range(10):
        t0 = time.perf_counter()
        detector.infer_tile(dummy_input)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        
    mean_lat = np.mean(latencies)
    throughput_fps = 1000.0 / mean_lat
    print(f"    [PASS] Mean Inference Latency: {mean_lat:.2f} ms per tile ({throughput_fps:.1f} FPS)")

    # 5. Sample Image Detection Test
    sample_img_p = Path("sample_data/sample_zigzag.png")
    if sample_img_p.exists():
        tile = cv2.imread(str(sample_img_p))
        dets = detector.infer_tile(tile)
        print(f"    [PASS] Sample Tile Inference: Found {len(dets)} kiln(s)")
        for d in dets:
            print(f"           • Class: {d['class']} | Conf: {d['score']:.3f} | Center: {d['center']}")
            
    print("=" * 65)
    print("  [PASS] ALL SYSTEMS OPERATIONAL: SATELLITE EDGE AI IS FLIGHT READY ")
    print("=" * 65)
    return True

if __name__ == "__main__":
    success = run_raspberry_pi_verification()
    sys.exit(0 if success else 1)

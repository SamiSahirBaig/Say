#!/usr/bin/env python3
"""Export YOLOv8n-OBB model for Edge Deployment on Raspberry Pi / Satellite Payloads."""

import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from ultralytics import YOLO

def export_edge_model(
    weights_path: str = "weights/best.pt",
    output_dir: str = "weights/edge",
    img_size: int = 128,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"[*] Loading PyTorch weights from: {weights_path}")
    model = YOLO(weights_path)
    
    # 1. Export standard fused ONNX (Opset 18, Slimmed)
    print(f"[*] Exporting to ONNX (imgsz={img_size})...")
    onnx_path = model.export(
        format="onnx",
        imgsz=img_size,
        dynamic=False,
        simplify=True,
        opset=18,
    )
    
    # Copy to edge weights directory
    target_onnx = output_path / "best.onnx"
    if Path(onnx_path).exists() and onnx_path != str(target_onnx):
        import shutil
        shutil.copy2(onnx_path, target_onnx)
    
    file_size_mb = os.path.getsize(target_onnx) / (1024 * 1024)
    print(f"[✓] Fused ONNX model saved to: {target_onnx} ({file_size_mb:.2f} MB)")
    
    # 2. Benchmark ONNX model on CPU (simulating Raspberry Pi execution)
    import onnxruntime as ort
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4  # Standard quad-core Pi 4/5
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(str(target_onnx), sess_options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    
    # Dummy warmup
    dummy_input = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
    for _ in range(5):
        session.run(None, {input_name: dummy_input})
        
    # Latency test
    latencies = []
    for _ in range(30):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        latencies.append((time.perf_counter() - t0) * 1000.0)
        
    mean_lat = np.mean(latencies)
    std_lat = np.std(latencies)
    fps = 1000.0 / mean_lat
    
    print(f"[✓] CPU Benchmark: {mean_lat:.2f} ± {std_lat:.2f} ms per tile ({fps:.1f} FPS)")
    
    metadata = {
        "model_name": "YOLOv8n-OBB Edge",
        "format": "ONNX",
        "input_shape": [1, 3, img_size, img_size],
        "output_shape": [1, 8, 336],
        "size_mb": round(file_size_mb, 2),
        "target_hardware": "Raspberry Pi 4 / Raspberry Pi 5 (ARM64 Cortex-A72 / A76)",
        "cpu_latency_ms": round(float(mean_lat), 2),
        "cpu_throughput_fps": round(float(fps), 1),
        "power_envelope_watts": "3.5 - 5.0 W (Raspberry Pi payload)",
    }
    
    meta_path = output_path / "edge_model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"[✓] Edge metadata saved to: {meta_path}")
    return str(target_onnx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLOv8n-OBB for Raspberry Pi / Satellite Edge")
    parser.add_argument("--weights", default="weights/best.pt", help="Path to best.pt")
    parser.add_argument("--out-dir", default="weights/edge", help="Output directory")
    parser.add_argument("--imgsz", type=int, default=128, help="Tile size")
    args = parser.parse_args()
    
    export_edge_model(args.weights, args.out_dir, args.imgsz)

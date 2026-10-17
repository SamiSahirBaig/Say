# Model Weights & Checkpoint Provenance

## Included Checkpoints

### 1. `yolov8n-obb.pt` (Base Checkpoint)
- **Source**: Ultralytics Official Model Zoo (`https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n-obb.pt`)
- **Architecture**: YOLOv8-OBB (Nano)
- **Task**: Oriented Object Detection (OBB)
- **Parameters**: ~3.1M
- **Input Resolution**: 128x128 (Native Sentinel-2 tile size)
- **Pre-training**: DOTAv1 / Aerial Imagery
- **License**: AGPL-3.0

### 2. `best.pt` (Fine-Tuned Checkpoint)
- **Dataset**: SentinelKilnDB (NeurIPS 2025 Datasets & Benchmarks)
- **Classes**:
  - Class 0: `CFCBK` (Circular Fixed Chimney Bull's Trench Kiln)
  - Class 1: `FCBK` (Fixed Chimney Bull's Trench Kiln)
  - Class 2: `Zigzag` (Cleaner Zig-zag firing design)
- **Training Strategy**:
  - Small-object preservation at native 128x128 resolution
  - Mixed precision (FP16 AMP) enabled
  - Rotated IoU bounding box loss
  - Stratified rare-class balancing to avoid CFCBK starvation
- **License**: CC BY-NC 4.0 (Non-commercial research use only, conforming to SentinelKilnDB license).

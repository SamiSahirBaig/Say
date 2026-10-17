"""Training pipeline for YOLO-OBB brick kiln detection."""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional
from ultralytics import YOLO


def train_yolo_obb_model(
    data_yaml: str = "configs/data.yaml",
    base_model: str = "weights/yolov8n-obb.pt",
    epochs: int = 10,
    imgsz: int = 128,
    batch_size: int = 64,
    device: str = "0",
    workers: int = 2,
    seed: int = 42,
    project: str = "runs/train",
    name: str = "chimney_census_obb",
    output_best_path: str = "weights/best.pt",
    metrics_json_path: str = "outputs/reports/training_metrics.json",
    resume: bool = False,
) -> Dict[str, Any]:
    """Train YOLO-OBB model on SentinelKilnDB tiles.

    Args:
        data_yaml: Path to data.yaml dataset config.
        base_model: Starting weights (e.g., weights/yolov8n-obb.pt).
        epochs: Training epochs.
        imgsz: Image resolution (128 for SentinelKilnDB).
        batch_size: Batch size.
        device: CUDA device index or 'cpu'.
        workers: DataLoader workers.
        seed: Random seed.
        project: Directory for runs.
        name: Name for this experiment run.
        output_best_path: Path to copy best checkpoint to.
        metrics_json_path: Path to save training summary metrics.
        resume: Whether to resume training from last checkpoint.

    Returns:
        Summary metrics dictionary.
    """
    print(f"Initializing YOLO model from base checkpoint: {base_model}")
    model = YOLO(base_model)

    print(
        f"Starting YOLO-OBB training on device={device}, epochs={epochs}, "
        f"batch={batch_size}, imgsz={imgsz}..."
    )

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        workers=workers,
        seed=seed,
        project=project,
        name=name,
        resume=resume,
        amp=True,
        save=True,
        verbose=True,
        plots=True,
    )

    # Locate best.pt from run directory or results save_dir
    best_pt = Path(getattr(results, "save_dir", project)) / "weights" / "best.pt"
    if not best_pt.exists():
        candidates = sorted(Path("runs").glob(f"**/{name}*/weights/best.pt"))
        if candidates:
            best_pt = candidates[-1]

    metrics_summary: Dict[str, Any] = {
        "base_model": base_model,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch_size": batch_size,
        "device": device,
    }

    if best_pt.exists():
        Path(output_best_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, output_best_path)
        print(f"Copied best model weights to: {output_best_path}")
        metrics_summary["best_checkpoint"] = str(output_best_path)
    else:
        print("Warning: best.pt not found, keeping base weights as fallback.")
        metrics_summary["best_checkpoint"] = str(base_model)

    # Extract metrics if available
    try:
        metrics_summary["metrics/mAP50(B)"] = float(getattr(results, "results_dict", {}).get("metrics/mAP50(B)", 0.0))
        metrics_summary["metrics/mAP50-95(B)"] = float(getattr(results, "results_dict", {}).get("metrics/mAP50-95(B)", 0.0))
    except Exception:
        pass

    if metrics_json_path:
        Path(metrics_json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(metrics_summary, f, indent=2)

    return metrics_summary

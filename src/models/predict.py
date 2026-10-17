"""Inference engine and competition submission generator."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

from ..evaluation.rotated_iou import polygon_from_8coords

CLASS_ID_TO_NAME: Dict[int, str] = {
    0: "CFCBK",
    1: "FCBK",
    2: "Zigzag",
}

SUBMISSION_COLUMNS = ["tile_id", "class", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4", "score"]
VALID_CLASSES = {"FCBK", "CFCBK", "Zigzag"}


def run_inference(
    model_path: str = "weights/best.pt",
    images_dir: Optional[str] = "data/yolo_dataset/images/test",
    image_paths: Optional[Sequence[Union[str, Path]]] = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    batch_size: int = 64,
    device: str = "0",
    imgsz: int = 128,
) -> pd.DataFrame:
    """Run batched YOLO-OBB inference over tile images.

    Args:
        model_path: Path to trained or base checkpoint.
        images_dir: Directory containing .png tiles.
        image_paths: Optional explicit list of image paths.
        conf_threshold: Minimum confidence score to retain detection.
        iou_threshold: NMS IoU threshold.
        batch_size: Inference batch size.
        device: CUDA device index or 'cpu'.
        imgsz: Image resolution.

    Returns:
        DataFrame with columns: tile_id, class, x1, y1, x2, y2, x3, y3, x4, y4, score
    """
    if image_paths is None:
        if images_dir is None or not Path(images_dir).exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        image_paths = sorted(list(Path(images_dir).glob("*.png")) + list(Path(images_dir).glob("*.jpg")))

    if len(image_paths) == 0:
        print("Warning: No images provided for inference.")
        return pd.DataFrame(columns=SUBMISSION_COLUMNS)

    print(f"Loading YOLO model from: {model_path}")
    model = YOLO(model_path)

    detections: List[Dict[str, Any]] = []

    # Run in batches
    num_batches = (len(image_paths) + batch_size - 1) // batch_size
    print(f"Running inference on {len(image_paths)} tiles ({num_batches} batches)...")

    for b_idx in tqdm(range(num_batches), desc="Inference"):
        batch_paths = image_paths[b_idx * batch_size : (b_idx + 1) * batch_size]
        str_paths = [str(p) for p in batch_paths]

        results = model.predict(
            source=str_paths,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )

        for path, result in zip(batch_paths, results):
            tile_id = Path(path).stem

            if result.obb is None or len(result.obb) == 0:
                # No detections for this tile -> no row in submission
                continue

            xyxyxyxyn = result.obb.xyxyxyxyn.cpu().numpy()  # (N, 4, 2)
            classes = result.obb.cls.cpu().numpy().astype(int)  # (N,)
            confs = result.obb.conf.cpu().numpy()  # (N,)

            for corners, cls_id, score in zip(xyxyxyxyn, classes, confs):
                class_name = CLASS_ID_TO_NAME.get(int(cls_id), "Zigzag")
                # Flatten the 4 corners: x1, y1, x2, y2, x3, y3, x4, y4
                # Clip lightly to [0.0, 1.0] to guard against tiny float overflow
                flat_coords = [float(np.clip(c, 0.0, 1.0)) for c in corners.flatten()]

                detections.append({
                    "tile_id": tile_id,
                    "class": class_name,
                    "x1": round(flat_coords[0], 4),
                    "y1": round(flat_coords[1], 4),
                    "x2": round(flat_coords[2], 4),
                    "y2": round(flat_coords[3], 4),
                    "x3": round(flat_coords[4], 4),
                    "y3": round(flat_coords[5], 4),
                    "x4": round(flat_coords[6], 4),
                    "y4": round(flat_coords[7], 4),
                    "score": round(float(score), 4),
                })

    df = pd.DataFrame(detections, columns=SUBMISSION_COLUMNS)
    return df


def create_submission_csv(
    detections_df: pd.DataFrame,
    output_csv_path: str = "outputs/submission.csv",
) -> str:
    """Save formatted submission CSV meeting official competition specifications."""
    out_p = Path(output_csv_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # Ensure column order
    df = detections_df[SUBMISSION_COLUMNS] if not detections_df.empty else pd.DataFrame(columns=SUBMISSION_COLUMNS)
    df.to_csv(out_p, index=False)
    print(f"Saved submission CSV with {len(df)} rows to: {output_csv_path}")
    return str(out_p)


def validate_submission_csv(csv_path: str) -> Dict[str, Any]:
    """Validate a submission CSV against competition rules.

    Checks:
    - Exact required header: tile_id,class,x1,y1,x2,y2,x3,y3,x4,y4,score
    - Allowed classes: FCBK, CFCBK, Zigzag
    - Normalized coordinates in [0, 1]
    - Valid non-NaN scores in [0, 1]
    - Valid non-degenerate polygon geometry for each detection
    """
    p = Path(csv_path)
    if not p.exists():
        return {"is_valid": False, "errors": [f"File does not exist: {csv_path}"]}

    df = pd.read_csv(p)

    errors = []
    # Check header
    if list(df.columns) != SUBMISSION_COLUMNS:
        errors.append(f"Header mismatch. Expected {SUBMISSION_COLUMNS}, got {list(df.columns)}")

    if len(df) == 0:
        return {
            "is_valid": True,
            "total_detections": 0,
            "errors": errors,
            "warning": "Submission file has 0 detections (empty dataset or high threshold).",
        }

    # Check classes
    unique_classes = set(df["class"].unique())
    invalid_classes = unique_classes - VALID_CLASSES
    if invalid_classes:
        errors.append(f"Invalid classes detected: {invalid_classes}. Must be subset of {VALID_CLASSES}")

    # Check NaNs
    if df.isna().any().any():
        errors.append("NaN values found in submission CSV.")

    # Check coordinates in [0, 1]
    coord_cols = ["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]
    min_c = df[coord_cols].min().min()
    max_c = df[coord_cols].max().max()
    if min_c < -0.01 or max_c > 1.01:
        errors.append(f"Coordinates outside normalized [0, 1] range: min={min_c}, max={max_c}")

    # Check scores in [0, 1]
    min_s = df["score"].min()
    max_s = df["score"].max()
    if min_s < 0.0 or max_s > 1.0:
        errors.append(f"Confidence score out of [0, 1] range: min={min_s}, max={max_s}")

    # Check polygon geometry validity
    invalid_poly_count = 0
    for _, row in df.iterrows():
        try:
            poly = polygon_from_8coords([row[c] for c in coord_cols])
            if poly.area <= 0:
                invalid_poly_count += 1
        except Exception:
            invalid_poly_count += 1

    if invalid_poly_count > 0:
        errors.append(f"Found {invalid_poly_count} degenerate or invalid polygon bounding boxes.")

    report = {
        "is_valid": len(errors) == 0,
        "total_detections": len(df),
        "unique_tiles_with_detections": df["tile_id"].nunique(),
        "class_counts": df["class"].value_counts().to_dict(),
        "errors": errors,
    }
    return report

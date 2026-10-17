"""Rotated bounding box evaluation and IoU module."""
from .rotated_iou import (
    polygon_from_8coords,
    calculate_rotated_iou,
    calculate_rotated_iou_matrix,
    ensure_valid_polygon,
)
from .dota_evaluator import (
    evaluate_detections,
    compute_ap_all_points,
    DotaEvaluator,
)

__all__ = [
    "polygon_from_8coords",
    "calculate_rotated_iou",
    "calculate_rotated_iou_matrix",
    "ensure_valid_polygon",
    "evaluate_detections",
    "compute_ap_all_points",
    "DotaEvaluator",
]

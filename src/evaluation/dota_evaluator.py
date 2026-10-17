"""Official DOTA benchmark rotated box mean Average Precision (mAP@0.5) evaluator."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from .rotated_iou import calculate_rotated_iou, polygon_from_8coords


@dataclass
class Detection:
    tile_id: str
    class_name: str
    score: float
    coords: Tuple[float, float, float, float, float, float, float, float]


@dataclass
class GroundTruth:
    tile_id: str
    class_name: str
    coords: Tuple[float, float, float, float, float, float, float, float]
    matched: bool = False


def compute_ap_all_points(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute Average Precision using all-point interpolation (standard DOTA / VOC 2012+ benchmark).

    Args:
        recalls: 1D array of recall points.
        precisions: 1D array of precision points.

    Returns:
        AP float in [0.0, 1.0].
    """
    if len(recalls) == 0:
        return 0.0

    # Ensure precision is monotonically decreasing from right to left
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Find points where recall changes
    indices = np.where(mrec[1:] != mrec[:-1])[0] + 1
    ap = np.sum((mrec[indices] - mrec[indices - 1]) * mpre[indices])
    return float(np.clip(ap, 0.0, 1.0))


class DotaEvaluator:
    """Evaluates rotated bounding box detections against ground truths according to DOTA rules."""

    CLASSES = ["FCBK", "CFCBK", "Zigzag"]

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold

    def evaluate(
        self,
        predictions: Union[pd.DataFrame, Sequence[Detection]],
        ground_truths: Union[pd.DataFrame, Sequence[GroundTruth]],
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """Run full rotated evaluation and compute per-class AP and mAP@0.5.

        Args:
            predictions: DataFrame or list of Detections.
            ground_truths: DataFrame or list of GroundTruths.

        Returns:
            Dictionary with 'mAP@0.5', 'per_class_ap', 'class_gt_counts', 'class_det_counts'.
        """
        pred_list = self._normalize_predictions(predictions)
        gt_list = self._normalize_ground_truths(ground_truths)

        # Index ground truths by tile_id and class
        gt_by_tile_and_class: Dict[Tuple[str, str], List[GroundTruth]] = {}
        gt_counts_by_class: Dict[str, int] = {cls: 0 for cls in self.CLASSES}

        for gt in gt_list:
            if gt.class_name not in gt_counts_by_class:
                continue
            key = (gt.tile_id, gt.class_name)
            if key not in gt_by_tile_and_class:
                gt_by_tile_and_class[key] = []
            # Create a fresh copy to allow matching flag
            gt_by_tile_and_class[key].append(
                GroundTruth(tile_id=gt.tile_id, class_name=gt.class_name, coords=gt.coords, matched=False)
            )
            gt_counts_by_class[gt.class_name] += 1

        # Per class evaluation
        per_class_ap: Dict[str, float] = {}
        per_class_details: Dict[str, Dict] = {}

        for cls in self.CLASSES:
            cls_preds = [p for p in pred_list if p.class_name == cls]
            # Sort by score descending
            cls_preds.sort(key=lambda p: p.score, reverse=True)

            n_gt = gt_counts_by_class[cls]
            n_det = len(cls_preds)

            if n_gt == 0:
                per_class_ap[cls] = 0.0 if n_det > 0 else 1.0
                per_class_details[cls] = {"ap": per_class_ap[cls], "tp": 0, "fp": n_det, "n_gt": 0, "n_det": n_det}
                continue

            if n_det == 0:
                per_class_ap[cls] = 0.0
                per_class_details[cls] = {"ap": 0.0, "tp": 0, "fp": 0, "n_gt": n_gt, "n_det": 0}
                continue

            tp = np.zeros(n_det, dtype=np.float32)
            fp = np.zeros(n_det, dtype=np.float32)

            for idx, pred in enumerate(cls_preds):
                key = (pred.tile_id, pred.class_name)
                candidate_gts = gt_by_tile_and_class.get(key, [])

                best_iou = 0.0
                best_gt_idx = -1

                for gt_idx, gt in enumerate(candidate_gts):
                    iou = calculate_rotated_iou(pred.coords, gt.coords)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_iou >= self.iou_threshold and best_gt_idx >= 0:
                    matched_gt = candidate_gts[best_gt_idx]
                    if not matched_gt.matched:
                        tp[idx] = 1.0
                        matched_gt.matched = True
                    else:
                        # Duplicate detection for already matched ground truth
                        fp[idx] = 1.0
                else:
                    fp[idx] = 1.0

            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            recalls = cum_tp / float(n_gt)
            precisions = cum_tp / (cum_tp + cum_fp)

            ap = compute_ap_all_points(recalls, precisions)
            per_class_ap[cls] = ap
            per_class_details[cls] = {
                "ap": ap,
                "tp": int(cum_tp[-1]) if len(cum_tp) > 0 else 0,
                "fp": int(cum_fp[-1]) if len(cum_fp) > 0 else 0,
                "n_gt": n_gt,
                "n_det": n_det,
            }

        mean_ap = float(np.mean([per_class_ap[cls] for cls in self.CLASSES]))

        return {
            "mAP@0.5": round(mean_ap, 5),
            "per_class_ap": {cls: round(per_class_ap[cls], 5) for cls in self.CLASSES},
            "per_class_details": per_class_details,
            "class_gt_counts": gt_counts_by_class,
            "class_det_counts": {cls: len([p for p in pred_list if p.class_name == cls]) for cls in self.CLASSES},
        }

    def _normalize_predictions(self, preds: Union[pd.DataFrame, Sequence[Detection]]) -> List[Detection]:
        if isinstance(preds, pd.DataFrame):
            # Expect columns: tile_id, class, x1, y1, x2, y2, x3, y3, x4, y4, score
            cols = ["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]
            res = []
            for _, r in preds.iterrows():
                coords = tuple(float(r[c]) for c in cols)
                res.append(
                    Detection(
                        tile_id=str(r["tile_id"]),
                        class_name=str(r["class"]),
                        score=float(r["score"]),
                        coords=coords,  # type: ignore
                    )
                )
            return res
        return list(preds)

    def _normalize_ground_truths(self, gts: Union[pd.DataFrame, Sequence[GroundTruth]]) -> List[GroundTruth]:
        if isinstance(gts, pd.DataFrame):
            cols = ["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]
            res = []
            for _, r in gts.iterrows():
                coords = tuple(float(r[c]) for c in cols)
                res.append(
                    GroundTruth(
                        tile_id=str(r["tile_id"]),
                        class_name=str(r["class"]),
                        coords=coords,  # type: ignore
                    )
                )
            return res
        return list(gts)


def evaluate_detections(
    predictions_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    iou_threshold: float = 0.5,
) -> Dict[str, Union[float, Dict[str, float]]]:
    """Convenience function to evaluate DataFrame predictions against ground truth."""
    evaluator = DotaEvaluator(iou_threshold=iou_threshold)
    return evaluator.evaluate(predictions_df, ground_truth_df)

"""Unit tests for DOTA rotated box evaluator and AP computation."""

import pandas as pd
import pytest
from src.evaluation.dota_evaluator import DotaEvaluator, compute_ap_all_points


def test_compute_ap_perfect():
    recalls = [0.2, 0.4, 0.6, 0.8, 1.0]
    precisions = [1.0, 1.0, 1.0, 1.0, 1.0]
    ap = compute_ap_all_points(recalls, precisions)
    assert ap == pytest.approx(1.0, abs=1e-4)


def test_dota_evaluator_perfect_matches():
    evaluator = DotaEvaluator(iou_threshold=0.5)

    gt_df = pd.DataFrame([
        {
            "tile_id": "tile_1",
            "class": "FCBK",
            "x1": 0.2, "y1": 0.2, "x2": 0.4, "y2": 0.2,
            "x3": 0.4, "y3": 0.3, "x4": 0.2, "y4": 0.3,
        },
        {
            "tile_id": "tile_1",
            "class": "Zigzag",
            "x1": 0.6, "y1": 0.6, "x2": 0.8, "y2": 0.6,
            "x3": 0.8, "y3": 0.7, "x4": 0.6, "y4": 0.7,
        },
        {
            "tile_id": "tile_2",
            "class": "CFCBK",
            "x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.1,
            "x3": 0.2, "y3": 0.2, "x4": 0.1, "y4": 0.2,
        },
    ])

    pred_df = pd.DataFrame([
        {
            "tile_id": "tile_1",
            "class": "FCBK",
            "x1": 0.201, "y1": 0.201, "x2": 0.401, "y2": 0.201,
            "x3": 0.401, "y3": 0.301, "x4": 0.201, "y4": 0.301,
            "score": 0.95,
        },
        {
            "tile_id": "tile_1",
            "class": "Zigzag",
            "x1": 0.6, "y1": 0.6, "x2": 0.8, "y2": 0.6,
            "x3": 0.8, "y3": 0.7, "x4": 0.6, "y4": 0.7,
            "score": 0.90,
        },
        {
            "tile_id": "tile_2",
            "class": "CFCBK",
            "x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.1,
            "x3": 0.2, "y3": 0.2, "x4": 0.1, "y4": 0.2,
            "score": 0.85,
        },
    ])

    results = evaluator.evaluate(pred_df, gt_df)
    assert results["mAP@0.5"] == pytest.approx(1.0, abs=1e-3)
    assert results["per_class_ap"]["FCBK"] == pytest.approx(1.0, abs=1e-3)
    assert results["per_class_ap"]["CFCBK"] == pytest.approx(1.0, abs=1e-3)
    assert results["per_class_ap"]["Zigzag"] == pytest.approx(1.0, abs=1e-3)


def test_dota_evaluator_false_positive():
    evaluator = DotaEvaluator(iou_threshold=0.5)

    gt_df = pd.DataFrame([
        {
            "tile_id": "tile_1",
            "class": "FCBK",
            "x1": 0.2, "y1": 0.2, "x2": 0.4, "y2": 0.2,
            "x3": 0.4, "y3": 0.3, "x4": 0.2, "y4": 0.3,
        },
    ])

    # Prediction has wrong class (Zigzag instead of FCBK)
    pred_df = pd.DataFrame([
        {
            "tile_id": "tile_1",
            "class": "Zigzag",
            "x1": 0.2, "y1": 0.2, "x2": 0.4, "y2": 0.2,
            "x3": 0.4, "y3": 0.3, "x4": 0.2, "y4": 0.3,
            "score": 0.90,
        },
    ])

    results = evaluator.evaluate(pred_df, gt_df)
    # FCBK was missed (AP = 0.0), Zigzag was a false alarm with 0 ground truths (AP = 0.0)
    assert results["per_class_ap"]["FCBK"] == 0.0
    assert results["per_class_ap"]["Zigzag"] == 0.0

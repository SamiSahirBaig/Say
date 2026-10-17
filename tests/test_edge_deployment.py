"""Unit tests for Raspberry Pi satellite edge deployment and video stream processing."""

import json
from pathlib import Path
import numpy as np
import pytest
from shapely.geometry import Polygon
from deploy.raspberry_pi.raspberry_pi_satellite_edge import (
    EdgeOBBDetector,
    apply_frame_nms,
    compute_polygon_iou,
)


def test_edge_model_exists():
    onnx_file = Path("weights/edge/best.onnx")
    assert onnx_file.exists(), "Edge ONNX model file does not exist"
    assert onnx_file.stat().st_size > 5 * 1024 * 1024, "Edge ONNX model file is unexpectedly small"


def test_compute_polygon_iou():
    p1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    p2 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    assert pytest.approx(compute_polygon_iou(p1, p2), 0.001) == 1.0

    p3 = Polygon([(20, 20), (30, 20), (30, 30), (20, 30)])
    assert compute_polygon_iou(p1, p3) == 0.0

    p4 = Polygon([(5, 0), (15, 0), (15, 10), (5, 10)])
    assert pytest.approx(compute_polygon_iou(p1, p4), 0.01) == 50.0 / 150.0


def test_apply_frame_nms():
    # Two almost identical detections
    det1 = {
        "class": "Zigzag",
        "score": 0.85,
        "center": (50, 50),
        "corners": [(40, 40), (60, 40), (60, 60), (40, 60)],
    }
    det2 = {
        "class": "Zigzag",
        "score": 0.70,
        "center": (51, 51),
        "corners": [(41, 41), (61, 41), (61, 61), (41, 61)],
    }
    # One distant detection
    det3 = {
        "class": "FCBK",
        "score": 0.90,
        "center": (150, 150),
        "corners": [(140, 140), (160, 140), (160, 160), (140, 160)],
    }

    filtered = apply_frame_nms([det1, det2, det3], iou_thresh=0.30)
    assert len(filtered) == 2, f"Expected 2 detections after NMS, got {len(filtered)}"
    assert any(d["class"] == "FCBK" for d in filtered)
    assert any(d["class"] == "Zigzag" for d in filtered)


def test_edge_detector_inference():
    onnx_file = Path("weights/edge/best.onnx")
    if not onnx_file.exists():
        pytest.skip("ONNX edge model not found")

    detector = EdgeOBBDetector(str(onnx_file), conf_thresh=0.01)
    dummy_tile = np.zeros((128, 128, 3), dtype=np.uint8)
    dets = detector.infer_tile(dummy_tile)
    assert isinstance(dets, list), "Inference on dummy tile should return a list"


def test_telemetry_json_schema():
    telem_file = Path("outputs/reports/downlink_telemetry.json")
    if not telem_file.exists():
        pytest.skip("Downlink telemetry JSON not yet generated")

    with open(telem_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "satellite_id" in data
    assert "alerts" in data
    assert isinstance(data["alerts"], list)

    if data["alerts"]:
        first = data["alerts"][0]
        assert "event_id" in first
        assert "class" in first
        assert "confidence" in first
        assert "compliance_flag" in first
        assert "telemetry" in first


def test_periodic_camera_edge():
    from deploy.raspberry_pi.raspberry_pi_camera_edge import RaspberryPiPeriodicCameraEdge

    engine = RaspberryPiPeriodicCameraEdge(
        model_path="weights/edge/best.onnx",
        camera_source="mock",
        interval_seconds=1.0,
        output_dir="outputs/camera_captures",
        output_telemetry="outputs/reports/downlink_telemetry.json",
        output_csv="outputs/camera_detections.csv",
    )
    # Process single test capture
    dummy_frame = np.zeros((256, 256, 3), dtype=np.uint8)
    annotated, alerts = engine.process_capture(dummy_frame, frame_id=99)
    assert annotated.shape == (256, 256, 3), "Annotated frame must match input dimensions"
    assert isinstance(alerts, list), "Alerts must be a list"
    assert engine.output_csv.exists(), "Camera detections CSV must exist"

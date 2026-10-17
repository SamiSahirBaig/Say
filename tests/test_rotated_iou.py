"""Unit tests for rotated polygon creation and IoU computation."""

import pytest
from shapely.geometry import Polygon
from src.evaluation.rotated_iou import (
    calculate_rotated_iou,
    calculate_rotated_iou_matrix,
    polygon_from_8coords,
)


def test_polygon_from_8coords():
    coords = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    poly = polygon_from_8coords(coords)
    assert isinstance(poly, Polygon)
    assert poly.area == pytest.approx(1.0, abs=1e-5)


def test_identical_boxes_iou():
    coords = [0.2, 0.2, 0.4, 0.2, 0.4, 0.3, 0.2, 0.3]
    iou = calculate_rotated_iou(coords, coords)
    assert iou == pytest.approx(1.0, abs=1e-5)


def test_disjoint_boxes_iou():
    box1 = [0.0, 0.0, 0.1, 0.0, 0.1, 0.1, 0.0, 0.1]
    box2 = [0.8, 0.8, 0.9, 0.8, 0.9, 0.9, 0.8, 0.9]
    iou = calculate_rotated_iou(box1, box2)
    assert iou == 0.0


def test_half_overlapping_boxes_iou():
    box1 = [0.0, 0.0, 0.4, 0.0, 0.4, 0.4, 0.0, 0.4]  # area = 0.16
    box2 = [0.2, 0.0, 0.6, 0.0, 0.6, 0.4, 0.2, 0.4]  # area = 0.16
    # intersection: [0.2, 0.0] to [0.4, 0.4] -> area = 0.08
    # union = 0.16 + 0.16 - 0.08 = 0.24
    # IoU = 0.08 / 0.24 = 1/3 ≈ 0.3333
    iou = calculate_rotated_iou(box1, box2)
    assert iou == pytest.approx(1.0 / 3.0, abs=1e-3)


def test_rotated_box_iou():
    # Box tilted at 45 degrees
    box1 = [0.405, 0.305, 0.495, 0.282, 0.507, 0.329, 0.417, 0.352]
    box2 = [0.400, 0.300, 0.490, 0.277, 0.502, 0.324, 0.412, 0.347]
    iou = calculate_rotated_iou(box1, box2)
    assert iou > 0.65  # high overlap on rotated box


def test_iou_matrix():
    box1 = [0.0, 0.0, 0.2, 0.0, 0.2, 0.2, 0.0, 0.2]
    box2 = [0.5, 0.5, 0.7, 0.5, 0.7, 0.7, 0.5, 0.7]
    mat = calculate_rotated_iou_matrix([box1, box2], [box1, box2])
    assert mat.shape == (2, 2)
    assert mat[0, 0] == pytest.approx(1.0, abs=1e-4)
    assert mat[1, 1] == pytest.approx(1.0, abs=1e-4)
    assert mat[0, 1] == 0.0
    assert mat[1, 0] == 0.0

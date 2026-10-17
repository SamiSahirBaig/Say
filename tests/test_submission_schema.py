"""Unit tests for submission CSV format and schema validation."""

from pathlib import Path
import pandas as pd
import pytest
from src.models.predict import create_submission_csv, validate_submission_csv


def test_valid_submission_csv(tmp_path):
    csv_file = tmp_path / "submission.csv"

    df = pd.DataFrame([
        {
            "tile_id": "t_08812",
            "class": "Zigzag",
            "x1": 0.405, "y1": 0.305, "x2": 0.495, "y2": 0.282,
            "x3": 0.507, "y3": 0.329, "x4": 0.417, "y4": 0.352,
            "score": 0.91,
        },
        {
            "tile_id": "t_08812",
            "class": "FCBK",
            "x1": 0.727, "y1": 0.697, "x2": 0.813, "y2": 0.747,
            "x3": 0.783, "y3": 0.799, "x4": 0.697, "y4": 0.749,
            "score": 0.66,
        },
    ])

    create_submission_csv(df, str(csv_file))
    report = validate_submission_csv(str(csv_file))

    assert report["is_valid"] is True
    assert len(report["errors"]) == 0
    assert report["total_detections"] == 2
    assert report["class_counts"]["Zigzag"] == 1
    assert report["class_counts"]["FCBK"] == 1


def test_invalid_class_submission(tmp_path):
    csv_file = tmp_path / "invalid_class.csv"
    df = pd.DataFrame([
        {
            "tile_id": "t_001",
            "class": "InvalidClass",
            "x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.1,
            "x3": 0.2, "y3": 0.2, "x4": 0.1, "y4": 0.2,
            "score": 0.8,
        }
    ])
    df.to_csv(csv_file, index=False)
    report = validate_submission_csv(str(csv_file))
    assert report["is_valid"] is False
    assert any("Invalid classes" in e for e in report["errors"])


def test_invalid_coordinates_submission(tmp_path):
    csv_file = tmp_path / "invalid_coords.csv"
    df = pd.DataFrame([
        {
            "tile_id": "t_002",
            "class": "FCBK",
            "x1": 1.5, "y1": 0.1, "x2": 0.2, "y2": 0.1,  # x1 > 1.0
            "x3": 0.2, "y3": 0.2, "x4": 0.1, "y4": 0.2,
            "score": 0.8,
        }
    ])
    df.to_csv(csv_file, index=False)
    report = validate_submission_csv(str(csv_file))
    assert report["is_valid"] is False
    assert any("Coordinates outside normalized" in e for e in report["errors"])

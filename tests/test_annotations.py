"""Unit tests for YOLO-OBB annotation parsing and class mappings."""

import pytest
from src.data.loader import (
    CLASS_ID_TO_NAME,
    CLASS_NAME_TO_ID,
    parse_yolo_obb_line,
)


def test_class_mappings():
    assert CLASS_ID_TO_NAME[0] == "CFCBK"
    assert CLASS_ID_TO_NAME[1] == "FCBK"
    assert CLASS_ID_TO_NAME[2] == "Zigzag"

    assert CLASS_NAME_TO_ID["CFCBK"] == 0
    assert CLASS_NAME_TO_ID["FCBK"] == 1
    assert CLASS_NAME_TO_ID["Zigzag"] == 2


def test_parse_yolo_obb_valid():
    line = "1 0.6651 0.5479 0.6929 0.5444 0.6838 0.4804 0.6560 0.4838"
    cls_id, cls_name, coords = parse_yolo_obb_line(line)
    assert cls_id == 1
    assert cls_name == "FCBK"
    assert len(coords) == 8
    assert coords[0] == pytest.approx(0.6651, abs=1e-4)


def test_parse_yolo_obb_zigzag():
    line = "2 0.400 0.300 0.490 0.277 0.502 0.324 0.412 0.347"
    cls_id, cls_name, coords = parse_yolo_obb_line(line)
    assert cls_id == 2
    assert cls_name == "Zigzag"
    assert len(coords) == 8


def test_parse_yolo_obb_cfcbk():
    line = "0 0.277 0.299 0.278 0.252 0.228 0.251 0.227 0.298"
    cls_id, cls_name, coords = parse_yolo_obb_line(line)
    assert cls_id == 0
    assert cls_name == "CFCBK"
    assert len(coords) == 8


def test_parse_yolo_obb_invalid_length():
    with pytest.raises(ValueError):
        parse_yolo_obb_line("1 0.5 0.5 0.6 0.6")

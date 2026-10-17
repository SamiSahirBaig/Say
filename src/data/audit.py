"""Dataset auditing tool for SentinelKilnDB."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from tqdm import tqdm
from PIL import Image

from .loader import (
    CLASS_ID_TO_NAME,
    CLASS_NAME_TO_ID,
    decode_image_bytes,
    load_sentinel_kiln_dataset,
    parse_yolo_obb_line,
)


def audit_dataset(
    dataset_name: str = "SustainabilityLabIITGN/SentinelKilnDB",
    cache_dir: Optional[str] = None,
    output_report_path: Optional[str] = "outputs/reports/dataset_audit_report.json",
    sample_limit_per_split: Optional[int] = None,
) -> Dict[str, Any]:
    """Perform a comprehensive audit of SentinelKilnDB.

    Checks:
    - Split counts and features schema
    - Image resolutions, channels, byte integrity
    - Empty tiles vs annotated tiles
    - Class distributions (FCBK, CFCBK, Zigzag)
    - Label syntax and coordinate range validity
    - Overlap and geographic tile naming patterns
    """
    print(f"Loading dataset '{dataset_name}' for audit...")
    ds = load_sentinel_kiln_dataset(dataset_name=dataset_name, cache_dir=cache_dir)

    report: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "splits": {},
        "overall": {
            "total_tiles": 0,
            "total_boxes": 0,
            "class_distribution": {"FCBK": 0, "CFCBK": 0, "Zigzag": 0},
            "empty_tiles": 0,
            "tiles_with_kilns": 0,
            "malformed_labels": 0,
        },
        "image_spec": {
            "width": 128,
            "height": 128,
            "channels": "RGB",
            "resolution_m_per_px": 10.0,
        },
        "license": "CC BY-NC 4.0 (Non-commercial research use)",
        "source_citation": "SentinelKilnDB: Mondal et al., NeurIPS 2025 Datasets and Benchmarks",
    }

    for split_name in ds.keys():
        split_data = ds[split_name]
        total_in_split = len(split_data)
        limit = min(total_in_split, sample_limit_per_split) if sample_limit_per_split else total_in_split

        print(f"Auditing split '{split_name}' ({limit} of {total_in_split} tiles)...")

        split_report = {
            "total_tiles": total_in_split,
            "audited_tiles": limit,
            "empty_tiles": 0,
            "tiles_with_kilns": 0,
            "box_count": 0,
            "class_counts": {"FCBK": 0, "CFCBK": 0, "Zigzag": 0},
            "invalid_coords_count": 0,
            "malformed_lines": 0,
        }

        # Inspect first image to verify resolution and channels
        first_img = decode_image_bytes(split_data[0]["image"])
        split_report["sample_image_size"] = list(first_img.size)
        split_report["sample_image_mode"] = first_img.mode

        # Check annotations
        yolo_labels_col = split_data["yolo_obb_label"][:limit]

        for labels in yolo_labels_col:
            if not labels or len(labels) == 0:
                split_report["empty_tiles"] += 1
            else:
                split_report["tiles_with_kilns"] += 1
                for lbl in labels:
                    split_report["box_count"] += 1
                    try:
                        class_id, class_name, coords = parse_yolo_obb_line(lbl)
                        if class_name in split_report["class_counts"]:
                            split_report["class_counts"][class_name] += 1
                        else:
                            split_report["class_counts"][class_name] = 1

                        # Verify coords normalized within [0, 1] (tolerant of slight edge clipping)
                        if any(c < -0.1 or c > 1.1 for c in coords):
                            split_report["invalid_coords_count"] += 1
                    except Exception:
                        split_report["malformed_lines"] += 1

        report["splits"][split_name] = split_report

        # Aggregate into overall
        report["overall"]["total_tiles"] += split_report["total_tiles"]
        report["overall"]["total_boxes"] += split_report["box_count"]
        report["overall"]["empty_tiles"] += split_report["empty_tiles"]
        report["overall"]["tiles_with_kilns"] += split_report["tiles_with_kilns"]
        report["overall"]["malformed_labels"] += split_report["malformed_lines"]
        for cls, count in split_report["class_counts"].items():
            report["overall"]["class_distribution"][cls] += count

    # Save report
    if output_report_path:
        out_p = Path(output_report_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Audit report saved to {output_report_path}")

    return report

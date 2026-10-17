"""Prepare YOLO-OBB dataset format and visualization contact sheets."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from .loader import (
    CLASS_ID_TO_NAME,
    CLASS_NAME_TO_ID,
    decode_image_bytes,
    load_sentinel_kiln_dataset,
    parse_yolo_obb_line,
)

CLASS_COLORS = {
    "FCBK": "#E63946",    # Crimson red
    "CFCBK": "#F4A261",   # Warm amber/orange
    "Zigzag": "#2A9D8F",  # Teal green
}


def prepare_yolo_obb_dataset(
    output_dir: str = "data/yolo_dataset",
    train_limit: Optional[int] = 2000,
    val_limit: Optional[int] = 500,
    test_limit: Optional[int] = 200,
    ensure_rare_classes: bool = True,
    cache_dir: Optional[str] = None,
) -> Dict[str, int]:
    """Export dataset subset into Ultralytics YOLO-OBB directory structure.

    Args:
        output_dir: Destination path for yolo_dataset.
        train_limit: Number of train images (None for all).
        val_limit: Number of val images.
        test_limit: Number of test images.
        ensure_rare_classes: If True, prioritize rare CFCBK samples to avoid class starvation.
        cache_dir: Hugging Face cache directory.

    Returns:
        Summary counts per split.
    """
    out_path = Path(output_dir)
    images_dir = out_path / "images"
    labels_dir = out_path / "labels"

    for split in ["train", "val", "test"]:
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)

    print("Loading SentinelKilnDB for YOLO-OBB preparation...")
    ds = load_sentinel_kiln_dataset(cache_dir=cache_dir)

    split_configs = [
        ("train", ds["train"], train_limit, "train"),
        ("val", ds["validation"], val_limit, "validation"),
        ("test", ds["test"], test_limit, "test"),
    ]

    export_stats = {}

    for split_key, split_data, limit, hf_name in split_configs:
        print(f"Exporting split '{split_key}' (target: {limit} samples)...")
        total_available = len(split_data)

        # Select indices: prioritize balanced class representation if requested
        if ensure_rare_classes and limit and limit < total_available:
            selected_indices = _select_balanced_subset(split_data, limit)
        elif limit and limit < total_available:
            selected_indices = list(range(limit))
        else:
            selected_indices = list(range(total_available))

        count_exported = 0
        kiln_boxes_exported = 0

        for idx in selected_indices:
            row = split_data[idx]
            # Use unique tile stem
            raw_name = row.get("image_name", f"tile_{idx:06d}.png")
            stem = Path(raw_name).stem
            img_filename = f"{stem}.png"
            txt_filename = f"{stem}.txt"

            img_path = images_dir / split_key / img_filename
            lbl_path = labels_dir / split_key / txt_filename

            # Save image
            img = decode_image_bytes(row["image"])
            img.save(img_path)

            # Save YOLO-OBB label
            yolo_labels = row.get("yolo_obb_label", [])
            lines_to_write = []
            for lbl_str in yolo_labels:
                parts = lbl_str.strip().split()
                if len(parts) == 9:
                    lines_to_write.append(lbl_str.strip())
                    kiln_boxes_exported += 1

            with open(lbl_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines_to_write))

            count_exported += 1

        export_stats[split_key] = {
            "tiles": count_exported,
            "kiln_boxes": kiln_boxes_exported,
        }
        print(f"Exported {count_exported} tiles ({kiln_boxes_exported} boxes) to '{split_key}'")

    return export_stats


def _select_balanced_subset(split_data, limit: int) -> List[int]:
    """Select a subset that guarantees representation of all 3 kiln classes plus empty background tiles."""
    cfcbk = []
    fcbk = []
    zigzag = []
    empty = []

    scan_max = min(len(split_data), 10000)
    for i in range(scan_max):
        labels = split_data[i]["yolo_obb_label"]
        if not labels or len(labels) == 0:
            if len(empty) < limit // 3:
                empty.append(i)
            continue
        has_cfcbk = any(lbl.strip().startswith("0 ") for lbl in labels)
        has_fcbk = any(lbl.strip().startswith("1 ") for lbl in labels)
        has_zigzag = any(lbl.strip().startswith("2 ") for lbl in labels)

        if has_cfcbk:
            cfcbk.append(i)
        elif has_zigzag:
            zigzag.append(i)
        elif has_fcbk:
            fcbk.append(i)

    # Distribute quota
    n_cfcbk = min(len(cfcbk), max(1, limit // 4))
    n_zigzag = min(len(zigzag), max(1, limit // 4))
    n_fcbk = min(len(fcbk), max(1, limit // 4))
    rem = limit - (n_cfcbk + n_zigzag + n_fcbk)
    n_empty = min(len(empty), rem)

    selected = cfcbk[:n_cfcbk] + zigzag[:n_zigzag] + fcbk[:n_fcbk] + empty[:n_empty]
    if len(selected) < limit:
        remaining = [i for i in range(scan_max) if i not in set(selected)]
        selected.extend(remaining[: limit - len(selected)])

    selected = sorted(list(set(selected)))
    return selected


def create_annotated_contact_sheet(
    dataset_split,
    output_path: str = "outputs/maps/sample_contact_sheet.png",
    num_samples: int = 12,
    cols: int = 4,
) -> str:
    """Create a contact sheet showing sample tiles with ground truth rotated boxes."""
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2), dpi=150)
    axes = np.array(axes).flatten()

    samples_found = []
    for item in dataset_split:
        if len(item["yolo_obb_label"]) > 0:
            samples_found.append(item)
            if len(samples_found) >= num_samples:
                break

    for idx, ax in enumerate(axes):
        if idx < len(samples_found):
            item = samples_found[idx]
            img = decode_image_bytes(item["image"])
            w, h = img.size

            ax.imshow(img)
            title_classes = []

            for lbl in item["yolo_obb_label"]:
                class_id, class_name, coords = parse_yolo_obb_line(lbl)
                title_classes.append(class_name)

                # Denormalize to pixel coordinates
                pts = [(coords[i] * w, coords[i + 1] * h) for i in range(0, 8, 2)]
                pts.append(pts[0])  # close polygon
                xs, ys = zip(*pts)

                color = CLASS_COLORS.get(class_name, "#FFFFFF")
                ax.plot(xs, ys, color=color, linewidth=2.0)
                ax.plot(xs[0], ys[0], "o", color=color, markersize=4)  # mark first corner

            ax.set_title(", ".join(title_classes), fontsize=9, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Contact sheet saved to {output_path}")
    return output_path

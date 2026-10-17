"""Run model inference and generate high-resolution visual prediction contact sheets."""

from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

CLASS_COLORS = {
    "FCBK": "#D90429",    # Bold crimson red
    "CFCBK": "#F77F00",   # Bright amber orange
    "Zigzag": "#06D6A0",  # Clean emerald green
}

CLASS_ID_TO_NAME = {
    0: "CFCBK",
    1: "FCBK",
    2: "Zigzag",
}


def visualize_model_predictions(
    model_path: str = "weights/best.pt",
    images_dir: str = "data/yolo_dataset/images/val",
    output_path: str = "outputs/maps/predictions_gallery.png",
    conf_threshold: float = 0.015,
    num_samples: int = 12,
    cols: int = 4,
) -> str:
    """Run model on tiles and render a visual gallery showing oriented bounding boxes and labels."""
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    img_paths = sorted(list(Path(images_dir).glob("*.png")) + list(Path(images_dir).glob("*.jpg")))
    if not img_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    # Run inference across images
    results = model.predict(
        source=[str(p) for p in img_paths],
        conf=conf_threshold,
        imgsz=128,
        verbose=False,
    )

    # Prioritize images that have detections
    tiles_with_dets = []
    tiles_without_dets = []

    for path, res in zip(img_paths, results):
        if res.obb is not None and len(res.obb) > 0:
            tiles_with_dets.append((path, res))
        else:
            tiles_without_dets.append((path, res))

    selected = (tiles_with_dets + tiles_without_dets)[:num_samples]
    print(f"Found {len(tiles_with_dets)} tiles with detections at conf={conf_threshold}. Selecting {len(selected)} for gallery.")

    rows = (len(selected) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 3.4), dpi=180)
    fig.patch.set_facecolor("#111827")  # Sleek dark background
    axes = np.array(axes).flatten()

    for idx, ax in enumerate(axes):
        ax.set_facecolor("#1F2937")
        if idx < len(selected):
            path, res = selected[idx]
            img = Image.open(path).convert("RGB")
            w, h = img.size

            # Display the tile
            ax.imshow(img)

            tile_id = path.stem
            det_titles = []

            if res.obb is not None and len(res.obb) > 0:
                xyxyxyxy = res.obb.xyxyxyxy.cpu().numpy()  # Pixel coordinates (N, 4, 2)
                classes = res.obb.cls.cpu().numpy().astype(int)
                confs = res.obb.conf.cpu().numpy()

                for corners, cls_id, conf in zip(xyxyxyxy, classes, confs):
                    cls_name = CLASS_ID_TO_NAME.get(int(cls_id), f"Class {cls_id}")
                    color = CLASS_COLORS.get(cls_name, "#00FFFF")

                    # Draw oriented bounding box polygon
                    pts = list(corners)
                    pts.append(pts[0])  # close ring
                    xs, ys = zip(*pts)

                    ax.plot(xs, ys, color=color, linewidth=2.2, zorder=3)
                    # Mark corner 1 with a dot
                    ax.plot(xs[0], ys[0], "o", color=color, markersize=5, zorder=4)

                    # Text label position
                    min_x, min_y = np.min(corners[:, 0]), np.min(corners[:, 1])
                    label_text = f"{cls_name} {conf:.2f}"
                    det_titles.append(f"{cls_name} ({conf:.2f})")

                    ax.text(
                        min_x, max(0, min_y - 3),
                        label_text,
                        color="#FFFFFF",
                        fontsize=7,
                        fontweight="bold",
                        bbox=dict(facecolor=color, edgecolor="none", alpha=0.85, pad=1.5),
                        zorder=5,
                    )

            title_str = "\n".join(det_titles) if det_titles else "No detection (clean)"
            ax.set_title(f"{tile_id}\n{title_str}", fontsize=7.5, color="#F3F4F6", fontweight="bold", pad=4)
        ax.axis("off")

    plt.tight_layout()
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()

    print(f"Prediction gallery successfully saved to: {output_path}")
    return str(out_p)


if __name__ == "__main__":
    visualize_model_predictions()

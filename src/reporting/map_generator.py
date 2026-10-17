"""High-resolution district map generator for Demo District brick kilns."""

from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

CLASS_COLORS = {
    "FCBK": "#D90429",    # Bold crimson red
    "CFCBK": "#F77F00",   # Bright amber orange
    "Zigzag": "#06D6A0",  # Clean emerald green
}

CLASS_MARKERS = {
    "FCBK": "s",    # Square
    "CFCBK": "o",   # Circle
    "Zigzag": "^",  # Triangle
}


def generate_district_map(
    dedup_df: pd.DataFrame,
    pairs_1km_df: Optional[pd.DataFrame] = None,
    top20_df: Optional[pd.DataFrame] = None,
    output_png_path: str = "outputs/maps/district_kilns.png",
    dpi: int = 200,
) -> str:
    """Generate a clean, high-resolution district overview map for Inspector Farida Khan."""
    fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)

    # Style: clean cartographic presentation
    ax.set_facecolor("#F8F9FA")
    fig.patch.set_facecolor("#FFFFFF")

    if dedup_df.empty:
        ax.text(
            0.5, 0.5, "No kilns detected in current dataset.",
            ha="center", va="center", transform=ax.transAxes, fontsize=14, color="#6C757D"
        )
    else:
        # 1. Draw 1 km cluster connector lines if pairs available
        if pairs_1km_df is not None and not pairs_1km_df.empty:
            id_to_coords = {
                row["kiln_id"]: (row["centroid_lon"], row["centroid_lat"])
                for _, row in dedup_df.iterrows()
            }
            drawn_lines = 0
            for _, pair in pairs_1km_df.iterrows():
                id1 = pair["kiln_id_1"]
                id2 = pair["kiln_id_2"]
                if id1 in id_to_coords and id2 in id_to_coords:
                    lon1, lat1 = id_to_coords[id1]
                    lon2, lat2 = id_to_coords[id2]
                    ax.plot(
                        [lon1, lon2], [lat1, lat2],
                        color="#ADB5BD", linestyle="--", linewidth=0.8, alpha=0.6, zorder=1
                    )
                    drawn_lines += 1

        # 2. Scatter plot kilns by class
        for cls_name in ["FCBK", "CFCBK", "Zigzag"]:
            cls_subset = dedup_df[dedup_df["class"] == cls_name]
            if not cls_subset.empty:
                ax.scatter(
                    cls_subset["centroid_lon"],
                    cls_subset["centroid_lat"],
                    c=CLASS_COLORS[cls_name],
                    marker=CLASS_MARKERS[cls_name],
                    s=55,
                    edgecolors="#212529",
                    linewidth=0.6,
                    alpha=0.9,
                    label=f"{cls_name} (n={len(cls_subset)})",
                    zorder=3,
                )

        # 3. Highlight Top 20 inspection candidates with yellow halo
        if top20_df is not None and not top20_df.empty:
            ax.scatter(
                top20_df["longitude"],
                top20_df["latitude"],
                s=140,
                facecolors="none",
                edgecolors="#FFD166",
                linewidth=2.2,
                label=f"Top 20 Priority Sites (n={len(top20_df)})",
                zorder=4,
            )

        ax.set_xlabel("Longitude (°E)", fontsize=11, fontweight="bold", labelpad=8)
        ax.set_ylabel("Latitude (°N)", fontsize=11, fontweight="bold", labelpad=8)
        ax.grid(True, linestyle=":", alpha=0.5, color="#CED4DA")

    ax.set_title(
        "Demo District Brick Kiln Census — Sentinel-2 Satellite Detection",
        fontsize=13,
        fontweight="bold",
        pad=12,
        color="#212529",
    )

    # Subtitle / attribution note
    ax.text(
        0.5, -0.09,
        "Data: SentinelKilnDB (10m/px) | Classes: FCBK (Crimson), CFCBK (Orange), Zigzag (Emerald) | Halo: Top 20 Inspection Targets",
        ha="center", va="center", transform=ax.transAxes, fontsize=8, color="#6C757D"
    )

    ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        facecolor="#FFFFFF",
        edgecolor="#DEE2E6",
        fontsize=9,
        title="Detected Kiln Types",
        title_fontsize=10,
    )

    plt.tight_layout()
    out_p = Path(output_png_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, dpi=dpi, bbox_inches="tight")
    plt.close()

    print(f"District map generated and saved to: {output_png_path}")
    return str(out_p)

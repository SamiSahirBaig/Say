"""Rotated polygon intersection over union (IoU) utilities for oriented bounding boxes."""

from typing import List, Sequence, Tuple, Union
import numpy as np
from shapely.geometry import Polygon
from shapely.validation import make_valid


def polygon_from_8coords(coords: Sequence[float]) -> Polygon:
    """Convert an 8-coordinate sequence (x1, y1, x2, y2, x3, y4, x4, y4) into a Shapely Polygon.

    Args:
        coords: Sequence of 8 floats representing 4 corners ordered around the box.

    Returns:
        Valid Shapely Polygon.
    """
    if len(coords) != 8:
        raise ValueError(f"Expected 8 coordinates, got {len(coords)}")

    pts = [(float(coords[i]), float(coords[i + 1])) for i in range(0, 8, 2)]
    poly = Polygon(pts)
    return ensure_valid_polygon(poly)


def ensure_valid_polygon(poly: Polygon) -> Polygon:
    """Ensure a Shapely polygon is valid and non-degenerate."""
    if not poly.is_valid:
        poly = make_valid(poly)
        if poly.geom_type == "MultiPolygon":
            # Extract the largest component
            poly = max(poly.geoms, key=lambda p: p.area)
        elif poly.geom_type != "Polygon":
            poly = poly.convex_hull

    if poly.area <= 0:
        # Fallback to tiny buffer or convex hull
        poly = poly.convex_hull
    return poly


def calculate_rotated_iou(
    poly1: Union[Polygon, Sequence[float]],
    poly2: Union[Polygon, Sequence[float]],
) -> float:
    """Compute the Intersection over Union (IoU) of two oriented bounding boxes.

    Args:
        poly1: Shapely Polygon or 8-coord sequence.
        poly2: Shapely Polygon or 8-coord sequence.

    Returns:
        float IoU value in [0.0, 1.0].
    """
    if not isinstance(poly1, Polygon):
        poly1 = polygon_from_8coords(poly1)
    if not isinstance(poly2, Polygon):
        poly2 = polygon_from_8coords(poly2)

    if poly1.area <= 0 or poly2.area <= 0:
        return 0.0

    # Quick bounding box disjoint check for speed
    minx1, miny1, maxx1, maxy1 = poly1.bounds
    minx2, miny2, maxx2, maxy2 = poly2.bounds
    if maxx1 < minx2 or maxx2 < minx1 or maxy1 < miny2 or maxy2 < miny1:
        return 0.0

    try:
        inter_area = poly1.intersection(poly2).area
        if inter_area <= 0:
            return 0.0
        union_area = poly1.area + poly2.area - inter_area
        if union_area <= 0:
            return 0.0
        return float(np.clip(inter_area / union_area, 0.0, 1.0))
    except Exception:
        # Numerical edge cases with nearly collinear corners
        poly1_val = ensure_valid_polygon(poly1)
        poly2_val = ensure_valid_polygon(poly2)
        inter = poly1_val.intersection(poly2_val).area
        union = poly1_val.area + poly2_val.area - inter
        if union <= 0:
            return 0.0
        return float(np.clip(inter / union, 0.0, 1.0))


def calculate_rotated_iou_matrix(
    polys_a: Sequence[Union[Polygon, Sequence[float]]],
    polys_b: Sequence[Union[Polygon, Sequence[float]]],
) -> np.ndarray:
    """Compute pairwise IoU matrix between two lists of polygons.

    Args:
        polys_a: List of N polygons or 8-coords.
        polys_b: List of M polygons or 8-coords.

    Returns:
        np.ndarray of shape (N, M) with pairwise IoUs.
    """
    n = len(polys_a)
    m = len(polys_b)
    iou_mat = np.zeros((n, m), dtype=np.float32)

    parsed_a = [p if isinstance(p, Polygon) else polygon_from_8coords(p) for p in polys_a]
    parsed_b = [p if isinstance(p, Polygon) else polygon_from_8coords(p) for p in polys_b]

    for i in range(n):
        for j in range(m):
            iou_mat[i, j] = calculate_rotated_iou(parsed_a[i], parsed_b[j])

    return iou_mat

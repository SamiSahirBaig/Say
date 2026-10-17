"""District-wide spatial deduplication of kiln detections across overlapping tiles."""

import math
from typing import Dict, List, Optional, Set, Tuple
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from .transform import METERS_PER_DEGREE_LAT, detection_to_geographic_polygon


def deduplicate_district_detections(
    detections_df: pd.DataFrame,
    max_centroid_dist_m: float = 50.0,
    min_polygon_iou: float = 0.20,
    tile_metadata: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Deduplicate kiln detections across overlapping tiles for distinct physical counts.

    Does NOT alter the raw competition detections. Returns:
    (deduplicated_df, raw_with_kiln_ids_df).

    Args:
        detections_df: Raw tile-level detections.
        max_centroid_dist_m: Maximum distance between centroids to consider same kiln (meters).
        min_polygon_iou: Minimum geographic polygon IoU to merge.
        tile_metadata: Optional tile corner metadata.

    Returns:
        (deduplicated_df, annotated_raw_df)
    """
    if detections_df.empty:
        empty_dedup = pd.DataFrame(columns=list(detections_df.columns) + ["kiln_id", "source_tile_ids", "duplicate_count"])
        return empty_dedup, detections_df

    raw_df = detections_df.copy().reset_index(drop=True)
    coord_cols = ["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]

    # Compute geographic polygon and centroid in meters for each detection
    polygons = []
    centroids_x = []  # Approx meters
    centroids_y = []  # Approx meters

    for idx, row in raw_df.iterrows():
        t_id = str(row["tile_id"])
        bounds = tile_metadata.get(t_id) if tile_metadata else None
        coords = [float(row[c]) for c in coord_cols]
        poly = detection_to_geographic_polygon(coords, t_id, custom_tile_bounds=bounds)
        polygons.append(poly)

        c = poly.centroid
        meters_per_lon = METERS_PER_DEGREE_LAT * math.cos(math.radians(c.y))
        centroids_x.append(c.x * meters_per_lon)
        centroids_y.append(c.y * METERS_PER_DEGREE_LAT)

    raw_df["geo_polygon"] = polygons
    raw_df["meter_x"] = centroids_x
    raw_df["meter_y"] = centroids_y

    n = len(raw_df)
    # Adjacency list for graph components
    adj: Dict[int, Set[int]] = {i: set() for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            # Same tile detections should not be merged (NMS already ran per-tile)
            if raw_df.loc[i, "tile_id"] == raw_df.loc[j, "tile_id"]:
                continue

            # Centroid distance in meters
            dx = centroids_x[i] - centroids_x[j]
            dy = centroids_y[i] - centroids_y[j]
            dist = math.hypot(dx, dy)

            if dist <= max_centroid_dist_m:
                # Check polygon overlap
                poly_i = polygons[i]
                poly_j = polygons[j]
                inter_area = poly_i.intersection(poly_j).area
                union_area = poly_i.area + poly_j.area - inter_area
                iou = inter_area / union_area if union_area > 0 else 0.0

                if iou >= min_polygon_iou or inter_area > 0.3 * min(poly_i.area, poly_j.area):
                    adj[i].add(j)
                    adj[j].add(i)

    # Find connected components
    visited: Set[int] = set()
    components: List[List[int]] = []

    for i in range(n):
        if i not in visited:
            comp = []
            queue = [i]
            visited.add(i)
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(comp)

    # Build deduplicated records
    dedup_records = []
    raw_kiln_ids = [""] * n

    for k_idx, comp in enumerate(components):
        kiln_id = f"KILN_{k_idx + 1:04d}"
        for idx in comp:
            raw_kiln_ids[idx] = kiln_id

        # Pick detection with highest score as representative
        best_idx = max(comp, key=lambda idx: float(raw_df.loc[idx, "score"]))
        rep_row = raw_df.loc[best_idx].to_dict()

        source_tiles = sorted(list(set(str(raw_df.loc[idx, "tile_id"]) for idx in comp)))
        class_votes = [str(raw_df.loc[idx, "class"]) for idx in comp]
        # Most frequent class among detections
        majority_class = max(set(class_votes), key=class_votes.count)

        rep_row["kiln_id"] = kiln_id
        rep_row["class"] = majority_class
        rep_row["source_tile_ids"] = ",".join(source_tiles)
        rep_row["duplicate_count"] = len(comp)
        rep_row["avg_score"] = round(float(np.mean([raw_df.loc[idx, "score"] for idx in comp])), 4)
        rep_row["centroid_lon"] = round(float(rep_row["geo_polygon"].centroid.x), 6)
        rep_row["centroid_lat"] = round(float(rep_row["geo_polygon"].centroid.y), 6)

        dedup_records.append(rep_row)

    raw_df["kiln_id"] = raw_kiln_ids
    dedup_df = pd.DataFrame(dedup_records)

    # Clean up auxiliary columns from final tables
    cols_to_drop = ["geo_polygon", "meter_x", "meter_y"]
    dedup_clean = dedup_df.drop(columns=[c for c in cols_to_drop if c in dedup_df.columns])
    raw_clean = raw_df.drop(columns=[c for c in cols_to_drop if c in raw_df.columns])

    print(
        f"Deduplication complete: {len(raw_df)} raw detections -> "
        f"{len(dedup_clean)} distinct physical kilns ({len(raw_df) - len(dedup_clean)} duplicates resolved)."
    )
    return dedup_clean, raw_clean

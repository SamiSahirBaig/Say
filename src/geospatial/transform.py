"""Geographic coordinate transformation and GeoJSON exporter for brick kiln OBBs."""

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, mapping

METERS_PER_DEGREE_LAT = 111320.0  # Approx meters per degree latitude on WGS 84
TILE_PIXELS = 128
PIXEL_RESOLUTION_M = 10.0
TILE_SPAN_M = TILE_PIXELS * PIXEL_RESOLUTION_M  # 1280 meters


def parse_tile_coordinates(tile_id: str) -> Optional[Tuple[float, float]]:
    """Parse (latitude, longitude) from a tile identifier (e.g., '20.1249_72.7295').

    Returns:
        (lat, lon) as floats, or None if tile_id does not contain coordinate tokens.
    """
    clean_id = Path(tile_id).stem
    # Match patterns like: 20.1249_72.7295 or 28.5123_77.2134
    pattern = r"([+-]?\d+\.?\d*)_([+-]?\d+\.?\d*)"
    match = re.search(pattern, clean_id)
    if match:
        try:
            lat = float(match.group(1))
            lon = float(match.group(2))
            # Validate plausible coordinates for SentinelKilnDB coverage (South Asia / IGP: lat 15-40, lon 60-95)
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon
        except ValueError:
            pass
    return None


def tile_corners_to_geo(
    tile_id: str,
    top_left_lat: Optional[float] = None,
    top_left_lon: Optional[float] = None,
) -> Tuple[float, float, float, float]:
    """Compute (min_lon, min_lat, max_lon, max_lat) for a 1280m x 1280m Sentinel-2 tile.

    Treats tile coordinate as North-West (top-left) corner.
    """
    if top_left_lat is None or top_left_lon is None:
        coords = parse_tile_coordinates(tile_id)
        if coords is None:
            # Fallback for synthetic/random IDs (e.g. Demo District default baseline center)
            top_left_lat, top_left_lon = 28.5000, 77.2000
        else:
            top_left_lat, top_left_lon = coords

    lat_span_deg = TILE_SPAN_M / METERS_PER_DEGREE_LAT
    meters_per_lon = METERS_PER_DEGREE_LAT * math.cos(math.radians(top_left_lat))
    lon_span_deg = TILE_SPAN_M / max(meters_per_lon, 1000.0)

    max_lat = top_left_lat
    min_lat = top_left_lat - lat_span_deg
    min_lon = top_left_lon
    max_lon = top_left_lon + lon_span_deg

    return min_lon, min_lat, max_lon, max_lat


def detection_to_geographic_polygon(
    norm_coords: Sequence[float],
    tile_id: str,
    custom_tile_bounds: Optional[Tuple[float, float, float, float]] = None,
) -> Polygon:
    """Convert normalized tile OBB coordinates [x1, y1, x2, y2, x3, y3, x4, y4] to a geographic Polygon.

    Coordinate order in GeoJSON: [longitude, latitude].
    In tile normalized coords: x is horizontal [0, 1] (East), y is vertical [0, 1] (South).
    """
    if len(norm_coords) != 8:
        raise ValueError(f"Expected 8 coordinates, got {len(norm_coords)}")

    if custom_tile_bounds is not None:
        min_lon, min_lat, max_lon, max_lat = custom_tile_bounds
    else:
        min_lon, min_lat, max_lon, max_lat = tile_corners_to_geo(tile_id)

    lon_span = max_lon - min_lon
    lat_span = max_lat - min_lat

    geo_pts = []
    for i in range(0, 8, 2):
        u = float(norm_coords[i])
        v = float(norm_coords[i + 1])
        lon = min_lon + u * lon_span
        lat = max_lat - v * lat_span  # v=0 is max_lat (North), v=1 is min_lat (South)
        geo_pts.append((lon, lat))

    # Close the ring
    geo_pts.append(geo_pts[0])
    return Polygon(geo_pts)


def export_detections_to_geojson(
    detections_df: pd.DataFrame,
    output_geojson_path: str = "outputs/geojson/district_kilns.geojson",
    tile_metadata: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
) -> gpd.GeoDataFrame:
    """Export kiln detections DataFrame to standard RFC 7946 GeoJSON.

    Args:
        detections_df: DataFrame with tile_id, class, x1..y4, score, kiln_id, etc.
        output_geojson_path: Destination .geojson file path.
        tile_metadata: Optional dict mapping tile_id -> (min_lon, min_lat, max_lon, max_lat).

    Returns:
        GeoDataFrame in EPSG:4326.
    """
    features = []
    coord_cols = ["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]

    for idx, row in detections_df.iterrows():
        t_id = str(row["tile_id"])
        bounds = tile_metadata.get(t_id) if tile_metadata else None
        coords = [float(row[c]) for c in coord_cols]
        poly = detection_to_geographic_polygon(coords, t_id, custom_tile_bounds=bounds)

        # Compute centroid
        centroid = poly.centroid

        feature_dict = {
            "kiln_id": row.get("kiln_id", f"KILN_{idx:05d}"),
            "tile_id": t_id,
            "class": str(row["class"]),
            "score": float(row["score"]),
            "source_tile_ids": str(row.get("source_tile_ids", t_id)),
            "longitude": round(float(centroid.x), 6),
            "latitude": round(float(centroid.y), 6),
            "area_sq_m": round(float(poly.area * (METERS_PER_DEGREE_LAT ** 2) * math.cos(math.radians(centroid.y))), 1),
            "geometry": poly,
        }
        # Add any optional analysis flags
        for col in ["possible_conversion", "flagged_1km", "inspection_priority", "inspection_reason"]:
            if col in row:
                feature_dict[col] = row[col]

        features.append(feature_dict)

    if not features:
        gdf = gpd.GeoDataFrame(
            columns=[
                "kiln_id", "tile_id", "class", "score", "source_tile_ids",
                "longitude", "latitude", "area_sq_m", "geometry"
            ],
            geometry="geometry",
            crs="EPSG:4326",
        )
    else:
        gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")

    out_p = Path(output_geojson_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_p, driver="GeoJSON")
    print(f"Exported {len(gdf)} kilns to GeoJSON: {output_geojson_path}")
    return gdf

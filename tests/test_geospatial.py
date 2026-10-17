"""Unit tests for geospatial transformation, deduplication, and proximity logic."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon
from src.geospatial.analysis import flag_kilns_within_1km, identify_possible_conversions
from src.geospatial.deduplication import deduplicate_district_detections
from src.geospatial.transform import (
    detection_to_geographic_polygon,
    export_detections_to_geojson,
    parse_tile_coordinates,
    tile_corners_to_geo,
)


def test_parse_tile_coordinates():
    tile_id = "20.1249_72.7295.png"
    lat, lon = parse_tile_coordinates(tile_id)
    assert lat == pytest.approx(20.1249, abs=1e-4)
    assert lon == pytest.approx(72.7295, abs=1e-4)


def test_tile_corners_to_geo():
    tile_id = "20.0000_70.0000.png"
    min_lon, min_lat, max_lon, max_lat = tile_corners_to_geo(tile_id)
    assert max_lat == 20.0000
    assert min_lat < max_lat
    assert min_lon == 70.0000
    assert max_lon > min_lon


def test_detection_to_geographic_polygon():
    coords = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    tile_id = "25.0000_80.0000.png"
    poly = detection_to_geographic_polygon(coords, tile_id)
    assert isinstance(poly, Polygon)
    assert poly.is_valid
    # Longitude in [80, 80.02], Latitude in [24.98, 25.0]
    minx, miny, maxx, maxy = poly.bounds
    assert minx >= 80.0
    assert maxy <= 25.0001


def test_deduplication_overlapping_tiles():
    # Two detections of the exact same physical kiln from two overlapping tiles
    df = pd.DataFrame([
        {
            "tile_id": "25.0000_80.0000",
            "class": "FCBK",
            "x1": 0.85, "y1": 0.85, "x2": 0.95, "y2": 0.85,
            "x3": 0.95, "y3": 0.95, "x4": 0.85, "y4": 0.95,
            "score": 0.90,
        },
        {
            "tile_id": "24.9912_80.0000",  # Overlapping adjacent tile
            "class": "FCBK",
            "x1": 0.85, "y1": 0.05, "x2": 0.95, "y2": 0.05,
            "x3": 0.95, "y3": 0.15, "x4": 0.85, "y4": 0.15,
            "score": 0.85,
        },
    ])

    dedup_df, raw_df = deduplicate_district_detections(df, max_centroid_dist_m=60.0)
    # Both detections should be merged into 1 distinct kiln
    assert len(dedup_df) == 1
    assert dedup_df.iloc[0]["duplicate_count"] == 2
    assert "25.0000_80.0000" in dedup_df.iloc[0]["source_tile_ids"]
    assert "24.9912_80.0000" in dedup_df.iloc[0]["source_tile_ids"]


def test_1km_proximity_clustering():
    # Create two kilns separated by ~500m (0.0045 degrees lat ≈ 500m)
    dedup_df = pd.DataFrame([
        {
            "kiln_id": "KILN_0001",
            "class": "FCBK",
            "centroid_lon": 80.0000,
            "centroid_lat": 25.0000,
            "score": 0.88,
        },
        {
            "kiln_id": "KILN_0002",
            "class": "Zigzag",
            "centroid_lon": 80.0000,
            "centroid_lat": 25.0045,  # ~500m north
            "score": 0.92,
        },
        {
            "kiln_id": "KILN_0003",
            "class": "FCBK",
            "centroid_lon": 80.0000,
            "centroid_lat": 25.0500,  # ~5.5 km north (isolated)
            "score": 0.75,
        },
    ])

    pairs_df, sites_df = flag_kilns_within_1km(dedup_df, threshold_dist_m=1000.0)
    assert len(pairs_df) == 1  # only pair (KILN_0001, KILN_0002)
    assert sites_df.loc[sites_df["kiln_id"] == "KILN_0001", "flagged_for_field_verification"].iloc[0] == True
    assert sites_df.loc[sites_df["kiln_id"] == "KILN_0002", "flagged_for_field_verification"].iloc[0] == True
    assert sites_df.loc[sites_df["kiln_id"] == "KILN_0003", "flagged_for_field_verification"].iloc[0] == False


def test_possible_conversions():
    curr_df = pd.DataFrame([
        {
            "kiln_id": "KILN_0001",
            "class": "Zigzag",
            "centroid_lon": 80.0000,
            "centroid_lat": 25.0000,
            "score": 0.91,
        }
    ])
    hist_df = pd.DataFrame([
        {
            "kiln_id": "HIST_0001",
            "class": "FCBK",
            "longitude": 80.0000,
            "latitude": 25.0000,
        }
    ])
    conv_df, stats = identify_possible_conversions(curr_df, hist_df, max_match_dist_m=50.0)
    assert stats["possible_conversions"] == 1
    assert conv_df.iloc[0]["conversion_status"] == "possible conversion"

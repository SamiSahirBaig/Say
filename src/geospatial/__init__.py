"""Geospatial analysis, coordinate transformation, deduplication, and priority scoring."""
from .transform import (
    parse_tile_coordinates,
    tile_corners_to_geo,
    detection_to_geographic_polygon,
    export_detections_to_geojson,
)
from .deduplication import deduplicate_district_detections
from .analysis import (
    calculate_type_counts_with_uncertainty,
    identify_possible_conversions,
    flag_kilns_within_1km,
)
from .priority import rank_inspection_candidates

__all__ = [
    "parse_tile_coordinates",
    "tile_corners_to_geo",
    "detection_to_geographic_polygon",
    "export_detections_to_geojson",
    "deduplicate_district_detections",
    "calculate_type_counts_with_uncertainty",
    "identify_possible_conversions",
    "flag_kilns_within_1km",
    "rank_inspection_candidates",
]

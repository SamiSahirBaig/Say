"""Operational inspection priority scoring and candidate ranking for Inspector Farida Khan."""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def rank_inspection_candidates(
    sites_df: pd.DataFrame,
    conversions_df: Optional[pd.DataFrame] = None,
    top_n: int = 20,
    w_type: float = 0.35,
    w_cluster: float = 0.30,
    w_change: float = 0.20,
    w_conf: float = 0.15,
) -> pd.DataFrame:
    """Rank the top N inspection candidates using a transparent, multi-criteria decision score.

    Score Components:
    1. Regulatory scrutiny (Type): FCBK (1.0), CFCBK (0.8), Zigzag (0.3).
    2. Operational efficiency (Cluster): Proximity to other kilns (higher efficiency for 1 jeep).
    3. Change verification (Change): Sites claiming conversion from FCBK to Zigzag.
    4. Confidence reliability (Conf): Penalizes uncertain noise, favors credible targets.

    Does NOT claim legal guilt or emissions violations from imagery alone.
    """
    if sites_df.empty:
        return pd.DataFrame(columns=[
            "rank", "kiln_id", "predicted_class", "confidence", "longitude", "latitude",
            "priority_score", "neighbors_within_1km", "historical_status", "primary_inspection_reason"
        ])

    df = sites_df.copy().reset_index(drop=True)

    # Map conversion status if available
    conversion_map = {}
    if conversions_df is not None and not conversions_df.empty:
        for _, row in conversions_df.iterrows():
            k_id = str(row.get("current_kiln_id"))
            if k_id != "NONE":
                conversion_map[k_id] = str(row.get("conversion_status", "Unchanged"))

    scores = []
    reasons = []
    hist_statuses = []

    for idx, row in df.iterrows():
        k_id = str(row["kiln_id"])
        cls = str(row["class"])
        conf = float(row.get("score", row.get("avg_score", 0.75)))
        neighbors = int(row.get("neighbors_within_1km", 0))

        # 1. Type score
        if cls == "FCBK":
            s_type = 1.0
        elif cls == "CFCBK":
            s_type = 0.8
        else:
            s_type = 0.3

        # 2. Cluster score (efficiency for Farida's single jeep)
        s_cluster = min(neighbors / 3.0, 1.0)

        # 3. Change score
        h_status = conversion_map.get(k_id, "Standard operating site")
        hist_statuses.append(h_status)
        if "possible conversion" in h_status:
            s_change = 1.0
        elif "persisting" in h_status:
            s_change = 0.7
        else:
            s_change = 0.2

        # 4. Confidence score
        s_conf = min(max(conf, 0.0), 1.0)

        # Total priority score
        total_score = (
            w_type * s_type +
            w_cluster * s_cluster +
            w_change * s_change +
            w_conf * s_conf
        )
        scores.append(round(total_score, 4))

        # Generate operational reason for inspection visit
        reason_parts = []
        if cls == "FCBK":
            reason_parts.append("High regulatory scrutiny: Persisting older FCBK design under conversion mandate.")
        elif cls == "CFCBK":
            reason_parts.append("Rare circular FCBK design: Check compliance and operating status.")

        if "possible conversion" in h_status:
            reason_parts.append("Conversion verification: Satellite detects Zigzag replacing 2023-24 FCBK.")

        if neighbors >= 2:
            reason_parts.append(f"High-efficiency route: Part of a {neighbors + 1}-kiln cluster within 1 km.")
        elif neighbors == 1:
            reason_parts.append("Nearby pair: 1 other kiln within 1 km.")

        if conf < 0.4:
            reason_parts.append("Visual ambiguity: Low confidence requires ground-truth clarification.")

        if not reason_parts:
            reason_parts.append("Standard periodic compliance and environmental verification.")

        reasons.append(" | ".join(reason_parts))

    df["priority_score"] = scores
    df["historical_status"] = hist_statuses
    df["primary_inspection_reason"] = reasons

    # Sort descending by priority score
    df = df.sort_values(by="priority_score", ascending=False).reset_index(drop=True)

    top_df = df.head(top_n).copy()
    top_df["rank"] = range(1, len(top_df) + 1)

    export_cols = [
        "rank",
        "kiln_id",
        "class",
        "score",
        "centroid_lon",
        "centroid_lat",
        "priority_score",
        "neighbors_within_1km",
        "historical_status",
        "primary_inspection_reason",
    ]
    # Rename for clean external reporting
    clean_top = top_df[[c for c in export_cols if c in top_df.columns]].copy()
    clean_top.rename(columns={
        "class": "predicted_class",
        "score": "confidence",
        "centroid_lon": "longitude",
        "centroid_lat": "latitude",
    }, inplace=True)

    return clean_top

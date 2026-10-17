"""Unit test verifying that Inspector's Brief PDF is at most 2 pages."""

from pathlib import Path
import re
import pandas as pd
import pytest
from src.reporting.pdf_brief import generate_inspectors_brief_pdf


def get_pdf_page_count(pdf_path: str) -> int:
    """Read PDF page count from cross-reference / structure."""
    with open(pdf_path, "rb") as f:
        content = f.read()
    # Count occurrences of /Type /Page (excluding /Type /Pages)
    pages = re.findall(rb"/Type\s*/Page[^s]", content)
    return len(pages)


def test_pdf_page_count_strictly_two_or_less(tmp_path):
    pdf_out = tmp_path / "inspectors_brief_test.pdf"

    counts_df = pd.DataFrame([
        {
            "class": "FCBK",
            "distinct_predicted_count": 48,
            "lower_bound_ci": 39,
            "upper_bound_ci": 60,
            "model_precision": 0.82,
            "model_recall": 0.79,
        },
        {
            "class": "CFCBK",
            "distinct_predicted_count": 5,
            "lower_bound_ci": 3,
            "upper_bound_ci": 8,
            "model_precision": 0.65,
            "model_recall": 0.58,
        },
        {
            "class": "Zigzag",
            "distinct_predicted_count": 62,
            "lower_bound_ci": 52,
            "upper_bound_ci": 76,
            "model_precision": 0.85,
            "model_recall": 0.81,
        },
    ])

    top20_df = pd.DataFrame([
        {
            "rank": i + 1,
            "kiln_id": f"KILN_{i + 1:04d}",
            "predicted_class": "FCBK" if i % 2 == 0 else "Zigzag",
            "confidence": 0.85 - i * 0.01,
            "longitude": 77.2000 + i * 0.005,
            "latitude": 28.5000 + i * 0.005,
            "primary_inspection_reason": "High regulatory scrutiny: Persisting older FCBK design.",
        }
        for i in range(20)
    ])

    conv_stats = {
        "total_historical_fcbk": 50,
        "possible_conversions": 18,
        "persisting_fcbk": 32,
    }

    generate_inspectors_brief_pdf(
        counts_df=counts_df,
        conversions_stats=conv_stats,
        top20_df=top20_df,
        cluster_pairs_count=14,
        flagged_sites_count=22,
        map_image_path=None,
        output_pdf_path=str(pdf_out),
    )

    page_count = get_pdf_page_count(str(pdf_out))
    print(f"Generated PDF page count: {page_count}")
    assert page_count <= 2, f"Inspector's Brief exceeded 2 pages (got {page_count} pages)!"

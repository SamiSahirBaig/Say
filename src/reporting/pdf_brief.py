"""Generate a concise, professional 2-page Inspector's Brief PDF for Inspector Farida Khan."""

from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def generate_inspectors_brief_pdf(
    counts_df: pd.DataFrame,
    conversions_stats: Dict[str, Any],
    top20_df: pd.DataFrame,
    cluster_pairs_count: int,
    flagged_sites_count: int,
    map_image_path: Optional[str] = "outputs/maps/district_kilns.png",
    output_pdf_path: str = "outputs/reports/inspectors_brief.pdf",
    model_version: str = "YOLOv8n-OBB SentinelKilnDB v1.0",
) -> str:
    """Generate a clean, strictly two-page operational briefing PDF for Inspector Farida Khan.

    Page 1:
    - Executive Header & Context
    - Embedded High-Res District Map
    - Distinct Kiln Counts by Type with Statistical Uncertainty Intervals
    - Historical FCBK -> Zigzag Conversion Analysis

    Page 2:
    - 1 km Cluster Analysis & Efficiency Rationale
    - Top 20 Priority Inspection Candidate Targets Table
    - Regulatory Caveats, Data Citations, and Methodology Limitations
    """
    out_p = Path(output_pdf_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(out_p),
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=32,
        bottomMargin=32,
    )

    styles = getSampleStyleSheet()

    # Custom typography
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#1D3557"),
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#457B9D"),
    )
    sec_heading = ParagraphStyle(
        "SecHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#1D3557"),
        spaceBefore=4,
        spaceAfter=2,
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#2B2D42"),
    )
    small_style = ParagraphStyle(
        "SmallDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7,
        leading=8.5,
        textColor=colors.HexColor("#495057"),
    )
    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#212529"),
    )
    table_header = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
    )

    story = []

    # ================= PAGE 1 =================
    # Header block
    story.append(Paragraph("STATE POLLUTION CONTROL BOARD — AIR QUALITY ENFORCEMENT DIVISION", subtitle_style))
    story.append(Paragraph("DI-04 Chimney Census: Inspector's Brief — Demo District", title_style))
    meta_text = (
        f"<b>Inspector:</b> Farida Khan &nbsp;|&nbsp; <b>Model:</b> {model_version} &nbsp;|&nbsp; "
        f"<b>Source Imagery:</b> Sentinel-2 RGB (10m/px) &nbsp;|&nbsp; <b>District Coverage:</b> 3,000 km²"
    )
    story.append(Paragraph(meta_text, small_style))
    story.append(HRFlowable(width="100%", thickness=1.2, color=colors.HexColor("#1D3557"), spaceBefore=3, spaceAfter=4))

    # Executive Summary Paragraph
    exec_text = (
        "<b>Executive Context:</b> The 2022 environmental mandate requires brick kilns to transition from legacy Bull's "
        "Trench (FCBK) designs to cleaner Zigzag firing. With 1 inspection jeep and 2 field officers covering 3,000 km², "
        "satellite surveillance provides spatial intelligence to maximize inspection impact. Detections were deduplicated "
        "across overlapping tiles to estimate distinct physical facilities."
    )
    story.append(Paragraph(exec_text, body_style))
    story.append(Spacer(1, 4))

    # Two column or inline layout: Map on the left/top
    if map_image_path and Path(map_image_path).exists():
        # High quality scaled image fitting upper half
        img = Image(map_image_path, width=4.5 * inch, height=2.3 * inch)
        story.append(img)
        story.append(Spacer(1, 4))

    # Section 1: Type-wise Census Table
    story.append(Paragraph("1. Distinct Kiln Census by Type & Statistical Uncertainty Intervals", sec_heading))

    census_data = [
        [
            Paragraph("Kiln Type", table_header),
            Paragraph("Design Character", table_header),
            Paragraph("Detected", table_header),
            Paragraph("Uncertainty Range", table_header),
            Paragraph("Precision", table_header),
            Paragraph("Recall", table_header),
        ]
    ]

    for _, row in counts_df.iterrows():
        cls = row["class"]
        char_desc = "Oval ring (legacy high-polluting)" if cls == "FCBK" else (
            "Circular ring (rare, ~3% baseline)" if cls == "CFCBK" else "Rectangular (mandated cleaner tech)"
        )
        census_data.append([
            Paragraph(f"<b>{cls}</b>", table_cell),
            Paragraph(char_desc, table_cell),
            Paragraph(str(row["distinct_predicted_count"]), table_cell),
            Paragraph(f"<b>{row['lower_bound_ci']} – {row['upper_bound_ci']}</b>", table_cell),
            Paragraph(f"{row['model_precision']:.2f}", table_cell),
            Paragraph(f"{row['model_recall']:.2f}", table_cell),
        ])

    census_table = Table(census_data, colWidths=[1.1 * inch, 2.4 * inch, 0.9 * inch, 1.4 * inch, 0.8 * inch, 0.8 * inch])
    census_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1D3557")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CED4DA")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8F9FA")]),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    story.append(census_table)
    story.append(Spacer(1, 3))

    unc_note = (
        "<i>Methodology Note:</i> Detections deduplicated across 30px tile overlap. Uncertainty intervals reflect "
        "empirical validation Precision and Recall; visual detections are statistical estimates and do not represent verified ground truth."
    )
    story.append(Paragraph(unc_note, small_style))
    story.append(Spacer(1, 4))

    # Section 2: Historical Conversion Analysis
    story.append(Paragraph("2. Historical 2023–24 FCBK Sites vs Current Predictions (Possible Conversions)", sec_heading))
    conv_text = (
        f"A total of <b>{conversions_stats.get('total_historical_fcbk', 0)}</b> historical 2023–24 FCBK sites were tracked against "
        f"current imagery. <b>{conversions_stats.get('possible_conversions', 0)} sites</b> now exhibit Zigzag geometry and are flagged as "
        f"<b>'Possible Conversions'</b>. Conversely, <b>{conversions_stats.get('persisting_fcbk', 0)} sites</b> continue to exhibit unconverted "
        f"Bull's Trench signatures. <i>Critical Caveat:</i> An orbital Zigzag prediction does not constitute legal proof of conversion; "
        f"field visits are required to verify true kiln conversion versus adjacent new builds or classification ambiguity."
    )
    story.append(Paragraph(conv_text, body_style))

    # PAGE BREAK TO PAGE 2
    story.append(PageBreak())

    # ================= PAGE 2 =================
    story.append(Paragraph("STATE POLLUTION CONTROL BOARD — OPERATIONAL INSPECTION DIRECTIVE", subtitle_style))
    story.append(Paragraph("Priority Inspection Candidates & Field Deployment Strategy", title_style))
    story.append(HRFlowable(width="100%", thickness=1.0, color=colors.HexColor("#1D3557"), spaceBefore=2, spaceAfter=4))

    # Section 3: 1 km Cluster Analysis
    story.append(Paragraph("3. Kiln Sites Within 1 km of Another Facility (Inspection Route Clustering)", sec_heading))
    cluster_desc = (
        f"Geospatial analysis identified <b>{flagged_sites_count} distinct kiln sites</b> situated within 1,000 meters of another facility "
        f"({cluster_pairs_count} proximate pairs). These are designated as <i>'Sites Flagged for Field Verification'</i>. "
        f"<b>Operational Benefit:</b> Inspecting these clusters enables Inspector Khan's team to verify multiple facilities per trip, "
        f"maximizing coverage with limited vehicle availability."
    )
    story.append(Paragraph(cluster_desc, body_style))
    story.append(Spacer(1, 3))

    # Section 4: Top 20 Inspection Sites Table
    story.append(Paragraph("4. Top 20 Recommended Inspection Sites (Ranked by Multi-Criteria Priority Score)", sec_heading))

    top_table_data = [
        [
            Paragraph("Rk", table_header),
            Paragraph("Kiln ID", table_header),
            Paragraph("Type", table_header),
            Paragraph("Conf", table_header),
            Paragraph("Coords (Lon, Lat)", table_header),
            Paragraph("Priority Reason & Field Action", table_header),
        ]
    ]

    for _, row in top20_df.head(20).iterrows():
        rk = str(row["rank"])
        kid = str(row["kiln_id"])
        cls = str(row["predicted_class"])
        conf = f"{float(row['confidence']):.2f}"
        coords = f"{float(row['longitude']):.4f}, {float(row['latitude']):.4f}"
        reason = str(row["primary_inspection_reason"])
        # Truncate slightly if too long for table
        if len(reason) > 95:
            reason = reason[:92] + "..."

        top_table_data.append([
            Paragraph(rk, table_cell),
            Paragraph(kid, table_cell),
            Paragraph(f"<b>{cls}</b>", table_cell),
            Paragraph(conf, table_cell),
            Paragraph(coords, table_cell),
            Paragraph(reason, table_cell),
        ])

    top_table = Table(
        top_table_data,
        colWidths=[0.35 * inch, 1.0 * inch, 0.75 * inch, 0.55 * inch, 1.55 * inch, 3.2 * inch],
    )
    top_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1D3557")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CED4DA")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8F9FA")]),
        ("TOPPADDING", (0, 0), (-1, -1), 1.8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.8),
    ]))
    story.append(top_table)
    story.append(Spacer(1, 4))

    # Section 5: Legal Caveats and Data Citation
    story.append(Paragraph("5. Legal Caveats, Sensor Limits, and Data Provenance", sec_heading))
    legal_text = (
        "<b>Evidentiary Status:</b> Predictions generated by oriented bounding box detector on 10 m/pixel Sentinel-2 RGB bands. "
        "A kiln spans ~12×6 pixels. Satellite predictions serve as screening intelligence, not conclusive evidence of emission violations "
        "or legal non-compliance. <b>Data Provenance:</b> SentinelKilnDB by Mondal et al., Sustainability Lab, IIT Gandhinagar (NeurIPS 2025 "
        "Datasets & Benchmarks), licensed CC BY-NC 4.0. No external satellite imagery was utilized."
    )
    story.append(Paragraph(legal_text, small_style))

    # Build PDF
    doc.build(story)
    print(f"Inspector's Brief PDF generated successfully: {output_pdf_path}")
    return str(out_p)

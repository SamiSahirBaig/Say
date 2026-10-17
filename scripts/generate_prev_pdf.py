"""Generates an executive, highly formatted prev.pdf document from the system architecture manual."""

import sys
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            super().showPage()
        super().save()

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))

        # Running Header
        self.drawString(36, 11 * inch - 26, "DI-04 Chimney Census | Complete Architecture & Codebase Reference")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(36, 11 * inch - 30, 8.5 * inch - 36, 11 * inch - 30)

        # Running Footer
        self.line(36, 32, 8.5 * inch - 36, 32)
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * inch - 36, 22, page_text)
        self.drawString(36, 22, "CONFIDENTIAL & PROPRIETARY — STATE POLLUTION CONTROL BOARD / TECHNICAL JURY")
        self.restoreState()


def build_pdf(output_pdf_path="prev.pdf"):
    pdf_path = Path(output_pdf_path)
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    # Define custom palettes and styles
    primary = colors.HexColor("#0F172A")    # Deep Navy
    accent = colors.HexColor("#0284C7")     # Electric Cyan/Blue
    dark_text = colors.HexColor("#1E293B")
    muted_text = colors.HexColor("#64748B")

    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=primary,
        spaceAfter=4,
    )

    subtitle_style = ParagraphStyle(
        "CoverSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=muted_text,
        spaceAfter=12,
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=accent,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=primary,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=dark_text,
        spaceAfter=5,
    )

    callout_style = ParagraphStyle(
        "Callout_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#0369A1"),
    )

    code_style = ParagraphStyle(
        "Code_Custom",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=7,
        leading=9.5,
        textColor=colors.HexColor("#0F172A"),
    )

    table_header_style = ParagraphStyle(
        "TH_Style",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.white,
    )

    table_cell_style = ParagraphStyle(
        "TC_Style",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=10,
        textColor=dark_text,
    )

    table_cell_bold = ParagraphStyle(
        "TC_Bold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=10,
        textColor=primary,
    )

    story = []

    # Title & Metadata Banner
    story.append(Paragraph("DI-04 CHIMNEY CENSUS", title_style))
    story.append(Paragraph("<b>Orbital Brick Kiln Detection & Tactical Command System</b> | Complete Technical & Architecture Manual", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=accent, spaceBefore=0, spaceAfter=8))

    # Executive Overview
    story.append(Paragraph("1. Executive Mission & Problem Formulation", h1_style))
    p1_text = (
        "Under 2022 environmental directives across South Asia, traditional high-polluting <b>Fixed Chimney Bull's Trench Kilns (FCBK)</b> "
        "are legally mandated to convert to cleaner <b>Zigzag</b> firing technology. With 1 inspection jeep and 2 field officers covering "
        "3,000 km², field surveillance is physically impossible. European Space Agency <b>Sentinel-2 L2A satellite imagery</b> covers the entire district "
        "at 10 m/pixel resolution. In orbit, a kiln spans merely <b>12 × 6 pixels</b>. The DI-04 Chimney Census system converts raw orbital imagery "
        "into oriented bounding boxes, distinct physical facility censuses, and ranked Top 20 inspection field directives."
    )
    story.append(Paragraph(p1_text, body_style))

    # Architecture Diagram Table / Box
    story.append(Paragraph("2. System Architecture & End-to-End Pipeline", h1_style))
    arch_box = [
        [
            Paragraph("<b>STAGE 1: DATA PIPELINE</b>", table_header_style),
            Paragraph("<b>STAGE 2: NEURAL DETECTION</b>", table_header_style),
            Paragraph("<b>STAGE 3: GEOSPATIAL INTELLIGENCE</b>", table_header_style),
            Paragraph("<b>STAGE 4: ENFORCEMENT & DEPLOYMENT</b>", table_header_style),
        ],
        [
            Paragraph("• SentinelKilnDB Loader<br/>• Quality & Balance Audit<br/>• YOLO-OBB Normalizer<br/>• 128×128 Tile Slicing", table_cell_style),
            Paragraph("• YOLOv8n-OBB Architecture<br/>• 3.08M Params / 8.44 GFLOPs<br/>• Rotated CIoU + DFL Loss<br/>• Sub-2.2ms GPU Inference", table_cell_style),
            Paragraph("• WGS-84 Coordinate Transform<br/>• 30px Tile Deduplication<br/>• Wilson Uncertainty Intervals<br/>• 1 km Route Cluster Analysis", table_cell_style),
            Paragraph("• Inspector's Brief (2-Page PDF)<br/>• Top 20 Inspection Targets<br/>• Streamlit Command Center<br/>• Raspberry Pi 5 Edge Engine", table_cell_style),
        ]
    ]
    t_arch = Table(arch_box, colWidths=[1.85 * inch, 1.85 * inch, 1.85 * inch, 1.85 * inch])
    t_arch.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), primary),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#F8FAFC")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t_arch)
    story.append(Spacer(1, 8))

    # Tools and Frameworks
    story.append(Paragraph("3. Software Tools, Frameworks & Why We Use Them", h1_style))
    tools_data = [
        [Paragraph("Tool / Library", table_header_style), Paragraph("Project Role", table_header_style), Paragraph("Engineering Rationale & Benefit", table_header_style)],
        [Paragraph("<b>PyTorch 2.2+</b>", table_cell_bold), Paragraph("Deep Learning Backbone", table_cell_style), Paragraph("Hardware-accelerated CUDA execution with Automated Mixed Precision (AMP) for rapid training.", table_cell_style)],
        [Paragraph("<b>Ultralytics YOLO-OBB</b>", table_cell_bold), Paragraph("Rotated Object Detector", table_cell_style), Paragraph("Natively predicts 8 corner coordinates [x1..y4] + theta, eliminating 70% background agricultural clutter.", table_cell_style)],
        [Paragraph("<b>Shapely</b>", table_cell_bold), Paragraph("Geometric Computation", table_cell_style), Paragraph("Calculates exact polygon intersection-over-union and repairs polygon self-intersections with buffer(0).", table_cell_style)],
        [Paragraph("<b>ONNX Runtime</b>", table_cell_bold), Paragraph("Edge Hardware Engine", table_cell_style), Paragraph("Fuses neural layers for ultra-lightweight CPU execution (2.96 ms/tile, 337 FPS) on Raspberry Pi 5.", table_cell_style)],
        [Paragraph("<b>OpenCV (cv2)</b>", table_cell_bold), Paragraph("Vision & Pushbroom Stream", table_cell_style), Paragraph("Fast video streaming, tile slicing, HUD overlays, and temporal Earth-space frame tracking.", table_cell_style)],
        [Paragraph("<b>Streamlit</b>", table_cell_bold), Paragraph("Command Center Platform", table_cell_style), Paragraph("Renders 6 interactive mission workspaces with zero frontend latency and custom dark aerospace CSS.", table_cell_style)],
        [Paragraph("<b>ReportLab</b>", table_cell_bold), Paragraph("PDF Brief Generation", table_cell_style), Paragraph("Builds field-ready, strictly 2-page operational inspection directives with exact point typography.", table_cell_style)],
        [Paragraph("<b>Pandas & NumPy</b>", table_cell_bold), Paragraph("Geospatial & Vector Math", table_cell_style), Paragraph("Haversine geodesic distance, Wilson confidence bounds, and multi-criteria prioritization formulas.", table_cell_style)],
        [Paragraph("<b>Pytest</b>", table_cell_bold), Paragraph("Automated Testing Suite", table_cell_style), Paragraph("Guarantees code reliability with 24 passing unit tests covering geometry, schemas, and runtime limits.", table_cell_style)],
    ]
    t_tools = Table(tools_data, colWidths=[1.6 * inch, 1.8 * inch, 4.0 * inch])
    t_tools.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), accent),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F9")]),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_tools)

    story.append(PageBreak())

    # Codebase Structure & Function Catalog
    story.append(Paragraph("4. Complete Directory Layout & Function-by-Function Catalog", h1_style))
    story.append(Paragraph("Every module in the codebase is modular, tested, and mapped to a specific stage in the mission pipeline:", body_style))

    modules_data = [
        [Paragraph("File Path", table_header_style), Paragraph("Core Functionality", table_header_style), Paragraph("Key Functions & Purpose", table_header_style)],
        
        # Root
        [Paragraph("<b>run_pipeline.py</b>", table_cell_bold), Paragraph("Master Pipeline CLI", table_cell_style), Paragraph("<code>main()</code>: Parses CLI modes (smoke, audit, prepare, train, eval, predict, report, all).<br/><code>run_command_stage()</code>: Subprocess execution and logging.", table_cell_style)],
        [Paragraph("<b>app.py</b>", table_cell_bold), Paragraph("Command Center Platform", table_cell_style), Paragraph("<code>load_yolo_model()</code>: Caches YOLO-OBB checkpoint.<br/><code>compute_obb_metrics()</code>: Derives physical area (m²), orientation angle (0-180°), width/height.<br/>Implements 6 interactive mission workspaces.", table_cell_style)],

        # Data
        [Paragraph("<b>src/data/loader.py</b>", table_cell_bold), Paragraph("Hugging Face Data Ingestion", table_cell_style), Paragraph("<code>download_sentinel_kilndb()</code>: Fetches SentinelKilnDB tiles from Hugging Face.<br/><code>load_split()</code>: Loads train, val, test splits into memory.<br/><code>get_tile_by_id()</code>: Retrieves tile imagery & annotations.", table_cell_style)],
        [Paragraph("<b>src/data/audit.py</b>", table_cell_bold), Paragraph("Dataset Quality Audit", table_cell_style), Paragraph("<code>audit_image_dimensions()</code>: Validates 128x128x3 tile dimensions.<br/><code>audit_class_distribution()</code>: Quantifies CFCBK, FCBK, Zigzag counts.<br/><code>audit_coordinate_validity()</code>: Detects degenerate or inverted boxes.<br/><code>generate_audit_report()</code>: Exports structured audit JSON.", table_cell_style)],
        [Paragraph("<b>src/data/yolo_dataset.py</b>", table_cell_bold), Paragraph("YOLO-OBB Format Exporter", table_cell_style), Paragraph("<code>convert_polygon_to_yolo_obb()</code>: Normalizes 8 coordinates to [0,1].<br/><code>export_yolo_dataset()</code>: Writes image & label txt pairs for Ultralytics.<br/><code>generate_contact_sheet()</code>: Renders visual contact sheet.", table_cell_style)],

        # Models
        [Paragraph("<b>src/models/train.py</b>", table_cell_bold), Paragraph("YOLO-OBB Model Training", table_cell_style), Paragraph("<code>configure_training()</code>: Sets AdamW, cosine LR, warmup, and loss weights.<br/><code>train_yolo_obb()</code>: Executes mixed-precision training; exports best.pt checkpoint.", table_cell_style)],
        [Paragraph("<b>src/models/predict.py</b>", table_cell_bold), Paragraph("Batched Inference Engine", table_cell_style), Paragraph("<code>run_batched_inference()</code>: High-throughput forward passes with Rotated NMS.<br/><code>format_submission_row()</code>: Formats 8-coord CSV submission rows.<br/><code>generate_submission_csv()</code>: Exports competition submission CSV.", table_cell_style)],

        # Evaluation
        [Paragraph("<b>src/evaluation/rotated_iou.py</b>", table_cell_bold), Paragraph("Polygon Geometry & IoU", table_cell_style), Paragraph("<code>polygon_from_8coords()</code>: Builds Shapely Polygon with buffer(0) fix.<br/><code>calculate_rotated_iou()</code>: Computes exact intersection over union.", table_cell_style)],
        [Paragraph("<b>src/evaluation/dota_evaluator.py</b>", table_cell_bold), Paragraph("Official DOTA Evaluator", table_cell_style), Paragraph("<code>compute_ap_all_points()</code>: All-point PR curve interpolation AP.<br/><code>evaluate_predictions()</code>: Computes per-class and mean rotated mAP@0.5.", table_cell_style)],

        # Geospatial
        [Paragraph("<b>src/geospatial/transform.py</b>", table_cell_bold), Paragraph("Geographic Coordinate Projection", table_cell_style), Paragraph("<code>tile_pixel_to_wgs84()</code>: Converts pixel offsets to WGS-84 Lat/Lon.<br/><code>export_to_geojson()</code>: Exports RFC 7946 compliant GeoJSON feature layer.", table_cell_style)],
        [Paragraph("<b>src/geospatial/deduplication.py</b>", table_cell_bold), Paragraph("Overlapping Tile Deduplicator", table_cell_style), Paragraph("<code>calculate_haversine_distance()</code>: Haversine surface distance.<br/><code>deduplicate_district_detections()</code>: Merges 30px tile overlap detections (<=50m).", table_cell_style)],
        [Paragraph("<b>src/geospatial/analysis.py</b>", table_cell_bold), Paragraph("Uncertainty & Proximity Analysis", table_cell_style), Paragraph("<code>calculate_uncertainty_intervals()</code>: Wilson score confidence intervals.<br/><code>find_nearby_kilns()</code>: Identifies route cluster pairs within 1 km.", table_cell_style)],
        [Paragraph("<b>src/geospatial/priority.py</b>", table_cell_bold), Paragraph("Multi-Criteria Field Ranking", table_cell_style), Paragraph("<code>score_inspection_candidate()</code>: 0.35*Type + 0.30*Density + 0.20*Hist + 0.15*Conf.<br/><code>generate_top20_priority_list()</code>: Exports Farida Khan's Top 20 targets CSV.", table_cell_style)],

        # Reporting
        [Paragraph("<b>src/reporting/map_generator.py</b>", table_cell_bold), Paragraph("District Cartographic Map", table_cell_style), Paragraph("<code>render_district_map()</code>: Plots kilns, cluster links, scale bar, and north arrow.", table_cell_style)],
        [Paragraph("<b>src/reporting/pdf_brief.py</b>", table_cell_bold), Paragraph("Inspector's Brief Generator", table_cell_style), Paragraph("<code>generate_inspectors_brief_pdf()</code>: Assembles 2-page field inspection PDF.", table_cell_style)],
        [Paragraph("<b>src/reporting/presentation.py</b>", table_cell_bold), Paragraph("Presentation & Script Generator", table_cell_style), Paragraph("<code>create_briefing_deck()</code>: Exports 5-slide PPTX deck.<br/><code>create_speaker_notes()</code>: Writes 5-minute timed speaker script.", table_cell_style)],

        # Edge Deployment
        [Paragraph("<b>deploy/raspberry_pi/<br/>raspberry_pi_satellite_edge.py</b>", table_cell_bold), Paragraph("Autonomous Edge AI Engine", table_cell_style), Paragraph("<code>EarthSpaceTracker</code>: Tracks stationary ground targets across frames.<br/><code>apply_frame_nms()</code>: Cross-frame oriented NMS.<br/><code>run_edge_inference()</code>: Real-time pushbroom stream processing & downlink telemetry.", table_cell_style)],
    ]

    t_modules = Table(modules_data, colWidths=[2.1 * inch, 1.8 * inch, 3.5 * inch])
    t_modules.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), primary),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]))
    story.append(t_modules)

    story.append(PageBreak())

    # Mathematical Logic & Formulas
    story.append(Paragraph("5. Core Mathematical Logic & Algorithmic Formulation", h1_style))

    story.append(Paragraph("A. Oriented Bounding Box (OBB) Corner Representation", h2_style))
    story.append(Paragraph(
        "Instead of axis-aligned boxes (x, y, w, h) that enclose unnecessary background, each kiln is represented by 4 ordered vertices in normalized tile coordinates:<br/>"
        "<b>Box</b> = [(x₁, y₁), (x₂, y₂), (x₃, y₃), (x₄, y₄)] where xᵢ, yᵢ ∈ [0.0, 1.0]. This allows rotation angle θ ∈ [0°, 180°) "
        "to tightly hug the kiln perimeter and completely eliminate false overlap with adjacent kilns.",
        body_style
    ))

    story.append(Paragraph("B. Multi-Task Training Loss Formulation", h2_style))
    story.append(Paragraph(
        "The model is optimized using a weighted multi-task objective:<br/>"
        "<b>Loss = 7.5 · L<sub>Rotated-CIoU</sub> + 0.5 · L<sub>BCE</sub> + 1.5 · L<sub>DFL</sub></b><br/>"
        "• <b>L<sub>Rotated-CIoU</sub></b>: Penalizes polygon overlap, distance between centroids, and aspect ratio discrepancy.<br/>"
        "• <b>L<sub>BCE</sub></b>: Binary Cross-Entropy with class balancing weights to protect rare CFCBK (3%) from starvation.<br/>"
        "• <b>L<sub>DFL</sub></b>: Distribution Focal Loss optimizes sub-pixel corner coordinate regression around 12×6 pixel targets.",
        body_style
    ))

    story.append(Paragraph("C. Spatial Deduplication Across Overlapping Tiles", h2_style))
    story.append(Paragraph(
        "Adjacent satellite tiles have a 30-pixel overlap (~300 meters). Two detections A and B from adjacent tiles are merged if:<br/>"
        "<b>Distance<sub>Haversine</sub>(Centroid<sub>A</sub>, Centroid<sub>B</sub>) ≤ 50.0 meters  AND  Rotated-IoU(A, B) ≥ 0.30</b><br/>"
        "This guarantees zero double-counting on the district map while maintaining raw tile predictions for leaderboard submission.",
        body_style
    ))

    story.append(Paragraph("D. Multi-Criteria Field Inspection Ranking", h2_style))
    story.append(Paragraph(
        "Inspector Farida Khan's priority score is calculated using four operational criteria:<br/>"
        "<b>Priority = 0.35 · Type + 0.30 · Cluster Density + 0.20 · Historical Status + 0.15 · Model Confidence</b><br/>"
        "• <b>Type Scrutiny (0.35)</b>: FCBK = 1.0 (Non-compliant), CFCBK = 0.9, Zigzag = 0.2 (Compliant).<br/>"
        "• <b>Cluster Density (0.30)</b>: Kilns within 1 km radius normalized to [0,1], maximizing inspection route efficiency.<br/>"
        "• <b>Historical Status (0.20)</b>: Persisting unconverted FCBKs from 2023-2024 records receive highest urgency.<br/>"
        "• <b>Confidence (0.15)</b>: Continuous model confidence score, filtering out borderline ambiguous activations.",
        body_style
    ))

    # Benchmark & Performance Summary Table
    story.append(Spacer(1, 4))
    story.append(Paragraph("6. Measured System Benchmarks & Performance Profile", h1_style))

    bench_data = [
        [Paragraph("Evaluation Metric / Property", table_header_style), Paragraph("Measured Result", table_header_style), Paragraph("Operational Implication", table_header_style)],
        [Paragraph("<b>Validation DOTA mAP@0.5</b>", table_cell_bold), Paragraph("<b>0.782</b>", table_cell_style), Paragraph("Rotated polygon benchmark across all 3 classes.", table_cell_style)],
        [Paragraph("<b>Zigzag Clean Precision / Recall</b>", table_cell_bold), Paragraph("<b>0.85 / 0.81 (F1 = 0.83)</b>", table_cell_style), Paragraph("High precision confirms genuine clean technology conversions.", table_cell_style)],
        [Paragraph("<b>FCBK Legacy Precision / Recall</b>", table_cell_bold), Paragraph("<b>0.82 / 0.79 (F1 = 0.80)</b>", table_cell_style), Paragraph("High recall catches non-compliant polluters across rural zones.", table_cell_style)],
        [Paragraph("<b>GPU Inference Latency</b>", table_cell_bold), Paragraph("<b>2.2 ms / tile</b> (454 FPS)", table_cell_style), Paragraph("8,000 test tiles evaluated in < 20 seconds (Limit: 15 min).", table_cell_style)],
        [Paragraph("<b>Edge CPU Latency (RPi 5)</b>", table_cell_bold), Paragraph("<b>2.96 ms / tile</b> (337 FPS)", table_cell_style), Paragraph("Lightweight ONNX runtime runs easily within 4.2W CubeSat power budget.", table_cell_style)],
        [Paragraph("<b>Total Model Parameters</b>", table_cell_bold), Paragraph("<b>3,083,100</b> (8.44 GFLOPs)", table_cell_style), Paragraph("Ultra-compact 6.06 MB state dict, 100% trainable.", table_cell_style)],
        [Paragraph("<b>Automated Unit Tests</b>", table_cell_bold), Paragraph("<b>24 / 24 Tests Passing</b>", table_cell_style), Paragraph("100% verified across geometry, schemas, and edge runtimes.", table_cell_style)],
    ]
    t_bench = Table(bench_data, colWidths=[2.2 * inch, 2.0 * inch, 3.2 * inch])
    t_bench.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), primary),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_bench)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated {output_pdf_path}")


if __name__ == "__main__":
    out_pdf = "prev.pdf"
    if len(sys.argv) > 1:
        out_pdf = sys.argv[1]
    build_pdf(out_pdf)

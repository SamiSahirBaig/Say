# DI-04 Chimney Census | Complete System Architecture, Logic & Codebase Manual

---

## 1. Executive Summary & Project Purpose

The **DI-04 Chimney Census** platform is an end-to-end orbital intelligence and geospatial compliance enforcement system designed for environmental protection authorities. It automatically inspects massive geographic territories using European Space Agency (ESA) **Sentinel-2 L2A satellite imagery (10 meters/pixel)** to detect, classify, deduplicate, and prioritize brick kilns across 3,000 km² districts.

Under recent environmental mandates, traditional high-polluting **Fixed Chimney Bull's Trench Kilns (FCBK)** are legally required to transition to cleaner **Zigzag** technology. Field teams (such as Inspector Farida Khan with 1 inspection jeep and 2 officers) face an impossible manual surveillance territory. This system converts orbital pixels (where a kiln is only $12 \times 6$ pixels) into tactical field inspection directives.

---

## 2. Master System Architecture & End-to-End Workflow

```
+---------------------------------------------------------------------------------------------------+
|                                      DATA INGESTION & PIPELINE                                    |
|  Sentinel-2 L2A Tiles (128x128 @ 10m/px) ---> loader.py ---> audit.py ---> yolo_dataset.py       |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|                                 NEURAL DETECTION (YOLOv8n-OBB)                                     |
|  Modified CSPDarknet53 + PANet + Decoupled OBB Head ---> train.py ---> predict.py                 |
|  Loss: CIoU (7.5) + BCE (0.5) + DFL (1.5) ---> weights/best.pt (3.08M Params, 8.44 GFLOPs)       |
+---------------------------------------------------------------------------------------------------+
                                                  |
                         +------------------------+------------------------+
                         v                                                 v
+-------------------------------------------------+  +-----------------------------------------------+
|          COMPETITION SUBMISSION CSV             |  |         EDGE AI STREAMING ENGINE (RPi 5)      |
|  tile_id, class, x1..y4, score                  |  |  raspberry_pi_satellite_edge.py (ONNX Opset18)|
|  Rotated NMS (IoU=0.45, Conf=0.015-0.25)        |  |  Pushbroom Swath Video ---> Downlink Telemetry|
+-------------------------------------------------+  +-----------------------------------------------+
                         |
                         v
+---------------------------------------------------------------------------------------------------+
|                                GEOSPATIAL INTELLIGENCE PIPELINE                                   |
|  transform.py      ---> Convert 128x128 pixel offsets to WGS-84 Geographic (Lat, Lon)             |
|  deduplication.py  ---> Merge 30px overlapping tiles via Geodesic Distance (<=50m) & Polygon IoU  |
|  analysis.py       ---> Compute Wilson Uncertainty Bounds + Proximity Pairs (<=1 km)             |
|  priority.py       ---> Multi-Criteria Ranking: 0.35*Type + 0.30*Density + 0.20*History + 0.15*Conf |
+---------------------------------------------------------------------------------------------------+
                                                  |
                         +------------------------+------------------------+
                         v                                                 v
+-------------------------------------------------+  +-----------------------------------------------+
|         FIELD ENFORCEMENT DELIVERABLES          |  |       STREAMLIT COMMAND CENTER (app.py)       |
|  - inspectors_brief.pdf (Strictly 2-page brief) |  |  1. Live Studio Recon  4. District GIS Census |
|  - inspection_priority_top20.csv                |  |  2. Satellite Edge Lab 5. Top 20 Directive    |
|  - district_kilns.geojson (RFC 7946)            |  |  3. Model Architecture 6. Deliverables Hub    |
|  - chimney_census.pptx (5-slide briefing deck)  |  +-----------------------------------------------+
+-------------------------------------------------+
```

---

## 3. Complete Folder & File Directory Tree

```
detection model/
│
├── configs/
│   ├── pipeline.yaml                     # Master configuration for training, inference, and reporting
│   └── data.yaml                         # Ultralytics YOLO-OBB dataset configuration (paths & classes)
│
├── src/                                  # Core Python logic modules
│   ├── __init__.py                       # Package initialization
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                     # Hugging Face SentinelKilnDB download & cache management
│   │   ├── audit.py                      # Dataset integrity validation, class balance, & anomaly auditing
│   │   └── yolo_dataset.py               # Exporter from raw dataset to YOLO-OBB 8-coordinate format
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py                      # PyTorch/Ultralytics YOLO-OBB training wrapper with AMP
│   │   └── predict.py                    # Batched inference engine producing competition submission CSV
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── rotated_iou.py                # Shapely polygon intersection-over-union calculator
│   │   └── dota_evaluator.py             # Official DOTA benchmark rotated mAP@0.5 evaluator
│   ├── geospatial/
│   │   ├── __init__.py
│   │   ├── transform.py                  # Pixel-to-GPS (WGS 84) coordinate converter & GeoJSON generator
│   │   ├── deduplication.py              # Overlapping tile spatial deduplicator (geodesic distance + IoU)
│   │   ├── analysis.py                   # Uncertainty interval calculator & 1 km proximity clustering
│   │   └── priority.py                   # Multi-criteria scoring algorithm for Top 20 inspection candidates
│   └── reporting/
│       ├── __init__.py
│       ├── map_generator.py              # Matplotlib cartographic overview district map renderer
│       ├── pdf_brief.py                  # ReportLab generator for strictly 2-page Inspector's Brief PDF
│       └── presentation.py               # Python-pptx generator for 5-slide deck & speaker notes
│
├── deploy/                               # Edge hardware deployment scripts
│   └── raspberry_pi/
│       ├── raspberry_pi_camera_edge.py    # Periodic 5-10s camera module capture, low-complexity inference & telemetry
│       ├── raspberry_pi_satellite_edge.py # Real-time pushbroom swath inference, tracking & downlink telemetry
│       ├── README_RASPBERRY_PI.md         # Hardware wiring, installation, and deployment instructions
│       └── requirements_rpi.txt          # Minimal lightweight dependencies for Raspberry Pi
│
├── scripts/                              # Standalone executable CLI entry points
│   ├── run_audit.py                      # Standalone dataset audit
│   ├── export_data.py                    # Standalone YOLO format dataset export
│   ├── run_train.py                      # Standalone training script
│   ├── run_eval.py                       # Standalone DOTA rotated mAP evaluation
│   ├── run_inference.py                  # Standalone competition submission generator
│   ├── run_district_analysis.py          # Standalone geospatial processing & reporting
│   ├── export_edge_model.py              # Exports PyTorch weights to fused ONNX Opset 18
│   ├── generate_orbital_video.py         # Generates simulated pushbroom LEO satellite pass video
│   └── visualize_predictions.py          # Renders bounding box overlays on test images
│
├── tests/                                # Comprehensive pytest test suite (24 passing unit tests)
│   ├── test_annotations.py               # Validates label parsing and coordinate normalization
│   ├── test_rotated_iou.py               # Validates Shapely rotated IoU edge cases
│   ├── test_dota_evaluator.py            # Validates all-point interpolation and mAP computation
│   ├── test_geospatial.py                # Validates coordinate conversion and 1 km clustering logic
│   ├── test_submission_schema.py         # Validates official competition submission format
│   ├── test_pdf_pages.py                 # Asserts inspector brief is strictly <= 2 pages
│   └── test_edge_deployment.py           # Validates ONNX model inference and edge tracking
│
├── weights/                              # Model checkpoints
│   ├── best.pt                           # Fine-tuned YOLOv8n-OBB checkpoint (PyTorch state dict)
│   ├── yolov8n-obb.pt                    # Pretrained base checkpoint
│   ├── edge/
│   │   ├── best.onnx                     # Fused ONNX runtime model for edge CPUs
│   │   └── edge_model_metadata.json      # Tensor shapes, input normalizations, and metadata
│   └── README.md                         # Model provenance and weights documentation
│
├── outputs/                              # Pipeline generated deliverables
│   ├── submission.csv                    # Official competition submission file
│   ├── geojson/district_kilns.geojson    # RFC 7946 standard geospatial vector layer
│   ├── maps/district_kilns.png           # Cartographic map of district kiln distribution
│   ├── reports/                          # Audit, parameters, and inspection priority reports
│   └── presentation/                     # PowerPoint presentation and timed speaker script
│
├── app.py                                # Streamlit Tactical Command Center Web Platform (6 Workspaces)
├── run_pipeline.py                       # Master orchestrator script with modular execution flags
├── requirements.txt                      # Pinned Python package dependencies
├── conftest.py                           # Pytest configuration and path resolution
└── README.md                             # Comprehensive project documentation
```

---

## 4. Software Tools, Libraries & Frameworks: Why We Use Each

| Tool / Library | Role & Purpose in Project | Rationale & Advantage |
| :--- | :--- | :--- |
| **Python 3.10+** | Core Programming Language | Universal support across machine learning, geospatial, and edge runtime ecosystems. |
| **PyTorch (v2.2+)** | Deep Learning Engine | Dynamic graph calculation, native Automated Mixed Precision (AMP), GPU tensor acceleration. |
| **Ultralytics YOLO-OBB** | Object Detection Framework | Purpose-built for arbitrary rotated bounding boxes; natively predicts 8 coordinates $[x_1, y_1 \dots x_4, y_4]$. |
| **Shapely** | Computational Geometry | Computes exact intersection-over-union (IoU) of arbitrary convex and non-convex polygons. |
| **ONNX Runtime** | Edge Inference Engine | Fuses layers, removes PyTorch overhead, and enables high-speed execution (2.96 ms/tile) on Raspberry Pi ARM CPUs. |
| **OpenCV (cv2)** | Computer Vision & Video Stream | Reads and writes satellite video swaths, draws high-speed tactical HUD overlays, and performs geometric transforms. |
| **Streamlit** | Command Center UI | Deploys an interactive web interface with zero frontend bloat, styled with custom dark-mode aerospace HUD CSS. |
| **ReportLab** | PDF Document Generation | Programmatically constructs the strictly 2-page operational briefing PDF with exact point-level typography and table layouts. |
| **python-pptx** | Presentation Deck Generation | Automatically generates 5-slide briefing decks for pollution control board stakeholders. |
| **Pandas & NumPy** | Numerical & Tabular Data | Handles tile arrays, geographic coordinate vectors, and multi-criteria scoring calculations. |
| **Pytest** | Continuous Quality Assurance | Automates 24 unit tests across geometry, schemas, submission constraints, and edge runtimes. |

---

## 5. File-by-File Code Structure & Function Catalog

### A. Root Orchestration & App
#### 1. `run_pipeline.py`
- **Purpose**: Single-command CLI master controller that executes the entire pipeline end-to-end or in modular phases.
- **Key Functions**:
  - `run_command_stage(cmd, stage_name)`: Executes a subprocess command, captures stdout/stderr, and logs timing.
  - `main()`: Parses CLI arguments (`--mode smoke`, `--mode audit`, `--mode prepare`, `--mode train`, `--mode eval`, `--mode predict`, `--mode report`, `--mode all`) and triggers the corresponding sequence.

#### 2. `app.py`
- **Purpose**: Futuristic dark aerospace tactical command platform built in Streamlit.
- **Key Functions**:
  - `load_yolo_model(weights_path)`: Caches and initializes the fine-tuned YOLO-OBB model in memory.
  - `compute_obb_metrics(corners_128)`: Derives physical area ($m^2$), orientation angle ($0-180^\circ$), width, and length from 8 normalized bounding box coordinates.
  - Workspaces: Implements 6 interactive views: (1) Neural Target Recon, (2) Satellite Video & Edge Lab, (3) Neural Architecture, (4) District GIS Census, (5) Inspector Farida Top 20 Directive, and (6) Competition Deliverables Hub.

---

### B. `src/data/` (Data Ingestion & Dataset Preparation)
#### 1. `src/data/loader.py`
- **Purpose**: Downloads and loads the `SustainabilityLabIITGN/SentinelKilnDB` dataset from Hugging Face.
- **Key Functions**:
  - `download_sentinel_kilndb(cache_dir)`: Authenticates and fetches raw satellite tiles and annotation files.
  - `load_split(split_name, cache_dir)`: Loads train, validation, or test subsets into memory.
  - `get_tile_by_id(tile_id)`: Retrieves a specific satellite image tile and its ground-truth annotations.

#### 2. `src/data/audit.py`
- **Purpose**: Validates dataset integrity, checks image resolutions, audits bounding box coordinates, and verifies class distributions.
- **Key Functions**:
  - `audit_image_dimensions(image_dir)`: Verifies all tiles are strictly $128 \times 128 \times 3$ pixels.
  - `audit_class_distribution(annotation_df)`: Identifies class imbalance (CFCBK 3%, FCBK 60%, Zigzag 37%).
  - `audit_coordinate_validity(annotation_df)`: Flags degenerate boxes, inverted coordinates, or out-of-bound coordinates.
  - `generate_audit_report(output_json_path)`: Exports comprehensive validation metrics to a structured JSON file.

#### 3. `src/data/yolo_dataset.py`
- **Purpose**: Converts raw annotations into the official Ultralytics YOLO-OBB text format.
- **Key Functions**:
  - `convert_polygon_to_yolo_obb(coords)`: Normalizes 4-corner pixel coordinates into normalized space $[0.0, 1.0]$.
  - `export_yolo_dataset(output_dir, sample_size)`: Writes image files and corresponding label `.txt` files into `train/` and `val/` directories.
  - `generate_contact_sheet(images, labels, output_path)`: Compiles a visual grid of labeled sample kilns for manual quality verification.

---

### C. `src/models/` (Deep Learning & Inference)
#### 1. `src/models/train.py`
- **Purpose**: Configures and runs Ultralytics YOLOv8-OBB fine-tuning on Sentinel-2 tiles.
- **Key Functions**:
  - `configure_training(config_yaml)`: Sets hyperparameters: AdamW optimizer, initial learning rate 0.005, cosine decay, and loss weights.
  - `train_yolo_obb(data_yaml, epochs, batch_size, device)`: Executes mixed-precision (FP16) model training and saves the best checkpoint (`weights/best.pt`).

#### 2. `src/models/predict.py`
- **Purpose**: Batched inference engine that processes unlabelled test tiles and produces the official competition submission CSV.
- **Key Functions**:
  - `run_batched_inference(model, image_paths, conf_thresh, iou_thresh)`: Performs high-throughput forward passes with Rotated NMS.
  - `format_submission_row(tile_id, detection)`: Formats coordinates as `tile_id, class, x1, y1, x2, y2, x3, y3, x4, y4, score`.
  - `generate_submission_csv(test_dir, output_csv_path)`: Scans all test images and outputs the final verified CSV file.

---

### D. `src/evaluation/` (Geometry & Benchmark Metrics)
#### 1. `src/evaluation/rotated_iou.py`
- **Purpose**: Computes intersection-over-union between rotated oriented bounding boxes.
- **Key Functions**:
  - `polygon_from_8coords(coords)`: Creates a valid Shapely `Polygon` from 8 flat coordinates; applies `.buffer(0)` to fix self-intersections.
  - `calculate_rotated_iou(poly1, poly2)`: Computes $\frac{\text{Area}(\text{Intersection})}{\text{Area}(\text{Union})}$ between two oriented polygons.

#### 2. `src/evaluation/dota_evaluator.py`
- **Purpose**: Evaluates model performance using the official DOTA benchmark rotated $m\text{AP}@0.5$ protocol.
- **Key Functions**:
  - `compute_ap_all_points(recalls, precisions)`: Calculates Average Precision via all-point PR curve interpolation (VOC 2012 / DOTA standard).
  - `evaluate_predictions(ground_truths, detections)`: Matches detections to ground truth at $\text{IoU} \ge 0.5$, penalizes duplicate detections, and computes per-class AP and mean AP ($m\text{AP}@0.5$).

---

### E. `src/geospatial/` (Spatial Deduplication & Field Prioritization)
#### 1. `src/geospatial/transform.py`
- **Purpose**: Converts satellite image tile pixel coordinates into Earth-surface geographic coordinates.
- **Key Functions**:
  - `tile_pixel_to_wgs84(tile_origin_lat, tile_origin_lon, px, py, gsd=10.0)`: Projects pixel offsets to latitude/longitude using UTM/WGS 84 spherical geodesy.
  - `export_to_geojson(detections, output_geojson_path)`: Creates an RFC 7946 compliant GeoJSON FeatureCollection with polygon geometries and kiln properties.

#### 2. `src/geospatial/deduplication.py`
- **Purpose**: Resolves duplicate detections across 30-pixel overlapping satellite tiles.
- **Key Functions**:
  - `calculate_haversine_distance(lat1, lon1, lat2, lon2)`: Computes surface distance in meters between two coordinates.
  - `deduplicate_district_detections(detections_df, distance_threshold_m=50.0, iou_threshold=0.30)`: Merges detections that represent the same physical kiln into a single physical entity, avoiding double counting on district maps.

#### 3. `src/geospatial/analysis.py`
- **Purpose**: Quantifies statistical uncertainty bounds and maps high-density inspection clusters.
- **Key Functions**:
  - `calculate_uncertainty_intervals(count, precision, recall)`: Generates lower and upper confidence bounds on actual physical kiln counts based on empirical test set precision and recall.
  - `find_nearby_kilns(kilns_df, max_distance_m=1000.0)`: Identifies pairs and groups of kilns located within 1 km of each other for route optimization.

#### 4. `src/geospatial/priority.py`
- **Purpose**: Ranks inspection candidates for field deployment using multi-criteria operational scoring.
- **Key Functions**:
  - `score_inspection_candidate(kiln_type, cluster_density, historical_status, confidence)`: Evaluates targets using:
    $$\text{Score} = 0.35 \times \text{Type} + 0.30 \times \text{Density} + 0.20 \times \text{History} + 0.15 \times \text{Confidence}$$
  - `generate_top20_priority_list(kilns_df, output_csv_path)`: Produces the top 20 candidate matrix with tactical rationale notes for Inspector Farida Khan.

---

### F. `src/reporting/` (Deliverable Generation)
#### 1. `src/reporting/map_generator.py`
- **Purpose**: Renders cartographic overview maps of the district.
- **Key Functions**:
  - `render_district_map(kilns_df, nearby_pairs, output_png_path)`: Plots all physical kilns by type, connects 1 km proximity clusters with visual route lines, and renders geographic scale bars and north arrows.

#### 2. `src/reporting/pdf_brief.py`
- **Purpose**: Compiles the official, strictly 2-page operational briefing PDF for field inspectors using ReportLab.
- **Key Functions**:
  - `generate_inspectors_brief_pdf(...)`: Assembles Page 1 (executive context, embedded district map, distinct counts with confidence intervals, historical conversion analysis) and Page 2 (1 km routing efficiency table, Top 20 target matrix, and legal disclaimers).

#### 3. `src/reporting/presentation.py`
- **Purpose**: Programmatically builds the executive presentation deck and speaker notes.
- **Key Functions**:
  - `create_briefing_deck(metrics_dict, output_pptx_path)`: Generates a 5-slide PowerPoint deck covering mission background, architecture, validation results, district census, and field deployment directives.
  - `create_speaker_notes(output_md_path)`: Writes a timed 5:00 minute pitch script (1:00 min per slide).

---

### G. `deploy/raspberry_pi/` (Edge AI Satellite Inference)
#### 1. `deploy/raspberry_pi/raspberry_pi_satellite_edge.py`
- **Purpose**: Autonomous onboard edge engine for CubeSats or aerial drones equipped with a Raspberry Pi 4 or 5.
- **Key Functions**:
  - `compute_polygon_iou(poly1, poly2)`: Geometry helper for edge NMS.
  - `apply_frame_nms(detections, iou_thresh)`: Eliminates duplicate detections across overlapping frames in the pushbroom swath.
  - `EarthSpaceTracker`: Tracks stationary kilns on Earth as the satellite moves overhead at 7.58 km/s, preventing alert flooding.
  - `run_edge_inference(...)`: Ingests video frames, tiles the swath into $128 \times 128$ chunks, runs fused ONNX inference, and writes downlink telemetry packets (`downlink_telemetry.json`).

---

## 6. How the Mathematical Logic Works

### 1. Oriented Bounding Box Representation
Unlike standard axis-aligned bounding boxes defined by $(x, y, w, h)$, an oriented box is represented by four ordered 2D vertices:
$$\text{Box} = \{(x_1, y_1), (x_2, y_2), (x_3, y_3), (x_4, y_4)\}, \quad x_i, y_i \in [0.0, 1.0]$$
This allows the detector to tightly enclose elongated kilns oriented at arbitrary angles ($\theta \in [0^\circ, 180^\circ)$), eliminating up to 70% of surrounding soil background noise.

### 2. Multi-Task Training Loss
The loss function balances geometry, classification, and boundary precision:
$$\mathcal{L}_{\text{total}} = 7.5 \cdot \mathcal{L}_{\text{Rotated-CIoU}} + 0.5 \cdot \mathcal{L}_{\text{BCE}} + 1.5 \cdot \mathcal{L}_{\text{DFL}}$$
- **Rotated-CIoU**: Penalizes spatial distance, overlap discrepancy, and aspect ratio error between rotated polygons.
- **BCE**: Multi-class Binary Cross-Entropy with class balancing to prevent rare CFCBK starvation.
- **DFL (Distribution Focal Loss)**: Optimizes sub-pixel corner coordinate regression around small $12 \times 6$ pixel targets.

### 3. Spatial Deduplication Formula
Overlapping tiles have an overlap width of 30 pixels ($\approx 300\text{ meters}$). Two candidate detections $A$ and $B$ from adjacent tiles are merged if:
$$D_{\text{geodesic}}(\text{Centroid}_A, \text{Centroid}_B) \le 50.0\text{ m} \quad \text{AND} \quad \text{IoU}(A, B) \ge 0.30$$

### 4. Field Inspection Prioritization Function
Inspector Farida Khan's Top 20 sites are computed via a weighted multi-criteria function:
$$\text{Score}_i = w_t \cdot T_i + w_d \cdot D_i + w_h \cdot H_i + w_c \cdot C_i$$
- $T_i$ (Regulatory Scrutiny): $\text{FCBK} = 1.0$, $\text{CFCBK} = 0.9$, $\text{Zigzag} = 0.2$ ($w_t = 0.35$).
- $D_i$ (Cluster Density): Proximity count within 1 km normalized to $[0, 1]$ ($w_d = 0.30$).
- $H_i$ (Historical Inconsistency): Persisting unconverted FCBKs receive highest priority ($w_h = 0.20$).
- $C_i$ (Model Confidence): Normalized prediction confidence score ($w_c = 0.15$).

---

## 7. Execution Quick Reference

```bash
# Rapid end-to-end smoke test (validates all modules & generates deliverables)
python run_pipeline.py --mode smoke

# Run all 24 automated unit tests
pytest tests/ -v

# Launch the Tactical Command Center Web Platform
streamlit run app.py

# Run standalone satellite edge simulation
python deploy/raspberry_pi/raspberry_pi_satellite_edge.py --source outputs/maps/orbital_pass_demo.mp4
```

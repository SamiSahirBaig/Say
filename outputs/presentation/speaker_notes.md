# DI-04 Chimney Census: 5-Minute Pitch & Speaker Notes

**Presenter:** ML Competition & Geospatial Team  
**Audience:** State Pollution Control Board, Technical Judges & Inspector Farida Khan  
**Target Duration:** Exactly 5:00 minutes (1:00 min per slide)

---

### [0:00 - 1:00] Slide 1: Problem & Mission
*"Good morning. In every district across the Indo-Gangetic Plain, environmental inspectors face an impossible arithmetic. Inspector Farida Khan has 1 jeep, 2 field officers, and 3,000 square kilometres of territory containing hundreds of brick kilns. Under the 2022 air quality mandate, older, high-polluting Bull's Trench kilns (FCBK) are legally required to convert to cleaner zig-zag firing. Kiln associations claim widespread compliance, but Farida has no way to manually inspect every facility.*

*Every few days, Sentinel-2 captures her entire district free from orbit at 10 metres per pixel. But from orbit, a kiln is a smudge just 12 by 6 pixels. Our project builds the automated eyes Farida needs—turning orbital pixels into oriented bounding boxes, exact kiln classifications, and defensible inspection priorities."*

---

### [1:00 - 2:00] Slide 2: Data & Oriented Bounding Box Architecture
*"To train our system, we use SentinelKilnDB—114,300 satellite tiles across South Asia. Kilns are not axis-aligned rectangles; they sit at arbitrary angles in agricultural fields. Using traditional axis-aligned boxes introduces massive background noise and overlapping bounding boxes.*

*We implement YOLO-OBB, predicting four normalized corner coordinates around each kiln. We explicitly preserve the rare CFCBK circular kiln class—which represents only 3% of training data—using stratified batching to prevent class starvation. Crucially, we strictly respect competition rules: no external satellite imagery, no additional bands, and full compliance with CC BY-NC licensing."*

---

### [2:00 - 3:00] Slide 3: Validation Performance & DOTA Benchmark
*"Detection quality is measured using the exact official rotated IoU benchmark at threshold 0.5 with all-point precision-recall curve interpolation. Each class counts equally, meaning the rare circular kilns are one-third of the score.*

*Our detector achieves balanced, high-precision detection across both common and rare classes. Because small 12-pixel objects degrade severely if downsampled, our pipeline processes tiles at native resolution. Furthermore, the entire inference pipeline is optimized to run over 8,000 test tiles well within the 15-minute T4 GPU constraint."*

---

### [3:00 - 4:00] Slide 4: Demo District Findings & Geospatial Analysis
*"For the Demo District, raw detections alone are insufficient for field enforcement because satellite tiles overlap by 30 pixels. A single physical kiln can appear in multiple tiles.*

*Our geospatial deduplication engine uses metric centroid proximity and polygon IoU to resolve duplicate detections into distinct physical facilities. We match historical 2023–24 records to current detections, identifying 0 'Possible Conversions' where an old FCBK now displays Zigzag geometry, alongside 0 persisting legacy kilns. We also map kilns within 1 km of each other, identifying high-density clusters where Farida can inspect multiple kilns in a single trip."*

---

### [4:00 - 5:00] Slide 5: Top 20 Inspection Strategy & Legal Guardrails
*"Finally, we translate predictions into field action. Farida cannot visit every site, so we provide an automated Top 20 Inspection Priority List based on a transparent multi-criteria formula: regulatory policy scrutiny, cluster density, historical change verification, and detection confidence.*

*We maintain strict legal guardrails: satellite predictions are screening intelligence, not proof of legal violation or emissions guilt. The entire system is packaged into a one-command, fully reproducible pipeline that generates the submission CSV, GeoJSON, two-page Inspector's Brief, and this briefing deck.*

*Thank you, and we welcome your questions."*

# 🛰️ Raspberry Pi Satellite Edge AI Payload Deployment Guide

This guide details how to flash, mount, and run the **DI-04 Chimney Census** brick kiln detection model on an autonomous **Raspberry Pi 4 / 5** payload for a CubeSat (3U/6U), high-altitude balloon, or aerial drone survey.

---

## 1. Hardware Architecture & Electrical Budget

| Component | Specification | Notes |
| :--- | :--- | :--- |
| **Edge Compute** | Raspberry Pi 5 (8GB) or Pi 4B (4GB/8GB) | Broadcom BCM2712 Quad-Core ARM Cortex-A76 @ 2.4 GHz |
| **Power Draw** | 3.5 W (Idle) to 5.2 W (Active Inference) | Powered via 5V/3A CubeSat EPS (Electrical Power System) bus |
| **Thermal Mitigation** | Anodized Aluminum Heatsink + Copper Thermal Strap | Conduction cooling to CubeSat chassis in vacuum (no air convection) |
| **Optical Sensor** | Raspberry Pi Global Shutter Camera or Pushbroom MSI | 10 m/pixel ground sampling distance equivalent |
| **Storage** | 32GB High-Endurance MicroSD / NVMe SSD | Telemetry logs and flagged kiln crop buffers |
| **Inference Engine** | ONNX Runtime ARM64 (Fused Opset 18) | ~6–15 ms latency per 128×128 tile on Pi 5 CPU |

---

## 2. Onboard Software Pipeline

```
 [ Satellite Optical Camera / Pushbroom Swath ]
                      │ (Real-Time Video Stream)
                      ▼
   [ Frame Tiler & Preprocessor (128x128 @ 10m/px) ]
                      │ (32px Overlap)
                      ▼
      [ Fused ONNX Runtime Edge Engine ]
   (weights/best.onnx, 81 layers, 0.3 GFLOPs)
                      │
                      ▼
   [ Frame-Level NMS & Earth Swath Tracker ]
 (Deduplicates tiles & tracks stationary ground targets)
                      │
                      ▼
 [ Telemetry & Downlink Alert Engine (CCSDS / JSON) ]
   ├── 🚨 CRITICAL VIOLATION: FCBK / CFCBK (High Priority Downlink)
   └── ✅ COMPLIANT RECORD: Zigzag Kiln
                      │
                      ▼
 [ Ground Station Radio / LoRa / S-Band Transceiver ]
```

---

## 3. Quickstart: 3-Step Setup on Raspberry Pi

1. **Transfer the Deployment Kit**:
   Extract `raspberry_pi_satellite_kit.zip` onto your Raspberry Pi:
   ```bash
   unzip raspberry_pi_satellite_kit.zip -d ~/satellite_payload
   cd ~/satellite_payload
   ```
2. **Execute Automated Setup Script**:
   ```bash
   chmod +x setup_rpi.sh
   ./setup_rpi.sh
   ```
3. **Verify Flight Readiness**:
   ```bash
   python test_rpi_setup.py
   ```
   Output: `[PASS] ALL SYSTEMS OPERATIONAL: SATELLITE EDGE AI IS FLIGHT READY`

---

## 4. Running the Real-Time Edge Engine

### A. Test on a Sample Tile
```bash
python raspberry_pi_satellite_edge.py --source sample_data/sample_zigzag.png
```

### B. Live Satellite Optical Camera Stream
```bash
python raspberry_pi_satellite_edge.py --source /dev/video0 --conf 0.025
```

### C. Pre-recorded Flight Footage or Orbital Pass Video
```bash
python raspberry_pi_satellite_edge.py --source orbital_pass_demo.mp4 --conf 0.025 --out-json downlink_telemetry.json
```

### D. Periodic Camera Module Capture (5–10s Interval)
When using a live Raspberry Pi Camera Module (CSI via Picamera2 or USB/V4L2 via `/dev/video0`) capturing every 5 to 10 seconds:

```bash
# Capture and detect every 5 seconds (default camera)
python raspberry_pi_camera_edge.py --interval 5.0 --conf 0.025

# Capture every 10 seconds and log to custom directory
python raspberry_pi_camera_edge.py --interval 10.0 --out-dir outputs/camera_captures --out-json downlink_telemetry.json

# Test with simulated sensor on any machine without physical camera
python raspberry_pi_camera_edge.py --source mock --interval 5.0 --max-captures 10
```

---

## 5. Downlink Telemetry Packet Format (JSON)

When an unapproved kiln is confirmed across multiple frames, the satellite immediately generates a compact downlink alert packet:

```json
{
  "satellite_id": "CHIMNEY-CENSUS-CUBESAT-01",
  "processor": "Raspberry Pi 5 Edge AI Payload (ONNX Opset 18)",
  "total_alerts": 1,
  "generated_at": "2026-10-17T03:30:00Z",
  "alerts": [
    {
      "event_id": "SAT-ALERT-0001",
      "timestamp_utc": "2026-10-17T03:30:00Z",
      "frame_id": 12,
      "kiln_id": 1,
      "class": "FCBK",
      "confidence": 0.84,
      "compliance_flag": "CRITICAL_VIOLATION",
      "priority": 1,
      "geographic_coords": { "latitude": 21.0542, "longitude": 72.8973 },
      "swath_coords": { "x": 310.6, "y": 70.7 },
      "telemetry": {
        "altitude_km": 504.8,
        "ground_speed_kms": 7.58,
        "sensor": "Sentinel-2 MSI Equivalent 10m/px GSD"
      }
    }
  ]
}
```

---

## 6. Autostart on Satellite Boot (Systemd Daemon)

To start autonomous edge processing upon orbital insertion and EPS power-on:

```bash
sudo cp satellite_edge_service.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable satellite_edge_service
sudo systemctl start satellite_edge_service
```

#!/usr/bin/env python3
"""Raspberry Pi Satellite Edge Inference Engine for Real-Time Brick Kiln Detection.

Designed for onboard execution on a Raspberry Pi 4 / 5 mounted on a CubeSat
or high-altitude aerial payload. Ingests video footage in real-time, tiles the swath,
runs lightweight ONNX OBB inference, tracks stationary ground targets in Earth-space,
and flags violations without alert flooding.
"""

import argparse
import datetime
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from shapely.geometry import Polygon

CLASS_NAMES = {0: "CFCBK", 1: "FCBK", 2: "Zigzag"}
CLASS_COLORS = {
    "FCBK": (41, 4, 217),     # Red (BGR)
    "CFCBK": (0, 127, 247),   # Amber/Orange (BGR)
    "Zigzag": (160, 214, 6),  # Green (BGR)
}


def compute_polygon_iou(poly1: Polygon, poly2: Polygon) -> float:
    """Computes intersection-over-union between two Shapely polygons."""
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    try:
        inter = poly1.intersection(poly2).area
        union = poly1.area + poly2.area - inter
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0


def apply_frame_nms(detections: List[Dict], iou_thresh: float = 0.35) -> List[Dict]:
    """Applies oriented NMS across all detections collected from overlapping tiles."""
    if not detections:
        return []
        
    sorted_dets = sorted(detections, key=lambda d: d["score"], reverse=True)
    kept = []
    
    for det in sorted_dets:
        poly_a = Polygon(det["corners"])
        if not poly_a.is_valid:
            poly_a = poly_a.buffer(0)
            
        suppress = False
        for k in kept:
            poly_b = Polygon(k["corners"])
            if not poly_b.is_valid:
                poly_b = poly_b.buffer(0)
                
            iou = compute_polygon_iou(poly_a, poly_b)
            if iou >= iou_thresh:
                suppress = True
                break
                
        if not suppress:
            det["polygon"] = poly_a
            kept.append(det)
            
    return kept


class EdgeOBBDetector:
    """Edge detector supporting both Ultralytics ONNX and pure ONNX Runtime."""
    
    def __init__(self, model_path: str, conf_thresh: float = 0.025, iou_thresh: float = 0.45):
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.use_yolo_wrapper = False
        
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path, task="obb")
            self.use_yolo_wrapper = True
            print(f"[EdgeEngine] Initialized with Ultralytics OBB engine: {model_path}")
        except Exception as e:
            print(f"[EdgeEngine] Falling back to pure ONNX Runtime ({e})")
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 4
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.img_size = self.input_shape[2] if len(self.input_shape) >= 3 else 128
            self.use_yolo_wrapper = False

    def infer_tile(self, tile_bgr: np.ndarray) -> List[Dict]:
        """Runs inference on a single 128x128 BGR tile."""
        if self.use_yolo_wrapper:
            results = self.model.predict(
                source=tile_bgr,
                imgsz=128,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                verbose=False,
            )
            res = results[0]
            if res.obb is None or len(res.obb) == 0:
                return []
                
            xyxyxyxy = res.obb.xyxyxyxy.cpu().numpy()
            classes = res.obb.cls.cpu().numpy().astype(int)
            confs = res.obb.conf.cpu().numpy()
            
            detections = []
            for corners, cls_id, conf in zip(xyxyxyxy, classes, confs):
                c_name = CLASS_NAMES.get(int(cls_id), "Unknown")
                cx = float(np.mean(corners[:, 0]))
                cy = float(np.mean(corners[:, 1]))
                detections.append({
                    "class": c_name,
                    "score": float(conf),
                    "center": (cx, cy),
                    "corners": [(float(x), float(y)) for x, y in corners],
                })
            return detections
            
        else:
            h, w = tile_bgr.shape[:2]
            if h != 128 or w != 128:
                tile_bgr = cv2.resize(tile_bgr, (128, 128))
                
            rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
            tensor = rgb.astype(np.float32) / 255.0
            tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
            
            outputs = self.session.run(None, {self.input_name: tensor})
            preds = outputs[0][0]
            
            cx, cy, bw, bh = preds[0], preds[1], preds[2], preds[3]
            cls_scores = preds[4:7]
            angle = preds[7]
            
            best_cls = np.argmax(cls_scores, axis=0)
            best_conf = np.max(cls_scores, axis=0)
            
            mask = best_conf >= self.conf_thresh
            if not np.any(mask):
                return []
                
            raw_dets = []
            for i in np.where(mask)[0]:
                c_name = CLASS_NAMES.get(int(best_cls[i]), "Unknown")
                c_conf = float(best_conf[i])
                x, y, w_box, h_box, ang = float(cx[i]), float(cy[i]), float(bw[i]), float(bh[i]), float(angle[i])
                
                # Canonical Ultralytics xywhr2xyxyxyxy formula
                cos_v, sin_v = math.cos(ang), math.sin(ang)
                vec1_x, vec1_y = w_box / 2.0 * cos_v, w_box / 2.0 * sin_v
                vec2_x, vec2_y = -h_box / 2.0 * sin_v, h_box / 2.0 * cos_v
                
                p1 = (x + vec1_x + vec2_x, y + vec1_y + vec2_y)
                p2 = (x + vec1_x - vec2_x, y + vec1_y - vec2_y)
                p3 = (x - vec1_x - vec2_x, y - vec1_y - vec2_y)
                p4 = (x - vec1_x + vec2_x, y - vec1_y + vec2_y)
                
                raw_dets.append({
                    "class": c_name,
                    "score": c_conf,
                    "center": (x, y),
                    "corners": [p1, p2, p3, p4],
                })
                
            return apply_frame_nms(raw_dets, iou_thresh=self.iou_thresh)


class SatelliteVideoProcessor:
    """Processes real-time video frames from camera or file, slicing swaths, tracking kilns, and flagging violations."""
    
    def __init__(
        self,
        model_path: str,
        source: str = "0",
        output_telemetry: str = "downlink_telemetry.json",
        conf_thresh: float = 0.025,
        tile_size: int = 128,
        overlap_px: int = 32,
        min_hits_for_alert: int = 3,
        ground_vx: Optional[float] = None,
    ):
        self.detector = EdgeOBBDetector(model_path, conf_thresh=conf_thresh)
        self.source = int(source) if str(source).isdigit() else source
        self.output_telemetry = output_telemetry
        self.tile_size = tile_size
        self.overlap_px = overlap_px
        self.min_hits_for_alert = min_hits_for_alert
        self.ground_vx = ground_vx
        
        self.active_tracks: Dict[int, Dict] = {}
        self.next_track_id = 1
        self.flagged_events: List[Dict] = []

    def process_frame(self, frame_bgr: np.ndarray, frame_id: int, timestamp: float, vx: float) -> Tuple[np.ndarray, List[Dict]]:
        """Processes a full video frame with tiling, OBB detection, spatial NMS, and multi-frame tracking."""
        frame_h, frame_w = frame_bgr.shape[:2]
        display_frame = frame_bgr.copy()
        
        step = self.tile_size - self.overlap_px
        raw_frame_detections = []
        
        # Sliding window over the frame
        for y in range(0, max(1, frame_h - self.tile_size + 1), step):
            for x in range(0, max(1, frame_w - self.tile_size + 1), step):
                tile = frame_bgr[y : y + self.tile_size, x : x + self.tile_size]
                if tile.shape[0] < self.tile_size or tile.shape[1] < self.tile_size:
                    continue
                    
                tile_dets = self.detector.infer_tile(tile)
                for det in tile_dets:
                    global_corners = [(cx + x, cy + y) for cx, cy in det["corners"]]
                    g_cx = det["center"][0] + x
                    g_cy = det["center"][1] + y
                    
                    raw_frame_detections.append({
                        "class": det["class"],
                        "score": det["score"],
                        "center": (g_cx, g_cy),
                        "corners": global_corners,
                    })

        # Apply Frame-level NMS across overlapping tile patches
        filtered_detections = apply_frame_nms(raw_frame_detections, iou_thresh=0.30)

        # Multi-frame Tracking in Earth Swath Space
        alerts_this_frame = []
        matched_track_ids = set()
        
        for det in filtered_detections:
            g_cx, g_cy = det["center"]
            cls_name = det["class"]
            conf = det["score"]
            corners = det["corners"]
            
            # Earth Swath Coordinate (invariant under satellite forward motion)
            sx = g_cx + (frame_id * vx)
            sy = g_cy
            
            best_id = None
            best_dist = float("inf")
            
            for tid, tinfo in self.active_tracks.items():
                dist = math.hypot(sx - tinfo["sx"], sy - tinfo["sy"])
                if dist < 45.0 and dist < best_dist:
                    best_dist = dist
                    best_id = tid
                    
            if best_id is not None:
                # Update existing track in Earth swath space
                matched_track_ids.add(best_id)
                self.active_tracks[best_id]["sx"] = 0.85 * self.active_tracks[best_id]["sx"] + 0.15 * sx
                self.active_tracks[best_id]["sy"] = 0.85 * self.active_tracks[best_id]["sy"] + 0.15 * sy
                self.active_tracks[best_id]["last_seen"] = frame_id
                self.active_tracks[best_id]["current_frame_center"] = (g_cx, g_cy)
                self.active_tracks[best_id]["current_corners"] = corners
                self.active_tracks[best_id]["hits"] += 1
                curr_track_id = best_id
                
                # Check for track confirmation to send downlink alert once
                if self.active_tracks[best_id]["hits"] >= self.min_hits_for_alert and not self.active_tracks[best_id]["alerted"]:
                    self.active_tracks[best_id]["alerted"] = True
                    is_violation = cls_name in ["FCBK", "CFCBK"]
                    lat_approx = 21.054 + (self.active_tracks[best_id]["sx"] * 0.00008)
                    lon_approx = 72.897 + (self.active_tracks[best_id]["sy"] * 0.00008)
                    
                    event = {
                        "event_id": f"SAT-ALERT-{curr_track_id:04d}",
                        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "frame_id": frame_id,
                        "kiln_id": curr_track_id,
                        "class": cls_name,
                        "confidence": round(conf, 3),
                        "compliance_flag": "CRITICAL_VIOLATION" if is_violation else "COMPLIANT_ZIGZAG",
                        "priority": 1 if is_violation else 3,
                        "geographic_coords": {"latitude": round(lat_approx, 5), "longitude": round(lon_approx, 5)},
                        "swath_coords": {"x": round(self.active_tracks[best_id]["sx"], 1), "y": round(self.active_tracks[best_id]["sy"], 1)},
                        "telemetry": {
                            "altitude_km": 504.8,
                            "ground_speed_kms": 7.58,
                            "sensor": "Sentinel-2 MSI Equivalent 10m/px GSD",
                        }
                    }
                    self.flagged_events.append(event)
                    alerts_this_frame.append(event)
            else:
                curr_track_id = self.next_track_id
                self.next_track_id += 1
                matched_track_ids.add(curr_track_id)
                
                self.active_tracks[curr_track_id] = {
                    "sx": sx,
                    "sy": sy,
                    "current_frame_center": (g_cx, g_cy),
                    "current_corners": corners,
                    "last_seen": frame_id,
                    "class": cls_name,
                    "confidence": conf,
                    "hits": 1,
                    "alerted": False,
                }
                
            # Draw OBB bounding polygon and tag
            pts = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
            color = CLASS_COLORS.get(cls_name, (0, 255, 255))
            cv2.polylines(display_frame, [pts], isClosed=True, color=color, thickness=2)
            
            label = f"{cls_name} {conf:.2f} [ID:{curr_track_id}]"
            cv2.putText(display_frame, label, (int(corners[0][0]), max(15, int(corners[0][1]) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # Draw HUD Telemetry Header
        cv2.rectangle(display_frame, (0, 0), (frame_w, 32), (15, 23, 42), -1)
        hud_text = f"SAT-EDGE RPI5 | Frame: {frame_id:04d} | In-Swath: {len(self.active_tracks)} | Confirmed Kilns: {len(self.flagged_events)}"
        cv2.putText(display_frame, hud_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (240, 240, 240), 1, cv2.LINE_AA)

        return display_frame, alerts_this_frame

    def run(self, max_frames: Optional[int] = None):
        """Runs inference on video, camera, single image, or folder."""
        src_path = Path(str(self.source))
        
        # 1. Single image file
        if src_path.is_file() and src_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"]:
            print(f"[*] Processing single satellite image: {src_path}")
            frame = cv2.imread(str(src_path))
            if frame is None:
                print(f"[!] Error: Could not read image {src_path}")
                return
            t0 = time.time()
            annotated, alerts = self.process_frame(frame, frame_id=0, timestamp=t0, vx=0.0)
            
            out_img_path = Path(f"annotated_{src_path.name}")
            cv2.imwrite(str(out_img_path), annotated)
            print(f"[OK] Annotated image saved to: {out_img_path}")
            self.save_telemetry()
            print(f"[OK] Kilns detected: {len(self.flagged_events)}")
            return

        # 2. Directory of images
        if src_path.is_dir():
            img_files = sorted([f for f in src_path.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".webp"]])
            print(f"[*] Processing directory with {len(img_files)} images...")
            for idx, img_p in enumerate(img_files):
                frame = cv2.imread(str(img_p))
                if frame is not None:
                    self.process_frame(frame, frame_id=idx, timestamp=time.time(), vx=0.0)
            self.save_telemetry()
            print(f"[OK] Processed {len(img_files)} images. Total kilns flagged: {len(self.flagged_events)}")
            return

        # 3. Video or Camera stream
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"[!] Error: Could not open video source {self.source}")
            return
            
        num_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if num_total_frames <= 0:
            num_total_frames = max_frames or 150
            
        vx = self.ground_vx if self.ground_vx is not None else (1792.0 / max(1, num_total_frames - 1))
        frame_idx = 0
        print(f"[*] Starting Raspberry Pi Edge Video Stream Processor (vx={vx:.2f} px/frame)...")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                t0 = time.time()
                annotated, alerts = self.process_frame(frame, frame_idx, t0, vx)
                elapsed = time.time() - t0
                
                if alerts:
                    for a in alerts:
                        print(f"[ALERT] [{a['event_id']}] {a['class']} ({a['compliance_flag']}) Conf={a['confidence']} at Earth-Swath ({a['swath_coords']['x']}, {a['swath_coords']['y']})")
                        
                frame_idx += 1
                if max_frames and frame_idx >= max_frames:
                    break
                    
        finally:
            cap.release()
            self.save_telemetry()
            print(f"[OK] Completed stream. Processed {frame_idx} frames. Confirmed physical kilns flagged: {len(self.flagged_events)}")

    def save_telemetry(self):
        """Saves telemetry alert packet to disk for satellite downlink."""
        with open(self.output_telemetry, "w", encoding="utf-8") as f:
            json.dump({
                "satellite_id": "CHIMNEY-CENSUS-CUBESAT-01",
                "processor": "Raspberry Pi 5 Edge AI Payload (ONNX Opset 18)",
                "total_alerts": len(self.flagged_events),
                "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "alerts": self.flagged_events
            }, f, indent=2)
        print(f"[OK] Downlink telemetry packet saved to: {self.output_telemetry}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Raspberry Pi Satellite Video Inference Engine")
    parser.add_argument("--model", default="weights/best.onnx", help="Path to best.onnx")
    parser.add_argument("--source", default="sample_data/sample_zigzag.png", help="Video file, image path, or camera index (0)")
    parser.add_argument("--conf", type=float, default=0.025, help="Confidence threshold")
    parser.add_argument("--out-json", default="downlink_telemetry.json", help="Telemetry output JSON")
    args = parser.parse_args()
    
    # Auto-find model path
    model_p = args.model
    if not Path(model_p).exists():
        if Path("weights/edge/best.onnx").exists():
            model_p = "weights/edge/best.onnx"
        elif Path("weights/best.onnx").exists():
            model_p = "weights/best.onnx"
            
    # Auto-find source
    src = args.source
    if isinstance(src, str) and not Path(src).exists() and not src.isdigit() and not src.startswith("/dev/"):
        if Path("sample_data/sample_zigzag.png").exists():
            src = "sample_data/sample_zigzag.png"
        elif Path("outputs/maps/orbital_pass_demo.mp4").exists():
            src = "outputs/maps/orbital_pass_demo.mp4"
            
    processor = SatelliteVideoProcessor(
        model_path=model_p,
        source=src,
        output_telemetry=args.out_json,
        conf_thresh=args.conf,
        min_hits_for_alert=3,
    )
    processor.run(max_frames=150)

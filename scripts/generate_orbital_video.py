#!/usr/bin/env python3
"""Generate a realistic simulated satellite orbital pass video over the Brick Kiln Belt."""

from pathlib import Path
import cv2
import numpy as np

def generate_orbital_video(
    output_path: str = "outputs/maps/orbital_pass_demo.mp4",
    num_frames: int = 150,
    fps: int = 15,
):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Gather sample tiles containing kilns
    tile_dir = Path("data/yolo_dataset/images/val")
    sample_files = [
        "21.0547_72.8972.png",  # FCBK pair
        "21.1173_92.1820.png",  # Zigzag cluster
        "22.5117_86.8431.png",  # CFCBK circular
        "21.0047_73.1384.png",  # FCBK single
        "21.1085_92.1820.png",  # Zigzag pair
        "20.9459_72.9384.png",  # Background field
    ]
    
    tiles = []
    for fn in sample_files:
        p = tile_dir / fn
        if p.exists():
            img = cv2.imread(str(p))
            if img is not None:
                tiles.append(cv2.resize(img, (256, 256)))
                
    if not tiles:
        for i in range(5):
            t = np.full((256, 256, 3), 70 + i * 20, dtype=np.uint8)
            tiles.append(t)
            
    # Stitch into a large linear swath mosaic
    mosaic = np.hstack(tiles)
    total_w = mosaic.shape[1]
    
    out_w, out_h = 512, 384
    
    # Scale mosaic height to out_h
    scale = out_h / mosaic.shape[0]
    swath_scaled = cv2.resize(mosaic, (int(total_w * scale), out_h))
    swath_w = swath_scaled.shape[1]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
    
    max_scroll = swath_w - out_w
    if max_scroll <= 0:
        max_scroll = 100
        swath_scaled = cv2.copyMakeBorder(swath_scaled, 0, 0, 0, out_w, cv2.BORDER_REFLECT)
        
    print(f"[*] Generating {num_frames} frames of clean raw satellite sensor video to {output_path}...")
    
    for i in range(num_frames):
        # Linear orbital ground track motion
        offset_x = int((i / (num_frames - 1)) * max_scroll)
        frame = swath_scaled[:, offset_x : offset_x + out_w].copy()
        
        # Subtle sensor photon noise (realistic satellite sensor response)
        noise = np.random.normal(0, 1.5, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        out.write(frame)
        
    out.release()
    print(f"[OK] Successfully generated clean raw orbital video: {output_path}")

if __name__ == "__main__":
    generate_orbital_video()

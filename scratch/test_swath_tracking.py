import sys
sys.path.insert(0, '.')
import cv2, math, json
import numpy as np
from deploy.raspberry_pi.raspberry_pi_satellite_edge import EdgeOBBDetector, apply_frame_nms

detector = EdgeOBBDetector('weights/edge/best.onnx', conf_thresh=0.03)
cap = cv2.VideoCapture('outputs/maps/orbital_pass_demo.mp4')
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
vx = 1792.0 / (num_frames - 1)

tracks = {}
next_id = 1
frame_idx = 0

step = 96
while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    raw_dets = []
    for y in range(0, h - 128 + 1, step):
        for x in range(0, w - 128 + 1, step):
            tile = frame[y:y+128, x:x+128]
            dets = detector.infer_tile(tile)
            for d in dets:
                gx = d['center'][0] + x
                gy = d['center'][1] + y
                corners = [(cx + x, cy + y) for cx, cy in d['corners']]
                raw_dets.append({'center': (gx, gy), 'corners': corners, 'score': d['score'], 'class': d['class']})
    
    filtered = apply_frame_nms(raw_dets, iou_thresh=0.30)
    for det in filtered:
        gx, gy = det['center']
        sx = gx + frame_idx * vx
        sy = gy
        
        matched = None
        for tid, tinfo in tracks.items():
            dist = math.hypot(sx - tinfo['sx'], sy - tinfo['sy'])
            if dist < 40.0:
                matched = tid
                break
                
        if matched is not None:
            tracks[matched]['hits'] += 1
            tracks[matched]['last_frame'] = frame_idx
            tracks[matched]['sx'] = 0.8 * tracks[matched]['sx'] + 0.2 * sx
            tracks[matched]['sy'] = 0.8 * tracks[matched]['sy'] + 0.2 * sy
        else:
            tracks[next_id] = {'sx': sx, 'sy': sy, 'class': det['class'], 'hits': 1, 'first_frame': frame_idx, 'last_frame': frame_idx, 'conf': det['score']}
            next_id += 1
            
    frame_idx += 1
cap.release()

confirmed = {k: v for k, v in tracks.items() if v['hits'] >= 3}
print(f"Total candidate tracks: {len(tracks)}")
print(f"Confirmed physical kilns (hits >= 3): {len(confirmed)}")
for k, v in confirmed.items():
    cls = v['class']
    hits = v['hits']
    cf = v['conf']
    sx = v['sx']
    sy = v['sy']
    print(f"Kiln {k:02d}: Class={cls}, Hits={hits}, Conf={cf:.3f}, Earth Swath=(X:{sx:.1f}, Y:{sy:.1f}), Frames {v['first_frame']}->{v['last_frame']}")

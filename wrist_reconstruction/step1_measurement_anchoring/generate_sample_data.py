"""
Generate realistic synthetic 2D AP (Anteroposterior) and Lateral (Side) wrist images
for testing Step 1: Measurement Anchoring.
"""

import cv2
import numpy as np
from pathlib import Path

def create_synthetic_wrist_images(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Image dimensions: 800 x 1200 px
    H, W = 1200, 800
    
    # -------------------------------------------------------------
    # 1. AP View (Front View) - Palm/Wrist facing camera
    # Target wrist width: 480 px in image (representing e.g. 56.0 mm)
    # -------------------------------------------------------------
    ap_img = np.full((H, W, 3), 245, dtype=np.uint8)  # Light neutral background
    
    # Draw Forearm & Hand Contour (AP)
    # Forearm base: y=1100 to y=800, width ~ 520 px
    # Wrist crease: y=750, width ~ 460 px (Radial to Ulnar styloid)
    # Palm: y=500, width ~ 560 px
    # Fingers: y=150 to y=500
    
    pts_ap_left = [
        (140, 1200), (150, 950), (170, 750),  # Forearm to wrist crease
        (130, 600), (120, 480), (180, 200),   # Thumb & Palm
        (260, 100), (380, 80), (500, 90),     # Fingers top
        (640, 220), (660, 480), (640, 600),   # Little finger & palm
        (630, 750), (650, 950), (660, 1200)   # Wrist crease to forearm
    ]
    
    pts_ap = np.array(pts_ap_left, dtype=np.int32)
    # Draw smooth filled silhouette
    cv2.fillPoly(ap_img, [pts_ap], (195, 205, 215)) # Soft anatomical skin tone
    cv2.polylines(ap_img, [pts_ap], True, (130, 140, 150), 3, cv2.LINE_AA)
    
    # Add subtle shading/contours
    cv2.line(ap_img, (170, 750), (630, 750), (160, 170, 180), 2, cv2.LINE_AA) # Distal wrist crease hint
    
    ap_path = output_dir / "sample_wrist_ap.png"
    cv2.imwrite(str(ap_path), ap_img)
    
    # -------------------------------------------------------------
    # 2. Lateral View (Side View) - Thumb/Radius facing camera
    # Target wrist depth: 310 px in image (representing e.g. 38.0 mm)
    # Wrist is naturally narrower in lateral view (depth < width)
    # -------------------------------------------------------------
    lat_img = np.full((H, W, 3), 245, dtype=np.uint8)
    
    pts_lat = np.array([
        (250, 1200), (255, 950), (265, 750),  # Posterior forearm & wrist
        (250, 550), (260, 350), (330, 120),   # Dorsal hand to fingers
        (420, 120), (460, 300), (550, 480),   # Fingers to thenar/thumb
        (575, 750), (560, 950), (550, 1200)   # Palmar wrist crease to forearm
    ], dtype=np.int32)
    
    cv2.fillPoly(lat_img, [pts_lat], (195, 205, 215))
    cv2.polylines(lat_img, [pts_lat], True, (130, 140, 150), 3, cv2.LINE_AA)
    cv2.line(lat_img, (265, 750), (575, 750), (160, 170, 180), 2, cv2.LINE_AA)
    
    lat_path = output_dir / "sample_wrist_lat.png"
    cv2.imwrite(str(lat_path), lat_img)
    
    print(f"[OK] Synthetic AP image created at: {ap_path}")
    print(f"[OK] Synthetic Lateral image created at: {lat_path}")
    return ap_path, lat_path

if __name__ == "__main__":
    out = Path(r"c:\Users\Admin\Healthcare\wrist_reconstruction\data")
    create_synthetic_wrist_images(out)


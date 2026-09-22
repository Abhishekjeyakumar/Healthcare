import cv2
import numpy as np
from reconstruction import extract_silhouette, generate_3d_volume, extract_and_clean_mesh, generate_obj_string

def create_dummy_ap():
    img = np.zeros((200, 200), dtype=np.uint8)
    # Leg from front: thicker at top (thigh), thinner at bottom
    cv2.ellipse(img, (100, 50), (40, 80), 0, 0, 360, 255, -1)
    cv2.ellipse(img, (100, 150), (25, 70), 0, 0, 360, 255, -1)
    return img

def create_dummy_lat():
    img = np.zeros((200, 200), dtype=np.uint8)
    # Leg from side: knee sticks out a bit
    pts = np.array([[100, 0], [120, 100], [100, 200], [60, 200], [70, 100], [60, 0]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(img, [pts], 255)
    return img

def run_test():
    ap = create_dummy_ap()
    lat = create_dummy_lat()
    
    print("Testing pair 1...")
    ap_sil = extract_silhouette(ap, target_size=64)
    lat_sil = extract_silhouette(lat, target_size=64)
    vol1 = generate_3d_volume(ap_sil, lat_sil, size=64, sigma=1.0)
    v1, f1 = extract_and_clean_mesh(vol1, threshold=0.5)
    print(f"Result 1: {len(v1)} vertices, {len(f1)} faces")
    
    print("Testing pair 2 (changing AP image)...")
    ap2 = np.zeros((200, 200), dtype=np.uint8)
    cv2.ellipse(ap2, (100, 100), (20, 90), 0, 0, 360, 255, -1) # much thinner
    ap2_sil = extract_silhouette(ap2, target_size=64)
    vol2 = generate_3d_volume(ap2_sil, lat_sil, size=64, sigma=1.0)
    v2, f2 = extract_and_clean_mesh(vol2, threshold=0.5)
    print(f"Result 2: {len(v2)} vertices, {len(f2)} faces")
    
    if len(v1) != len(v2):
        print("SUCCESS: Mesh changed when input changed.")
    else:
        print("FAILED: Mesh is identical despite input change.")
        
if __name__ == "__main__":
    run_test()

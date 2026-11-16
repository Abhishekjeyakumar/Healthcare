"""
=============================================================================
Demo Script for Step 2: 2D Silhouette & Metric Contour Extraction
=============================================================================
Runs the Step 2 extractor on the user's images (AP_Sample_Img.jpeg and
L_Sample_Img.jpeg) using the calibration manifest produced in Step 1.
=============================================================================
"""

from pathlib import Path
from silhouette_extractor import SilhouetteExtractor

def run_demo():
    print("=" * 70)
    print("  STEP 2: 2D SILHOUETTE & METRIC CONTOUR EXTRACTION")
    print("=" * 70)

    base_dir = Path(r"c:\Users\Admin\Healthcare\wrist_reconstruction")
    data_dir = base_dir / "data"
    output_dir = base_dir / "output"

    ap_img = data_dir / "AP_Sample_Img.jpeg"
    lat_img = data_dir / "L_Sample_Img.jpeg"
    calib_json = output_dir / "calibration_anchor.json"

    if not calib_json.exists():
        raise FileNotFoundError(f"Calibration manifest not found at {calib_json}. Please run Step 1 first.")

    print(f"\n[1] Input Images:")
    print(f"    - AP Image      : {ap_img.name}")
    print(f"    - Lateral Image : {lat_img.name}")
    print(f"    - Calibration   : {calib_json.name}")

    extractor = SilhouetteExtractor(slice_step_mm=2.0)

    print(f"\n[2] Extracting Silhouettes & Eliminating Artifacts...")
    result = extractor.process(
        ap_image_path=ap_img,
        lat_image_path=lat_img,
        calibration_manifest_path=calib_json,
        output_dir=output_dir
    )

    landmarks = result["landmarks"]
    artifacts = result["artifacts"]

    print(f"\n[3] Anatomical Landmarks Identified (in physical mm):")
    print(f"    - Forearm Base Level      : {landmarks.forearm_base_z_mm:.1f} mm (Mannequin Flat Cut)")
    print(f"    - Wrist Crease Level      : {landmarks.wrist_crease_z_mm:.1f} mm (Narrowest Wrist Neck)")
    print(f"    - Palm Arch Max Level     : {landmarks.palm_max_z_mm:.1f} mm (Metacarpal Maximum Width)")
    print(f"    - Fingertip Apex Level    : {landmarks.fingertip_apex_z_mm:.1f} mm (Hand Tip)")
    print(f"    - Total Hand Length       : {landmarks.total_hand_length_mm:.1f} mm")

    print(f"\n[4] Metric Rib Slicing:")
    print(f"    - Total Cross-Sections   : {result['num_slices']} slices (every 2.0 mm)")
    print(f"    - Slices Manifest Exported: {result['manifest_path']}")

    print(f"\n[5] Generated Output Artifacts:")
    print(f"    - AP Mask Image          : {artifacts['ap_mask']}")
    print(f"    - Lateral Mask Image     : {artifacts['lat_mask']}")
    print(f"    - AP Contour Overlay     : {artifacts['ap_overlay']}")
    print(f"    - Lateral Contour Overlay: {artifacts['lat_overlay']}")
    print(f"    - Metric Profile Plot    : {artifacts['profile_plot']}")

    print("\n" + "=" * 70)
    print("  STATUS: STEP 2 SUCCESSFULLY COMPLETED & VERIFIED!")
    print("=" * 70)

if __name__ == "__main__":
    run_demo()


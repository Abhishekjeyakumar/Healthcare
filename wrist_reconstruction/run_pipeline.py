"""
=============================================================================
MASTER PIPELINE RUNNER: 2D TO 3D MANNEQUIN HAND RECONSTRUCTION
=============================================================================
Runs the complete 4-step pipeline end-to-end:
  Step 1: Manual Measurement Anchoring (1mm metric scale lock)
  Step 2: 2D Silhouette & Metric Contour Extraction (All 5 digits)
  Step 3: Mannequin 3D Lofting & Laplacian Fairing
  Step 4: Precision Planar Cut & Production Watertight STL/3MF Export
=============================================================================
"""

import argparse
import sys
import time
from pathlib import Path

# Add project subfolders to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "step1_measurement_anchoring"))
sys.path.append(str(BASE_DIR / "step2_silhouette_extraction"))
sys.path.append(str(BASE_DIR / "step3_3d_reconstruction"))
sys.path.append(str(BASE_DIR / "step4_mannequin_cut_and_export"))

from measurement_anchor import MeasurementAnchorEngine, PhysicalMeasurements
from silhouette_extractor import SilhouetteExtractor
from mannequin_reconstructor import MannequinReconstructor
from mannequin_cutter import MannequinCutterAndExporter


def run_full_pipeline(
    ap_path: Path,
    lat_path: Path,
    wrist_w_mm: float,
    wrist_d_mm: float,
    wrist_c_mm: float,
    output_dir: Path
):
    print("=" * 75)
    print("  HEALTHCARE 3D WRIST RECONSTRUCTION PIPELINE (1mm ACCURACY)")
    print("=" * 75)
    start_time = time.time()

    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # STEP 1: Metric Scale Anchoring
    # -------------------------------------------------------------
    print("\n>>> [STEP 1/4] Anchoring Real-World Physical Measurements...")
    measurements = PhysicalMeasurements(
        wrist_width_mm=wrist_w_mm,
        wrist_depth_mm=wrist_d_mm,
        wrist_circumference_mm=wrist_c_mm
    )

    anchor_engine = MeasurementAnchorEngine()
    step1_res = anchor_engine.process(
        ap_image_path=ap_path,
        lat_image_path=lat_path,
        measurements=measurements,
        output_dir=output_dir
    )
    calib_json = Path(step1_res["manifest_path"])
    print(f"    [OK] AP Scale  : {step1_res['manifest']['calibrations']['AP']['scale_mm_per_pixel']:.4f} mm/px")
    print(f"    [OK] LAT Scale : {step1_res['manifest']['calibrations']['Lateral']['scale_mm_per_pixel']:.4f} mm/px")
    print(f"    [OK] Manifest  : {calib_json.name}")

    # -------------------------------------------------------------
    # STEP 2: 2D Silhouette & Metric Contours
    # -------------------------------------------------------------
    print("\n>>> [STEP 2/4] Extracting 2D Silhouettes & Slicing Metric Ribs...")
    extractor = SilhouetteExtractor(slice_step_mm=2.0)
    step2_res = extractor.process(
        ap_image_path=ap_path,
        lat_image_path=lat_path,
        calibration_manifest_path=calib_json,
        output_dir=output_dir
    )
    slices_json = Path(step2_res["manifest_path"])
    print(f"    [OK] Total Slices  : {step2_res['num_slices']} cross-sections (every 2.0 mm)")
    print(f"    [OK] Hand Length   : {step2_res['landmarks'].total_hand_length_mm:.1f} mm")
    print(f"    [OK] Slices File   : {slices_json.name}")

    # -------------------------------------------------------------
    # STEP 3: 3D Lofting & Watertight STL
    # -------------------------------------------------------------
    print("\n>>> [STEP 3/4] Lofts 3D Mannequin Mesh & Smooths Surface...")
    reconstructor = MannequinReconstructor()
    step3_res = reconstructor.process(
        slices_manifest_path=slices_json,
        calibration_manifest_path=calib_json,
        output_dir=output_dir
    )
    raw_stl = Path(step3_res["stl_path"])
    s3_calip = step3_res["report"]["caliper_verification"]
    print(f"    [OK] Watertight Solid : {step3_res['report']['mesh_statistics']['is_watertight']}")
    print(f"    [OK] Reconstructed W : {s3_calip['mesh_width_mm']} mm (Error: {s3_calip['width_error_mm']} mm)")
    print(f"    [OK] Reconstructed D : {s3_calip['mesh_depth_mm']} mm (Error: {s3_calip['depth_error_mm']} mm)")
    print(f"    [OK] 1mm Status      : {s3_calip['1mm_accuracy_guarantee']}")

    # -------------------------------------------------------------
    # STEP 4: Flush Planar Cut & Slicer Quality Audit
    # -------------------------------------------------------------
    print("\n>>> [STEP 4/4] Applying Precision Planar Cut & Final Quality Audit...")
    cutter = MannequinCutterAndExporter(trim_height_from_base_mm=6.0)
    step4_res = cutter.process(
        raw_stl_path=raw_stl,
        slices_manifest_path=slices_json,
        calibration_manifest_path=calib_json,
        output_dir=output_dir
    )

    elapsed = round(time.time() - start_time, 2)
    slicer_data = step4_res["report"]["3d_printing_slicer_audit"]

    print("\n" + "=" * 75)
    print("  3D RECONSTRUCTION COMPLETE (SUCCESS)")
    print("=" * 75)
    print(f"  Execution Time         : {elapsed} seconds")
    print(f"  Production STL File    : {step4_res['prod_stl']}")
    print(f"  Production 3MF File    : {step4_res['prod_3mf']}")
    print(f"  Printability QC Sheet  : {step4_res['qc_sheet']}")
    print(f"  TPU Flexible Weight    : {slicer_data['material_weights_grams']['TPU_Flexible']} grams")
    print(f"  Estimated Print Time   : ~{slicer_data['estimated_print_time_hours']} hours")
    print("=" * 75)
    print("\nTo view the 3D model in Windows, run:")
    print(f"    start {step4_res['prod_stl']}")


def main():
    parser = argparse.ArgumentParser(description="2D to 3D Mannequin Hand Reconstruction Pipeline")
    parser.add_argument("--ap", type=str, default=str(BASE_DIR / "data" / "AP_Sample_Img.jpeg"),
                        help="Path to AP (Front) image")
    parser.add_argument("--lat", type=str, default=str(BASE_DIR / "data" / "L_Sample_Img.jpeg"),
                        help="Path to Lateral (Side) image")
    parser.add_argument("--width", type=float, default=56.0,
                        help="Patient wrist width in mm across crease (default: 56.0)")
    parser.add_argument("--depth", type=float, default=38.0,
                        help="Patient wrist depth in mm across crease (default: 38.0)")
    parser.add_argument("--circumference", type=float, default=150.0,
                        help="Patient wrist circumference in mm (default: 150.0)")
    parser.add_argument("--output", type=str, default=str(BASE_DIR / "output"),
                        help="Output directory path")

    args = parser.parse_args()

    ap_file = Path(args.ap)
    lat_file = Path(args.lat)
    out_dir = Path(args.output)

    if not ap_file.exists() or not lat_file.exists():
        print(f"[ERROR] Input image not found: {ap_file} or {lat_file}")
        sys.exit(1)

    run_full_pipeline(
        ap_path=ap_file,
        lat_path=lat_file,
        wrist_w_mm=args.width,
        wrist_d_mm=args.depth,
        wrist_c_mm=args.circumference,
        output_dir=out_dir
    )


if __name__ == "__main__":
    main()


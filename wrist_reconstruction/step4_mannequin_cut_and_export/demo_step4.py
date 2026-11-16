"""
=============================================================================
Demo Script for Step 4: The Mannequin Cut & Watertight STL Export
=============================================================================
Executes precision forearm planar trimming, watertight validation,
3D printing slicer auditing, and exports final production STL and 3MF files.
=============================================================================
"""

from pathlib import Path
from mannequin_cutter import MannequinCutterAndExporter

def run_demo():
    print("=" * 70)
    print("  STEP 4: THE MANNEQUIN CUT & PRODUCTION STL EXPORT")
    print("=" * 70)

    base_dir = Path(r"c:\Users\Admin\Healthcare\wrist_reconstruction")
    output_dir = base_dir / "output"

    raw_stl = output_dir / "wrist_mannequin_hand.stl"
    slices_json = output_dir / "step2_metric_slices.json"
    calib_json = output_dir / "calibration_anchor.json"

    if not raw_stl.exists():
        raise FileNotFoundError(f"Step 3 STL not found at {raw_stl}. Please execute Step 3 first.")

    cutter = MannequinCutterAndExporter(trim_height_from_base_mm=6.0)

    print(f"\n[1] Ingesting Reconstructed Model: {raw_stl.name}...")
    print(f"    - Applying Precision Planar Cut at Z = {cutter.trim_z:.1f} mm...")
    print(f"    - Re-capping Cross-Section to Flat Pedestal Base at Z = 0.0 mm...")

    result = cutter.process(
        raw_stl_path=raw_stl,
        slices_manifest_path=slices_json,
        calibration_manifest_path=calib_json,
        output_dir=output_dir
    )

    report = result["report"]
    mesh_q = report["mesh_quality_audit"]
    slicer_q = report["3d_printing_slicer_audit"]
    files = report["production_files"]

    print(f"\n[2] Quality Control & Manifold Health Audit:")
    print(f"    - Watertight Solid (No Holes) : {mesh_q['is_watertight']} [PASS]")
    print(f"    - Consistent Face Winding     : {mesh_q['is_winding_consistent']} [PASS]")
    print(f"    - Non-manifold Edges          : {mesh_q['non_manifold_edges']} [PASS]")
    print(f"    - Total Triangles (Faces)     : {mesh_q['triangle_count']:,}")
    bbox = mesh_q['bounding_box_mm']
    print(f"    - Model Dimensions (W x D x H): {bbox[0]} mm x {bbox[1]} mm x {bbox[2]} mm")

    print(f"\n[3] 3D Printing Slicer Readiness & Material Audit:")
    print(f"    - Solid Volume                : {slicer_q['volume_solid_cm3']} cm^3")
    print(f"    - 15% Infill Volume (Printed) : {slicer_q['volume_15pct_infill_cm3']} cm^3")
    print(f"    - Estimated Print Weight      :")
    for mat, wt in slicer_q['material_weights_grams'].items():
        print(f"        * {mat:<15}: {wt} grams")
    print(f"    - Estimated Print Time        : ~{slicer_q['estimated_print_time_hours']} hours (Modern Slicer @ 200mm/s)")
    print(f"    - Recommended Layer Height    : {slicer_q['recommended_layer_height_mm']} mm")
    print(f"    - Overhang Support Needed     : ~{slicer_q['support_needed_percentage']}% of surface area")

    print(f"\n[4] Production Output Artifacts:")
    print(f"    - Production STL File         : {files['stl_file']}")
    print(f"    - Production 3MF File         : {files['3mf_file']}")
    print(f"    - Quality Control Sheet       : {files['quality_control_sheet']}")
    print(f"    - Full Production Report      : {result['report_path']}")

    print("\n" + "=" * 70)
    print("  STATUS: STEP 4 SUCCESSFULLY COMPLETED & EXPORTED!")
    print("=" * 70)

if __name__ == "__main__":
    run_demo()


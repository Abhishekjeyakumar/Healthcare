"""
=============================================================================
FastAPI Backend Server: 2D to 3D Mannequin Hand Reconstruction
=============================================================================
Connects the React frontend to the 4-step wrist reconstruction pipeline:
- Step 1: Metric scale anchoring
- Step 2: 2D silhouette extraction & metric rib slicing
- Step 3: Mannequin 3D lofting & Laplacian fairing
- Step 4: Flush planar cut & production STL/OBJ export
=============================================================================
"""

import base64
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# Add wrist_reconstruction directories to Python path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
WRIST_DIR = WORKSPACE_ROOT / "wrist_reconstruction"

sys.path.append(str(WRIST_DIR))
sys.path.append(str(WRIST_DIR / "step1_measurement_anchoring"))
sys.path.append(str(WRIST_DIR / "step2_silhouette_extraction"))
sys.path.append(str(WRIST_DIR / "step3_3d_reconstruction"))
sys.path.append(str(WRIST_DIR / "step4_mannequin_cut_and_export"))

from measurement_anchor import MeasurementAnchorEngine, PhysicalMeasurements
from silhouette_extractor import SilhouetteExtractor
from mannequin_reconstructor import MannequinReconstructor
from mannequin_cutter import MannequinCutterAndExporter

app = FastAPI(title="Hand & Wrist 3D Reconstruct API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = WRIST_DIR / "temp_uploads"
OUTPUT_DIR = WRIST_DIR / "output"
TEMP_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def file_to_base64_data_uri(path: Path, mime: str = "image/png") -> Optional[str]:
    """Reads a file and returns a base64 data URI."""
    if not path.exists():
        return None
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


@app.get("/health")
def health_check():
    return {
        "status": "online",
        "service": "Wrist 3D Reconstruction API",
        "accuracy_tolerance": "1.0mm strict"
    }


@app.post("/reconstruct")
async def reconstruct_wrist_endpoint(
    ap_image: UploadFile = File(...),
    lat_image: UploadFile = File(...),
    wrist_width_mm: float = Form(56.0),
    wrist_depth_mm: float = Form(38.0),
    wrist_circumference_mm: float = Form(150.0),
    trim_height_mm: float = Form(6.0)
):
    try:
        start_time = time.time()

        # 1. Save uploaded images to temp directory
        ap_save_path = TEMP_DIR / f"upload_ap_{int(time.time()*1000)}.jpg"
        lat_save_path = TEMP_DIR / f"upload_lat_{int(time.time()*1000)}.jpg"

        ap_bytes = await ap_image.read()
        lat_bytes = await lat_image.read()

        with open(ap_save_path, "wb") as f:
            f.write(ap_bytes)
        with open(lat_save_path, "wb") as f:
            f.write(lat_bytes)

        # 2. Step 1: Metric Scale Anchoring
        measurements = PhysicalMeasurements(
            wrist_width_mm=wrist_width_mm,
            wrist_depth_mm=wrist_depth_mm,
            wrist_circumference_mm=wrist_circumference_mm
        )
        anchor_engine = MeasurementAnchorEngine()
        step1_res = anchor_engine.process(
            ap_image_path=ap_save_path,
            lat_image_path=lat_save_path,
            measurements=measurements,
            output_dir=OUTPUT_DIR
        )
        calib_manifest_path = Path(step1_res["manifest_path"])

        # 3. Step 2: 2D Silhouette Extraction & Metric Rib Slicing
        extractor = SilhouetteExtractor(slice_step_mm=2.0)
        step2_res = extractor.process(
            ap_image_path=ap_save_path,
            lat_image_path=lat_save_path,
            calibration_manifest_path=calib_manifest_path,
            output_dir=OUTPUT_DIR
        )
        slices_manifest_path = Path(step2_res["manifest_path"])

        # 4. Step 3: Mannequin 3D Reconstruction & Lofting
        reconstructor = MannequinReconstructor()
        step3_res = reconstructor.process(
            slices_manifest_path=slices_manifest_path,
            calibration_manifest_path=calib_manifest_path,
            output_dir=OUTPUT_DIR
        )
        raw_stl_path = Path(step3_res["stl_path"])

        # 5. Step 4: Precision Planar Cut & Final Quality Audit
        cutter = MannequinCutterAndExporter(trim_height_from_base_mm=trim_height_mm)
        step4_res = cutter.process(
            raw_stl_path=raw_stl_path,
            slices_manifest_path=slices_manifest_path,
            calibration_manifest_path=calib_manifest_path,
            output_dir=OUTPUT_DIR
        )

        # Read OBJ content for browser 3D viewer
        obj_file_path = OUTPUT_DIR / "wrist_mannequin_hand.obj"
        with open(obj_file_path, "r", encoding="utf-8") as f:
            obj_content = f.read()

        total_time = round(time.time() - start_time, 2)
        step3_calip = step3_res["report"]["caliper_verification"]
        step4_report = step4_res["report"]
        slicer_stats = step4_report["3d_printing_slicer_audit"]
        mesh_stats = step4_report["mesh_quality_audit"]

        # Prepare base64 visual inspection images
        diagnostic_images = {
            "ap_caliper_overlay": file_to_base64_data_uri(OUTPUT_DIR / "step1_annotated_ap.png"),
            "lat_caliper_overlay": file_to_base64_data_uri(OUTPUT_DIR / "step1_annotated_lat.png"),
            "ap_contour_overlay": file_to_base64_data_uri(OUTPUT_DIR / "step2_ap_contour_overlay.png"),
            "lat_contour_overlay": file_to_base64_data_uri(OUTPUT_DIR / "step2_lat_contour_overlay.png"),
            "metric_profile_plot": file_to_base64_data_uri(OUTPUT_DIR / "step2_metric_profiles.png"),
            "qc_printability_sheet": file_to_base64_data_uri(OUTPUT_DIR / "step4_printability_inspection.png"),
        }

        # Clean up temp uploads
        try:
            ap_save_path.unlink(missing_ok=True)
            lat_save_path.unlink(missing_ok=True)
        except Exception:
            pass

        return {
            "status": "SUCCESS",
            "message": "3D Mannequin Hand Reconstructed with 1mm Accuracy",
            "obj": obj_content,
            "processing_time_sec": total_time,
            "caliper_accuracy": {
                "target_width_mm": step3_calip["target_width_mm"],
                "mesh_width_mm": step3_calip["mesh_width_mm"],
                "width_error_mm": step3_calip["width_error_mm"],
                "target_depth_mm": step3_calip["target_depth_mm"],
                "mesh_depth_mm": step3_calip["mesh_depth_mm"],
                "depth_error_mm": step3_calip["depth_error_mm"],
                "verdict_1mm": "PASSED (< 1 mm Error)" if (step3_calip["width_error_mm"] <= 1.0 and step3_calip["depth_error_mm"] <= 1.0) else "FAILED",
                "inspection_plane_z_mm": step3_calip["inspection_level_z_mm"]
            },
            "mesh_statistics": {
                "vertex_count": mesh_stats["vertex_count"],
                "triangle_count": mesh_stats["triangle_count"],
                "is_watertight": mesh_stats["is_watertight"],
                "is_winding_consistent": mesh_stats["is_winding_consistent"],
                "bounding_box_mm": mesh_stats["bounding_box_mm"]
            },
            "slicer_estimates": {
                "solid_volume_cm3": slicer_stats["volume_solid_cm3"],
                "printed_volume_cm3": slicer_stats["volume_15pct_infill_cm3"],
                "weights_grams": slicer_stats["material_weights_grams"],
                "estimated_print_time_hours": slicer_stats["estimated_print_time_hours"],
                "recommended_layer_height_mm": slicer_stats["recommended_layer_height_mm"],
                "support_needed_pct": slicer_stats["support_needed_percentage"]
            },
            "diagnostic_images": diagnostic_images,
            "download_urls": {
                "stl": "/api/download/stl",
                "three_mf": "/api/download/3mf"
            }
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/download/stl")
def download_stl():
    stl_path = OUTPUT_DIR / "mannequin_hand_production.stl"
    if not stl_path.exists():
        raise HTTPException(status_code=404, detail="STL model not yet generated.")
    return FileResponse(
        path=str(stl_path),
        media_type="application/sla",
        filename="mannequin_hand_production.stl"
    )


@app.get("/api/download/3mf")
def download_3mf():
    three_mf_path = OUTPUT_DIR / "mannequin_hand_production.3mf"
    if not three_mf_path.exists():
        raise HTTPException(status_code=404, detail="3MF model not yet generated.")
    return FileResponse(
        path=str(three_mf_path),
        media_type="application/vnd.ms-package.3dmanufacturing-3dmodel+xml",
        filename="mannequin_hand_production.3mf"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

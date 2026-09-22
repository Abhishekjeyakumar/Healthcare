from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import time

from reconstruction import inflate_silhouettes, generate_obj_string

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def read_image_from_bytes(data):
    np_img = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    return img

@app.post("/reconstruct")
async def reconstruct(
    ap_image: UploadFile = File(...),
    lat_image: UploadFile = File(...),
    threshold: float = Form(0.5),
    sigma: float = Form(1.5),
    size: int = Form(128)
):
    try:
        start_time = time.time()
        
        ap_bytes = await ap_image.read()
        lat_bytes = await lat_image.read()
        
        ap_img = read_image_from_bytes(ap_bytes)
        lat_img = read_image_from_bytes(lat_bytes)
        
        if ap_img is None or lat_img is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image files provided."})

        # Generate smooth 3D mannequin mesh from input silhouettes
        # Use dense vertical sampling so knee and calf mask-edge turns survive
        # interpolation into the mesh.
        verts, faces, normals = inflate_silhouettes(ap_img, lat_img, resolution=256, radial_segments=96)
        
        # Generate OBJ string with smooth vertex normals
        obj_string = generate_obj_string(verts, faces, normals)
        
        processing_time = time.time() - start_time
        
        return {
            "message": "Reconstruction complete",
            "obj": obj_string,
            "debug": {
                "ap_resolution": f"{ap_img.shape[1]}x{ap_img.shape[0]}",
                "lat_resolution": f"{lat_img.shape[1]}x{lat_img.shape[0]}",
                "volume_dimensions": "256x96 slices",
                "vertices_count": int(len(verts)),
                "faces_count": int(len(faces)),
                "processing_time_sec": float(round(processing_time, 2))
            }
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

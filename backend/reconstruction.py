import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
import trimesh

def extract_mask(image):
    """
    Extracts a binary silhouette mask from the X-ray image.
    """
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # The bundled images are already binary masks.  Do not blur or morph those
    # masks: doing so moves their black/white edge inward and removes the small
    # posterior knee and calf transitions that define the lateral silhouette.
    values = np.unique(gray)
    is_binary_mask = len(values) <= 2
    if is_binary_mask:
        thresh = np.where(gray > 127, 255, 0).astype(np.uint8)
    else:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # If background is bright, invert
    border_pixels = np.concatenate([thresh[0, :], thresh[-1, :], thresh[:, 0], thresh[:, -1]])
    if np.mean(border_pixels) > 127:
        thresh = cv2.bitwise_not(thresh)

    # Cleanup is appropriate for a radiograph, but is deliberately skipped for
    # a supplied mask so its drawn boundary remains the reconstruction boundary.
    if not is_binary_mask:
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Keep only the largest connected component
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        thresh = np.zeros_like(thresh)
        thresh[labels == largest_label] = 255
        
    return thresh

def get_profile_from_mask(mask):
    """
    Given a binary mask, extracts the min and max coordinates for each row.
    Returns normalized y (0 to 1), and corresponding min/max bounds.
    """
    y_coords, x_coords = np.where(mask > 0)
    if len(y_coords) == 0:
        return None
        
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    height = y_max - y_min
    
    profile = {}
    for y, x in zip(y_coords, x_coords):
        if y not in profile:
            profile[y] = [x, x]
        else:
            profile[y][0] = min(profile[y][0], x)
            profile[y][1] = max(profile[y][1], x)
            
    # Normalize y to 0-1, and x to -0.5 to 0.5 (relative to image width)
    img_width = mask.shape[1]
    
    normalized_profile = []
    for y in range(y_min, y_max + 1):
        if y in profile:
            norm_y = 1.0 - ((y - y_min) / height) # 1.0 is top, 0.0 is bottom
            norm_min = (profile[y][0] - img_width/2) / height
            norm_max = (profile[y][1] - img_width/2) / height
            normalized_profile.append((norm_y, norm_min, norm_max))
            
    return normalized_profile

def inflate_silhouettes(ap_img, lat_img, resolution=256, radial_segments=96, smooth_sigma=1.15):
    """
    Generates a 3D mannequin mesh by inflating and smoothing the 2D silhouettes.
    """
    ap_mask = extract_mask(ap_img)
    lat_mask = extract_mask(lat_img)
    
    ap_profile = get_profile_from_mask(ap_mask)
    lat_profile = get_profile_from_mask(lat_mask)
    
    # Fallback to cylinder if extraction completely fails
    if not ap_profile or not lat_profile:
        return generate_fallback_cylinder(resolution, radial_segments)
        
    # Sort profiles by Y descending (1.0 to 0.0)
    ap_profile.sort(key=lambda p: p[0], reverse=True)
    lat_profile.sort(key=lambda p: p[0], reverse=True)
    
    ap_y = [p[0] for p in ap_profile]
    ap_min = [p[1] for p in ap_profile]
    ap_max = [p[2] for p in ap_profile]
    
    lat_y = [p[0] for p in lat_profile]
    lat_min = [p[1] for p in lat_profile]
    lat_max = [p[2] for p in lat_profile]
    
    # Common Y sampling points
    y_vals = np.linspace(1.0, 0.0, resolution)
    
    # Interpolate bounds
    ap_min_interp = np.interp(y_vals[::-1], ap_y[::-1], ap_min[::-1])[::-1]
    ap_max_interp = np.interp(y_vals[::-1], ap_y[::-1], ap_max[::-1])[::-1]
    
    lat_min_interp = np.interp(y_vals[::-1], lat_y[::-1], lat_min[::-1])[::-1]
    lat_max_interp = np.interp(y_vals[::-1], lat_y[::-1], lat_max[::-1])[::-1]
    
    # Compute center lines and radii
    cx = (ap_min_interp + ap_max_interp) / 2.0
    cz = (lat_min_interp + lat_max_interp) / 2.0
    rx = (ap_max_interp - ap_min_interp) / 2.0
    rz = (lat_max_interp - lat_min_interp) / 2.0
    
    # Smooth at about one sampled slice: this removes the pixel staircase from
    # the mask edge while retaining the broader calf and knee transitions.
    cx = gaussian_filter1d(cx, sigma=smooth_sigma, mode='nearest')
    cz = gaussian_filter1d(cz, sigma=smooth_sigma, mode='nearest')
    rx = gaussian_filter1d(rx, sigma=smooth_sigma, mode='nearest')
    rz = gaussian_filter1d(rz, sigma=smooth_sigma, mode='nearest')
    
    # Ensure minimum radius to avoid degenerate geometry
    rx = np.maximum(rx, 0.02)
    rz = np.maximum(rz, 0.02)
    
    # 2. Gently round the top and bottom end slices for a sleek mannequin limb finish
    n_cap = 6
    for k in range(n_cap):
        factor = np.sin((k + 1) / (n_cap + 1) * np.pi / 2)
        # Top end
        rx[k] *= factor
        rz[k] *= factor
        # Bottom end
        rx[-(k+1)] *= factor
        rz[-(k+1)] *= factor

    vertices = []
    faces = []
    
    for i in range(resolution):
        y = y_vals[i]
        c_x = cx[i]
        c_z = cz[i]
        r_x = rx[i]
        r_z = rz[i]
        
        for j in range(radial_segments):
            theta = 2.0 * np.pi * j / radial_segments
            x = c_x + r_x * np.cos(theta)
            z = c_z + r_z * np.sin(theta)
            
            # Scale coordinates appropriately
            vertices.append([x * 3.0, (y - 0.5) * 4.0, z * 3.0])
            
    # Add top and bottom center pole vertices
    top_center_idx = len(vertices)
    vertices.append([cx[0] * 3.0, (y_vals[0] - 0.5) * 4.0, cz[0] * 3.0])
    
    bottom_center_idx = len(vertices)
    vertices.append([cx[-1] * 3.0, (y_vals[-1] - 0.5) * 4.0, cz[-1] * 3.0])

    # Generate side faces
    for i in range(resolution - 1):
        for j in range(radial_segments):
            next_j = (j + 1) % radial_segments
            
            v1 = i * radial_segments + j
            v2 = i * radial_segments + next_j
            v3 = (i + 1) * radial_segments + j
            v4 = (i + 1) * radial_segments + next_j
            
            faces.append([v1, v2, v3])
            faces.append([v2, v4, v3])
            
    # Close top
    for j in range(radial_segments):
        next_j = (j + 1) % radial_segments
        v1 = j
        v2 = next_j
        faces.append([v2, v1, top_center_idx])
        
    # Close bottom
    for j in range(radial_segments):
        next_j = (j + 1) % radial_segments
        v1 = (resolution - 1) * radial_segments + j
        v2 = (resolution - 1) * radial_segments + next_j
        faces.append([v1, v2, bottom_center_idx])
        
    raw_verts = np.array(vertices, dtype=np.float64)
    raw_faces = np.array(faces, dtype=np.int64)
    
    # Dense rings plus interpolated normals make the surface smooth without a
    # destructive mesh-wide relaxation pass.
    mesh = trimesh.Trimesh(vertices=raw_verts, faces=raw_faces, process=False)
    
    # Recompute vertex normals for smooth Phong shading
    normals = mesh.vertex_normals
    
    return mesh.vertices, mesh.faces, normals

def generate_fallback_cylinder(resolution, radial_segments):
    vertices, faces = [], []
    for i in range(resolution):
        y = 1.0 - i / (resolution - 1)
        for j in range(radial_segments):
            theta = 2.0 * np.pi * j / radial_segments
            vertices.append([np.cos(theta), (y - 0.5) * 4.0, np.sin(theta)])
            
    top_center_idx = len(vertices)
    vertices.append([0.0, (1.0 - 0.5) * 4.0, 0.0])
    bottom_center_idx = len(vertices)
    vertices.append([0.0, (0.0 - 0.5) * 4.0, 0.0])
    
    for i in range(resolution - 1):
        for j in range(radial_segments):
            next_j = (j + 1) % radial_segments
            faces.append([i*radial_segments + j, i*radial_segments + next_j, (i+1)*radial_segments + j])
            faces.append([i*radial_segments + next_j, (i+1)*radial_segments + next_j, (i+1)*radial_segments + j])
            
    mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces), process=False)
    return mesh.vertices, mesh.faces, mesh.vertex_normals

def generate_obj_string(vertices, faces, normals=None):
    lines = []
    lines.append("# Biplanar X-Ray Reconstructed Mannequin Mesh")
    for v in vertices:
        lines.append(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}")
    if normals is not None and len(normals) == len(vertices):
        for n in normals:
            lines.append(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}")
        for f in faces:
            lines.append(f"f {f[0]+1}//{f[0]+1} {f[1]+1}//{f[1]+1} {f[2]+1}//{f[2]+1}")
    else:
        for f in faces:
            lines.append(f"f {f[0]+1} {f[1]+1} {f[2]+1}")
    return "\n".join(lines)

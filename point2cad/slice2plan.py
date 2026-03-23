"""
Slice-based point cloud to 2D floor plan pipeline.

Takes a raw point cloud (LAZ/LAS/PLY/E57/etc.), slices it at a given height,
rasterizes the slice to a 2D density image, traces wall lines, and exports
a DXF floor plan.

This is a simpler, more robust alternative to the full Point2CAD surface
fitting pipeline — designed for building scans (GeoSLAM, FARO, etc.).

Dependencies: laspy, lazrs, numpy, scipy, ezdxf (all pure Python / C-ext,
no Open3D required).

Usage:
    python -m point2cad.slice2plan scan.laz --output plan.dxf
    python -m point2cad.slice2plan scan.laz --slice_height 1.2 --resolution 0.02
"""

import os
import numpy as np


# ---------------------------------------------------------------------------
# 1. Point cloud loading (reuse input_adapter readers, no Open3D)
# ---------------------------------------------------------------------------

def load_points(path):
    """Load XYZ points from any supported format. Returns Nx3 float32 array."""
    from point2cad.input_adapter import detect_format, _READERS

    fmt = detect_format(path)
    reader = _READERS[fmt]
    print(f"Reading {fmt.upper()} file: {path}")
    points, _ = reader(path)
    print(f"  Loaded {len(points):,} points")

    # Filter NaN/Inf
    valid = np.all(np.isfinite(points), axis=1)
    if not np.all(valid):
        n_bad = np.sum(~valid)
        print(f"  Filtered {n_bad:,} invalid points")
        points = points[valid]

    return points.astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Horizontal slice extraction
# ---------------------------------------------------------------------------

def slice_points(points, height, thickness=0.3):
    """Extract a horizontal slab of points at the given Z height.

    Args:
        points: Nx3 array.
        height: Center Z height of the slice (metres).
        thickness: Total thickness of the slab (metres). Points within
            [height - thickness/2, height + thickness/2] are kept.

    Returns:
        Mx3 array of points in the slab.
    """
    half = thickness / 2.0
    mask = (points[:, 2] >= height - half) & (points[:, 2] <= height + half)
    slab = points[mask]
    print(f"  Slice at z={height:.2f}m (±{half:.2f}m): {len(slab):,} points")
    return slab


def auto_slice_height(points, floor_percentile=5, wall_offset=1.0):
    """Estimate a good slice height automatically.

    Finds the floor level (low Z percentile) and adds an offset to capture
    wall cross-sections at roughly waist/chest height, avoiding floor clutter
    and furniture.

    Args:
        points: Nx3 array.
        floor_percentile: Percentile of Z values to estimate floor level.
        wall_offset: Height above floor to slice (metres).

    Returns:
        Estimated slice height (float).
    """
    z_vals = points[:, 2]
    floor_z = np.percentile(z_vals, floor_percentile)
    ceiling_z = np.percentile(z_vals, 95)
    height = floor_z + wall_offset
    print(f"  Auto height: floor={floor_z:.2f}m, ceiling={ceiling_z:.2f}m, "
          f"slice={height:.2f}m")
    return height


# ---------------------------------------------------------------------------
# 3. Rasterization to 2D density image
# ---------------------------------------------------------------------------

def rasterize(points_2d, resolution=0.02):
    """Rasterize 2D points to a density grid.

    Args:
        points_2d: Mx2 array (X, Y positions from the slice).
        resolution: Grid cell size in metres.

    Returns:
        (image, origin, resolution) where:
          - image is a 2D uint8 array (0-255, higher = more points)
          - origin is (x_min, y_min) of the grid in world coords
    """
    x = points_2d[:, 0]
    y = points_2d[:, 1]

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    # Add a small margin
    margin = resolution * 5
    x_min -= margin
    y_min -= margin
    x_max += margin
    y_max += margin

    nx = int(np.ceil((x_max - x_min) / resolution))
    ny = int(np.ceil((y_max - y_min) / resolution))

    print(f"  Raster grid: {nx} x {ny} pixels ({resolution}m/px)")

    # Bin points into grid cells
    ix = np.clip(((x - x_min) / resolution).astype(int), 0, nx - 1)
    iy = np.clip(((y - y_min) / resolution).astype(int), 0, ny - 1)

    grid = np.zeros((ny, nx), dtype=np.int32)
    np.add.at(grid, (iy, ix), 1)

    # Normalize to 0-255
    if grid.max() > 0:
        image = (grid.astype(np.float32) / grid.max() * 255).astype(np.uint8)
    else:
        image = grid.astype(np.uint8)

    origin = (x_min, y_min)
    return image, origin, resolution


# ---------------------------------------------------------------------------
# 4. Wall detection — pixel chaining + polyline simplification
#
# Tuned for GeoSLAM/FARO scanner orthoimages where walls appear as thin
# bright lines (1-3px) on a black background. The key insight is that
# walls are already thin — no morphological closing/dilation needed (that
# would merge parallel walls). Instead we:
#   1. Contrast-stretch the very faint wall pixels
#   2. Threshold to binary
#   3. Light cleanup (close 1px gaps, remove isolated noise)
#   4. Skeletonize to guaranteed 1px-wide centrelines
#   5. Chain skeleton pixels into ordered polylines
#   6. Simplify each polyline (RDP) into clean wall segments
# ---------------------------------------------------------------------------

def detect_walls(image, origin, resolution, min_density=3,
                 min_line_length=0.3, simplify_tolerance=None,
                 morph_iterations=None):
    """Detect wall segments from a scanner orthoimage.

    Args:
        image: 2D uint8 density image (bright = walls, dark = empty).
        origin: (x_min, y_min) world coordinate origin.
        resolution: Grid cell size in metres.
        min_density: Minimum pixel value to consider as wall.
        min_line_length: Minimum wall segment length in metres.
        simplify_tolerance: RDP tolerance in metres (default: 3*resolution).
        morph_iterations: Unused, kept for CLI compatibility.

    Returns:
        List of ((x1, y1), (x2, y2)) line segments in world coordinates.
    """
    from scipy import ndimage

    if simplify_tolerance is None:
        simplify_tolerance = resolution * 3.0

    # --- 1. Contrast stretch ---
    # Scanner orthoimages have very faint walls (pixel values 2-40).
    # Stretch the non-zero range to 0-255 so thresholding works well.
    nonzero = image[image > 0]
    if len(nonzero) == 0:
        print("  WARNING: Image is completely black.")
        return []

    p_low = np.percentile(nonzero, 5)
    p_high = np.percentile(nonzero, 99)
    print(f"  Pixel stats: non-zero range [{nonzero.min()}-{nonzero.max()}], "
          f"p5={p_low:.0f}, p99={p_high:.0f}")

    if p_high > p_low:
        stretched = np.clip(
            (image.astype(np.float32) - p_low) / (p_high - p_low) * 255,
            0, 255
        ).astype(np.uint8)
    else:
        stretched = image.copy()

    # --- 2. Threshold ---
    # After stretching, walls should be bright. Use a moderate threshold
    # to separate walls from the diffuse scanner glow.
    threshold = max(min_density, 30)  # post-stretch threshold
    binary = (stretched >= threshold).astype(np.uint8)
    n_wall_px = binary.sum()
    print(f"  Binary mask: {n_wall_px:,} wall pixels "
          f"({100*n_wall_px/binary.size:.1f}% of image)")

    if n_wall_px == 0:
        print("  WARNING: No wall pixels found. Try lowering --min_density.")
        return []

    # --- 3. Light cleanup ---
    struct8 = ndimage.generate_binary_structure(2, 2)  # 8-connected
    struct4 = ndimage.generate_binary_structure(2, 1)  # 4-connected

    # Close tiny 1px gaps in wall lines (but don't over-thicken)
    binary = ndimage.binary_closing(
        binary, structure=struct4, iterations=1
    ).astype(np.uint8)

    # Remove small isolated noise blobs
    labeled_noise, n_noise = ndimage.label(binary, structure=struct8)
    if n_noise > 0:
        sizes = ndimage.sum(binary, labeled_noise, range(1, n_noise + 1))
        min_blob = max(5, int(0.1 / resolution))  # 10cm minimum blob
        for i, sz in enumerate(sizes):
            if sz < min_blob:
                binary[labeled_noise == (i + 1)] = 0
        n_removed = sum(1 for s in sizes if s < min_blob)
        if n_removed > 0:
            print(f"  Removed {n_removed} noise blobs (< {min_blob}px)")

    # --- 4. Skeletonize to 1px centrelines ---
    skeleton = _skeletonize_proper(binary)
    n_skel = skeleton.sum()
    print(f"  Skeleton: {n_skel:,} pixels")

    if n_skel == 0:
        return []

    # --- 5. Chain skeleton pixels into polylines ---
    polylines = _chain_skeleton(skeleton)
    print(f"  Chained into {len(polylines)} polylines")

    # --- 6. Convert to world coords, simplify, emit segments ---
    min_px_len = max(3, int(min_line_length / resolution))
    all_segments = []

    for chain in polylines:
        if len(chain) < min_px_len:
            continue

        # Convert (row, col) to world (x, y)
        world_pts = np.empty((len(chain), 2), dtype=np.float64)
        world_pts[:, 0] = chain[:, 1] * resolution + origin[0]  # col -> x
        world_pts[:, 1] = chain[:, 0] * resolution + origin[1]  # row -> y

        simplified = _rdp_simplify(world_pts, simplify_tolerance)
        if len(simplified) < 2:
            continue

        for i in range(len(simplified) - 1):
            p1, p2 = simplified[i], simplified[i + 1]
            seg_len = np.linalg.norm(p2 - p1)
            if seg_len >= min_line_length:
                all_segments.append(
                    ((p1[0], p1[1]), (p2[0], p2[1]))
                )

    print(f"  Final: {len(all_segments)} wall segments")
    return all_segments


def _skeletonize_proper(binary):
    """Zhang-Suen thinning to produce clean 1px-wide skeletons.

    Unlike simple erosion-based skeletonization, this preserves connectivity
    and produces clean centrelines suitable for chaining.
    """
    img = binary.astype(np.uint8).copy()
    rows, cols = img.shape
    changed = True

    while changed:
        changed = False
        for step in (0, 1):
            # Pad for safe neighbour access
            padded = np.pad(img, 1, mode='constant', constant_values=0)
            markers = np.zeros_like(img, dtype=bool)

            # Get all foreground pixels
            ys, xs = np.where(img == 1)

            for idx in range(len(ys)):
                r, c = ys[idx], xs[idx]
                rp, cp = r + 1, c + 1  # padded coords

                # 8-neighbours (P2..P9 in Zhang-Suen convention)
                p2 = padded[rp - 1, cp]
                p3 = padded[rp - 1, cp + 1]
                p4 = padded[rp, cp + 1]
                p5 = padded[rp + 1, cp + 1]
                p6 = padded[rp + 1, cp]
                p7 = padded[rp + 1, cp - 1]
                p8 = padded[rp, cp - 1]
                p9 = padded[rp - 1, cp - 1]

                neighbours = [p2, p3, p4, p5, p6, p7, p8, p9]
                B = sum(neighbours)  # number of non-zero neighbours

                if B < 2 or B > 6:
                    continue

                # Count 0->1 transitions in the ordered sequence
                A = 0
                seq = neighbours + [neighbours[0]]
                for k in range(8):
                    if seq[k] == 0 and seq[k + 1] == 1:
                        A += 1

                if A != 1:
                    continue

                if step == 0:
                    if p2 * p4 * p6 != 0:
                        continue
                    if p4 * p6 * p8 != 0:
                        continue
                else:
                    if p2 * p4 * p8 != 0:
                        continue
                    if p2 * p6 * p8 != 0:
                        continue

                markers[r, c] = True

            if markers.any():
                img[markers] = 0
                changed = True

    return img


def _chain_skeleton(skeleton):
    """Chain skeleton pixels into ordered polylines.

    Walks along connected skeleton pixels, starting from endpoints
    (pixels with only 1 neighbour) or junction pixels. Produces a list
    of Nx2 arrays of (row, col) coordinates.
    """
    skel = skeleton.astype(np.uint8).copy()
    rows, cols = skel.shape

    # Precompute neighbour count for each pixel
    from scipy import ndimage
    struct8 = ndimage.generate_binary_structure(2, 2)
    neighbour_count = ndimage.convolve(
        skel.astype(np.int32), struct8.astype(np.int32), mode='constant'
    ) - skel.astype(np.int32)  # subtract self

    # 8-neighbour offsets
    dr = [-1, -1, -1, 0, 0, 1, 1, 1]
    dc = [-1, 0, 1, -1, 1, -1, 0, 1]

    # Find endpoints (1 neighbour) — best starting points for chains
    endpoints = set()
    junctions = set()
    ys, xs = np.where(skel > 0)
    for r, c in zip(ys, xs):
        nc = neighbour_count[r, c]
        if nc == 1:
            endpoints.add((r, c))
        elif nc >= 3:
            junctions.add((r, c))

    visited = np.zeros_like(skel, dtype=bool)
    polylines = []

    def _walk(start_r, start_c):
        """Walk along connected skeleton pixels from a starting point."""
        chain = [(start_r, start_c)]
        visited[start_r, start_c] = True
        r, c = start_r, start_c

        while True:
            found_next = False
            for k in range(8):
                nr, nc_ = r + dr[k], c + dc[k]
                if (0 <= nr < rows and 0 <= nc_ < cols
                        and skel[nr, nc_] and not visited[nr, nc_]):
                    visited[nr, nc_] = True
                    chain.append((nr, nc_))
                    r, c = nr, nc_
                    found_next = True
                    break
            if not found_next:
                break

        return np.array(chain, dtype=np.float64)

    # Start from endpoints first (gives cleaner chains)
    for r, c in endpoints:
        if not visited[r, c]:
            chain = _walk(r, c)
            if len(chain) >= 2:
                polylines.append(chain)

    # Then pick up any remaining unvisited skeleton pixels (loops, etc.)
    for r, c in zip(ys, xs):
        if not visited[r, c]:
            chain = _walk(r, c)
            if len(chain) >= 2:
                polylines.append(chain)

    return polylines


def _rdp_simplify(points, tolerance):
    """Ramer-Douglas-Peucker polyline simplification.

    Args:
        points: Nx2 array of polyline vertices.
        tolerance: Maximum perpendicular distance to discard a point.

    Returns:
        Simplified Mx2 array (M <= N).
    """
    if len(points) <= 2:
        return points

    start, end = points[0], points[-1]
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)

    if line_len < 1e-12:
        dists = np.linalg.norm(points - start, axis=1)
        idx = np.argmax(dists)
        if dists[idx] > tolerance:
            return np.array([start, points[idx], end])
        return np.array([start, end])

    line_unit = line_vec / line_len
    vecs = points - start
    proj = vecs @ line_unit
    perp = vecs - np.outer(proj, line_unit)
    dists = np.linalg.norm(perp, axis=1)

    max_idx = np.argmax(dists)
    max_dist = dists[max_idx]

    if max_dist <= tolerance:
        return np.array([start, end])

    left = _rdp_simplify(points[:max_idx + 1], tolerance)
    right = _rdp_simplify(points[max_idx:], tolerance)

    return np.vstack([left[:-1], right])


# ---------------------------------------------------------------------------
# 5. DXF export
# ---------------------------------------------------------------------------

def export_dxf(segments, out_path, origin_offset=True):
    """Export wall line segments to a DXF file.

    Args:
        segments: List of ((x1,y1), (x2,y2)) in world coordinates (metres).
        out_path: Output DXF file path.
        origin_offset: If True, shift geometry so bottom-left is near origin.
    """
    import ezdxf
    from ezdxf import units

    doc = ezdxf.new("R2010")
    doc.units = units.M  # metres, matching the point cloud

    doc.layers.add("WALLS", color=7)  # White/black
    doc.layers.add("DIMENSIONS", color=6)  # Magenta

    msp = doc.modelspace()

    if not segments:
        print("  Warning: No wall segments to export.")
        doc.saveas(out_path)
        return

    # Optionally shift to near-origin
    all_pts = np.array([(p[0], p[1]) for seg in segments for p in seg])
    if origin_offset:
        shift = all_pts.min(axis=0) - 1.0  # 1m margin
    else:
        shift = np.zeros(2)

    for (x1, y1), (x2, y2) in segments:
        msp.add_line(
            (x1 - shift[0], y1 - shift[1]),
            (x2 - shift[0], y2 - shift[1]),
            dxfattribs={"layer": "WALLS"},
        )

    # Add bounding dimensions
    shifted = all_pts - shift
    min_x, min_y = shifted.min(axis=0)
    max_x, max_y = shifted.max(axis=0)
    width = max_x - min_x
    height = max_y - min_y

    dim_offset = max(width, height) * 0.05 + 0.5

    # Create dimension style
    dim_style = doc.dimstyles.new("PLAN")
    dim_style.dxf.dimtxt = max(width, height) * 0.015  # text height
    dim_style.dxf.dimasz = max(width, height) * 0.01   # arrow size

    # Width dimension (bottom)
    msp.add_linear_dim(
        base=(min_x, min_y - dim_offset),
        p1=(min_x, min_y),
        p2=(max_x, min_y),
        dimstyle="PLAN",
        dxfattribs={"layer": "DIMENSIONS"},
    ).render()

    # Height dimension (left)
    msp.add_linear_dim(
        base=(min_x - dim_offset, min_y),
        p1=(min_x, min_y),
        p2=(min_x, max_y),
        angle=90,
        dimstyle="PLAN",
        dxfattribs={"layer": "DIMENSIONS"},
    ).render()

    doc.saveas(out_path)
    print(f"  Saved DXF: {out_path}")
    print(f"  Extents: {width:.2f}m x {height:.2f}m")


# ---------------------------------------------------------------------------
# 6. Optional: save the orthoimage as PNG
# ---------------------------------------------------------------------------

def save_orthoimage(image, out_path):
    """Save the density raster as a PNG image.

    Uses raw PNM format fallback if Pillow is not available.
    """
    try:
        from PIL import Image
        img = Image.fromarray(image)
        img.save(out_path)
    except ImportError:
        # Fallback: save as PGM (portable graymap) — no dependencies
        if not out_path.endswith(".pgm"):
            out_path = os.path.splitext(out_path)[0] + ".pgm"
        h, w = image.shape
        with open(out_path, "wb") as f:
            f.write(f"P5\n{w} {h}\n255\n".encode())
            f.write(image.tobytes())
    print(f"  Saved orthoimage: {out_path}")


# ---------------------------------------------------------------------------
# 7. Load a pre-rendered PNG/JPG orthoimage
# ---------------------------------------------------------------------------

def load_image(path):
    """Load a grayscale image from PNG/JPG/BMP/PGM.

    Returns a 2D uint8 numpy array (grayscale).
    """
    try:
        from PIL import Image
        img = Image.open(path).convert("L")
        return np.array(img, dtype=np.uint8)
    except ImportError:
        pass

    # Fallback: try PGM (our own save format)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pgm":
        return _read_pgm(path)

    raise ImportError(
        f"Pillow is required to read {ext} images. "
        "Install with: pip install Pillow"
    )


def read_worldfile(image_path):
    """Try to read a sidecar worldfile (.pgw, .tfw, .jgw, .wld) for an image.

    Worldfiles are 6-line text files used in GIS to georeference raster images.
    Lines: [x_scale, rotation_y, rotation_x, y_scale, x_origin, y_origin].

    For floor plans we only care about the pixel size (x_scale) and the origin.

    Args:
        image_path: Path to the image file.

    Returns:
        (resolution, origin) tuple if a worldfile is found, or None.
        resolution is in the worldfile's units (typically metres).
        origin is (x_min, y_min).
    """
    base, ext = os.path.splitext(image_path)

    # Common worldfile extension patterns
    # .png -> .pgw, .jpg -> .jgw, .tif -> .tfw, or generic .wld
    wf_candidates = []
    if len(ext) >= 4:
        # "First + last + w" convention: .png -> .pgw, .tif -> .tfw
        wf_ext = ext[0:2] + ext[-1] + "w"
        wf_candidates.append(base + wf_ext)
        wf_candidates.append(base + wf_ext.upper())
    wf_candidates.append(base + ".wld")
    wf_candidates.append(base + ".WLD")
    # Also try appending "w" directly: .png -> .pngw
    wf_candidates.append(image_path + "w")
    wf_candidates.append(image_path + "W")

    for wf_path in wf_candidates:
        if not os.path.isfile(wf_path):
            continue
        try:
            with open(wf_path, "r") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            if len(lines) < 6:
                continue

            x_scale = float(lines[0])    # pixel width (metres/pixel)
            rot_y = float(lines[1])       # rotation (usually 0)
            rot_x = float(lines[2])       # rotation (usually 0)
            y_scale = float(lines[3])     # pixel height (negative = top-down)
            x_origin = float(lines[4])    # X of centre of top-left pixel
            y_origin = float(lines[5])    # Y of centre of top-left pixel

            resolution = abs(x_scale)
            # Origin is the top-left pixel centre; shift to corner
            origin_x = x_origin - resolution / 2.0
            origin_y = y_origin - abs(y_scale) / 2.0 if y_scale > 0 else y_origin + y_scale / 2.0

            print(f"  Found worldfile: {wf_path}")
            print(f"    Pixel size: {resolution}m, "
                  f"Origin: ({origin_x:.2f}, {origin_y:.2f})")

            if abs(rot_y) > 1e-6 or abs(rot_x) > 1e-6:
                print(f"    WARNING: Worldfile has rotation ({rot_y}, {rot_x}) "
                      f"which is ignored — assuming axis-aligned image.")

            return resolution, (origin_x, origin_y)
        except (ValueError, IndexError):
            continue

    return None


def _read_pgm(path):
    """Read a binary PGM (P5) file without Pillow."""
    with open(path, "rb") as f:
        magic = f.readline().strip()
        if magic != b"P5":
            raise ValueError(f"Not a PGM file: {path}")
        # Skip comments
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        w, h = map(int, line.split())
        maxval = int(f.readline().strip())
        data = f.read()
    image = np.frombuffer(data, dtype=np.uint8).reshape((h, w))
    if maxval != 255:
        image = (image.astype(np.float32) / maxval * 255).astype(np.uint8)
    return image


# ---------------------------------------------------------------------------
# 8. Main pipelines
# ---------------------------------------------------------------------------

def run_pipeline(input_path, output_dir=None, output_dxf=None,
                 slice_height=None, thickness=0.3, resolution=0.02,
                 min_density=3, save_image=True):
    """Run the full slice-to-plan pipeline.

    Args:
        input_path: Path to point cloud (LAZ/LAS/PLY/E57/etc.).
        output_dir: Output directory (default: same as input file).
        output_dxf: Explicit DXF output path (overrides output_dir).
        slice_height: Z height for the horizontal slice (metres).
            If None, auto-detected.
        thickness: Slab thickness for the slice (metres).
        resolution: Raster resolution in metres per pixel.
        min_density: Minimum raster pixel value for wall detection.
        save_image: If True, also save the orthoimage as PNG.

    Returns:
        Path to the output DXF file.
    """
    basename = os.path.splitext(os.path.basename(input_path))[0]
    if output_dir is None:
        output_dir = os.path.dirname(input_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    if output_dxf is None:
        output_dxf = os.path.join(output_dir, f"{basename}_plan.dxf")

    # 1. Load
    points = load_points(input_path)

    # 2. Subsample if huge (pure numpy, no Open3D)
    if len(points) > 2_000_000:
        target = 2_000_000
        print(f"  Subsampling {len(points):,} -> {target:,} points...")
        idx = np.random.default_rng(42).choice(len(points), target, replace=False)
        points = points[idx]

    # 3. Determine slice height
    if slice_height is None:
        slice_height = auto_slice_height(points)
    print(f"  Using slice height: {slice_height:.2f}m")

    # 4. Extract horizontal slice
    slab = slice_points(points, slice_height, thickness)
    if len(slab) < 10:
        print("  ERROR: Slice contains too few points. Try a different "
              "--slice_height or --thickness.")
        return None

    # 5. Project to 2D (drop Z)
    points_2d = slab[:, :2]

    # 6. Rasterize
    image, origin, res = rasterize(points_2d, resolution)

    # 7. Save orthoimage
    if save_image:
        img_path = os.path.join(output_dir, f"{basename}_ortho.png")
        save_orthoimage(image, img_path)

    # 8. Detect walls
    print("  Detecting walls...")
    segments = detect_walls(image, origin, res, min_density=min_density)

    # 9. Export DXF
    export_dxf(segments, output_dxf)

    return output_dxf


def run_pipeline_from_image(image_path, output_dxf=None, resolution=0.02,
                            min_density=3, invert=False):
    """Run wall detection + DXF export from a pre-rendered PNG orthoimage.

    Skips point cloud loading, slicing, and rasterization — reads the image
    and traces walls directly.

    Args:
        image_path: Path to a grayscale PNG/JPG/BMP/PGM image. Bright pixels
            are treated as walls (high point density). Use invert=True if
            walls are dark on a light background.
        output_dxf: Output DXF path (default: <image_name>_plan.dxf).
        resolution: Metres per pixel — controls the real-world scale of the
            output DXF. E.g. 0.02 means each pixel = 2cm.
        min_density: Minimum pixel brightness (0-255) to consider as wall.
        invert: If True, invert the image (dark pixels become walls).

    Returns:
        Path to the output DXF file.
    """
    basename = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.dirname(image_path) or "."

    if output_dxf is None:
        output_dxf = os.path.join(output_dir, f"{basename}_plan.dxf")

    print(f"Reading image: {image_path}")
    image = load_image(image_path)
    print(f"  Image size: {image.shape[1]} x {image.shape[0]} pixels")

    if invert:
        image = 255 - image
        print("  Inverted image (dark walls -> bright)")

    # Try worldfile for georeferencing; fall back to CLI --resolution
    wf = read_worldfile(image_path)
    if wf is not None:
        resolution, origin = wf
        print(f"  Using worldfile: {resolution} m/px, "
              f"origin=({origin[0]:.2f}, {origin[1]:.2f})")
    else:
        origin = (0.0, 0.0)
        print(f"  No worldfile found, using --resolution={resolution} m/px")

    print(f"  Real-world extents: "
          f"{image.shape[1]*resolution:.1f}m x {image.shape[0]*resolution:.1f}m")

    print("  Detecting walls...")
    segments = detect_walls(image, origin, resolution, min_density=min_density)

    export_dxf(segments, output_dxf)
    return output_dxf


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pgm"}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a 2D floor plan (DXF) from a point cloud or "
                    "a flattened orthoimage (PNG/JPG)."
    )
    parser.add_argument(
        "input", type=str,
        help="Input file: point cloud (LAZ/LAS/PLY/E57) or image (PNG/JPG/BMP)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output DXF file path (default: <input_name>_plan.dxf)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (default: same as input file)",
    )
    parser.add_argument(
        "--resolution", type=float, default=0.02,
        help="Metres per pixel (default: 0.02 = 2cm). For image inputs this "
             "sets the real-world scale; for point clouds it sets the raster "
             "grid size.",
    )
    parser.add_argument(
        "--min_density", type=int, default=3,
        help="Minimum pixel brightness for wall detection (default: 3)",
    )
    parser.add_argument(
        "--invert", action="store_true", default=False,
        help="Invert the image (use when walls are dark on light background)",
    )

    # Point-cloud-only options
    pc_group = parser.add_argument_group("Point cloud options")
    pc_group.add_argument(
        "--slice_height", type=float, default=None,
        help="Z height for the horizontal slice in metres (default: auto)",
    )
    pc_group.add_argument(
        "--thickness", type=float, default=0.3,
        help="Slab thickness in metres (default: 0.3)",
    )
    pc_group.add_argument(
        "--no_image", action="store_true", default=False,
        help="Skip saving the orthoimage PNG",
    )

    args = parser.parse_args()

    # Auto-detect mode based on file extension
    ext = os.path.splitext(args.input)[1].lower()
    is_image = ext in _IMAGE_EXTENSIONS

    if is_image:
        print(f"Image mode: reading {ext} file directly")
        result = run_pipeline_from_image(
            args.input,
            output_dxf=args.output,
            resolution=args.resolution,
            min_density=args.min_density,
            invert=args.invert,
        )
    else:
        print(f"Point cloud mode: reading {ext} file")
        result = run_pipeline(
            args.input,
            output_dir=args.output_dir,
            output_dxf=args.output,
            slice_height=args.slice_height,
            thickness=args.thickness,
            resolution=args.resolution,
            min_density=args.min_density,
            save_image=not args.no_image,
        )

    if result:
        print(f"\nDone! Floor plan saved to: {result}")
    else:
        print("\nPipeline failed. Check the errors above.")

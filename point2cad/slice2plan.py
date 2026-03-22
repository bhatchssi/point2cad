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
# 4. Wall detection via image processing (scipy only)
# ---------------------------------------------------------------------------

def detect_walls(image, origin, resolution, min_density=5,
                 morph_iterations=2, min_line_length=0.3):
    """Detect wall segments from the density image.

    Uses morphological operations and connected-component labeling to find
    wall regions, then fits line segments along each wall cluster.

    Args:
        image: 2D uint8 density image from rasterize().
        origin: (x_min, y_min) world coordinate origin.
        resolution: Grid cell size in metres.
        min_density: Minimum pixel value to consider as "wall".
        morph_iterations: Iterations of morphological closing to fill gaps.
        min_line_length: Minimum wall segment length in metres.

    Returns:
        List of ((x1, y1), (x2, y2)) line segments in world coordinates.
    """
    from scipy import ndimage

    # Threshold to binary
    binary = (image >= min_density).astype(np.uint8)

    # Morphological closing to connect nearby wall points
    struct = ndimage.generate_binary_structure(2, 2)  # 8-connected
    binary = ndimage.binary_closing(binary, structure=struct,
                                     iterations=morph_iterations).astype(np.uint8)

    # Thin to skeleton (approximate by erosion-based thinning)
    skeleton = _skeletonize(binary)

    # Label connected components in the skeleton
    labeled, n_features = ndimage.label(skeleton, structure=struct)
    print(f"  Found {n_features} wall segments in skeleton")

    # For each connected component, fit a line segment
    segments = []
    min_pixels = max(3, int(min_line_length / resolution))

    for label_id in range(1, n_features + 1):
        ys, xs = np.where(labeled == label_id)
        if len(xs) < min_pixels:
            continue

        # Convert pixel coords to world coords
        wx = xs * resolution + origin[0]
        wy = ys * resolution + origin[1]

        # Fit line segments to this cluster
        cluster_segments = _fit_line_segments(wx, wy, resolution,
                                              min_line_length)
        segments.extend(cluster_segments)

    print(f"  Traced {len(segments)} wall line segments")
    return segments


def _skeletonize(binary):
    """Simple morphological skeletonization using scipy.

    Repeatedly erodes the image and accumulates the skeleton pixels that
    would be removed by further erosion.
    """
    from scipy import ndimage

    skel = np.zeros_like(binary)
    element = ndimage.generate_binary_structure(2, 2)
    img = binary.copy()

    while img.any():
        eroded = ndimage.binary_erosion(img, element)
        opened = ndimage.binary_dilation(eroded, element)
        # Pixels in img but not in opened are skeleton pixels
        skel |= (img & ~opened)
        img = eroded.astype(np.uint8)

    return skel


def _fit_line_segments(wx, wy, resolution, min_length):
    """Fit line segments to a cluster of wall points.

    Uses PCA to find the principal direction, then projects points onto it
    to get the extent. For L-shaped or curved walls, splits the cluster
    into sub-segments.

    Args:
        wx, wy: World-coordinate arrays of the cluster points.
        resolution: Grid resolution for splitting threshold.
        min_length: Minimum segment length.

    Returns:
        List of ((x1, y1), (x2, y2)) segments.
    """
    points = np.column_stack([wx, wy])

    # If the cluster is small enough, fit a single line
    if len(points) < 6:
        return _fit_single_segment(points, min_length)

    # PCA to find principal direction
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Principal direction = eigenvector with largest eigenvalue
    principal = eigvecs[:, 1]  # eigh returns sorted ascending

    # Project onto principal axis
    projections = centered @ principal
    p_min, p_max = projections.min(), projections.max()
    length = p_max - p_min

    if length < min_length:
        return []

    # Check "thickness" along secondary axis — if thick, might be L-shaped
    secondary = eigvecs[:, 0]
    sec_proj = centered @ secondary
    thickness = sec_proj.max() - sec_proj.min()

    # If width/length ratio is high, try splitting
    if thickness > length * 0.4 and len(points) > 20:
        return _split_and_fit(points, resolution, min_length)

    # Single line segment: endpoints from projection extremes
    p1 = centroid + principal * p_min
    p2 = centroid + principal * p_max
    return [((p1[0], p1[1]), (p2[0], p2[1]))]


def _fit_single_segment(points, min_length):
    """Fit a single line segment to a small set of points."""
    if len(points) < 2:
        return []

    # Use the two most distant points
    from scipy.spatial.distance import pdist, squareform
    if len(points) <= 50:
        dists = squareform(pdist(points))
        i, j = np.unravel_index(dists.argmax(), dists.shape)
    else:
        i, j = 0, len(points) - 1

    p1, p2 = points[i], points[j]
    if np.linalg.norm(p2 - p1) < min_length:
        return []
    return [((p1[0], p1[1]), (p2[0], p2[1]))]


def _split_and_fit(points, resolution, min_length, depth=0):
    """Recursively split a wide cluster and fit segments to sub-clusters."""
    if depth > 3 or len(points) < 6:
        return _fit_single_segment(points, min_length)

    # K-means with k=2 to split the cluster
    from scipy.cluster.vq import kmeans2
    try:
        centroids, labels = kmeans2(points.astype(np.float64), 2, minit='points')
    except Exception:
        return _fit_single_segment(points, min_length)

    segments = []
    for k in range(2):
        sub = points[labels == k]
        if len(sub) >= 3:
            segments.extend(
                _fit_line_segments(sub[:, 0], sub[:, 1], resolution, min_length)
            )
    return segments


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
# 7. Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(input_path, output_dir=None, output_dxf=None,
                 slice_height=None, thickness=0.3, resolution=0.02,
                 min_density=5, save_image=True):
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Slice a point cloud and generate a 2D floor plan (DXF)"
    )
    parser.add_argument(
        "input", type=str,
        help="Input point cloud file (LAZ, LAS, PLY, E57, PTS, PTX, XYZ)",
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
        "--slice_height", type=float, default=None,
        help="Z height for the horizontal slice in metres (default: auto-detect)",
    )
    parser.add_argument(
        "--thickness", type=float, default=0.3,
        help="Slab thickness in metres (default: 0.3)",
    )
    parser.add_argument(
        "--resolution", type=float, default=0.02,
        help="Raster resolution in metres/pixel (default: 0.02 = 2cm)",
    )
    parser.add_argument(
        "--min_density", type=int, default=5,
        help="Minimum raster pixel density for wall detection (default: 5)",
    )
    parser.add_argument(
        "--no_image", action="store_true", default=False,
        help="Skip saving the orthoimage PNG",
    )

    args = parser.parse_args()

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

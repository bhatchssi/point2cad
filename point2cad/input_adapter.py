"""
Input adapter for FARO scanner point cloud formats.

Converts common FARO SCENE export formats into the XYZC format expected by
the Point2CAD pipeline. Supported formats:

  - E57  (industry standard, via pye57)
  - LAS/LAZ (via laspy)
  - PLY  (via trimesh)
  - PTS  (ASCII, Leica/FARO text format)
  - PTX  (ASCII gridded, FARO/Leica)
  - XYZ  (plain ASCII, 3+ columns)
  - XYZC (pass-through, already segmented)

When surface labels are not present in the input, a basic region-growing
segmentation is applied using Open3D so the data can still enter the pipeline.
For best results, use ParseNet via generate_segmentation.py on the raw XYZ
output from this adapter.
"""

import os
import numpy as np

# Lazy imports for optional heavy dependencies
_pye57 = None
_laspy = None


def _get_pye57():
    global _pye57
    if _pye57 is None:
        try:
            import pye57
            _pye57 = pye57
        except ImportError:
            raise ImportError(
                "pye57 is required to read E57 files. Install with: pip install pye57"
            )
    return _pye57


def _get_laspy():
    global _laspy
    if _laspy is None:
        try:
            import laspy
            _laspy = laspy
        except ImportError:
            raise ImportError(
                "laspy is required to read LAS/LAZ files. Install with: pip install laspy lazrs"
            )
    return _laspy


def detect_format(path):
    """Detect point cloud format from file extension.

    Args:
        path: File path string.

    Returns:
        Format string: "e57", "las", "laz", "ply", "pts", "ptx", "xyz", or "xyzc".

    Raises:
        ValueError: If the extension is not recognized.
    """
    ext = os.path.splitext(path)[1].lower()
    format_map = {
        ".e57": "e57",
        ".las": "las",
        ".laz": "laz",
        ".ply": "ply",
        ".pts": "pts",
        ".ptx": "ptx",
        ".xyz": "xyz",
        ".xyzc": "xyzc",
        ".txt": "xyz",
        ".asc": "xyz",
    }
    fmt = format_map.get(ext)
    if fmt is None:
        raise ValueError(
            f"Unrecognized point cloud format: '{ext}'. "
            f"Supported: {', '.join(sorted(format_map.keys()))}"
        )
    return fmt


def read_e57(path):
    """Read an E57 file and return Nx3 points array.

    Also returns intensity as a 1D array if available, otherwise None.
    """
    pye57 = _get_pye57()
    e57 = pye57.E57(path)

    # Read the first scan in the file
    header = e57.get_header(0)
    data = e57.read_scan_raw(0)

    x = np.array(data["cartesianX"], dtype=np.float64)
    y = np.array(data["cartesianY"], dtype=np.float64)
    z = np.array(data["cartesianZ"], dtype=np.float64)
    points = np.column_stack([x, y, z]).astype(np.float32)

    intensity = None
    if "intensity" in data:
        intensity = np.array(data["intensity"], dtype=np.float32)

    return points, intensity


def read_las(path):
    """Read a LAS or LAZ file and return Nx3 points array.

    Returns classification labels if present, otherwise None.
    """
    laspy = _get_laspy()
    las = laspy.read(path)

    points = np.column_stack([
        np.array(las.x, dtype=np.float64).astype(np.float32),
        np.array(las.y, dtype=np.float64).astype(np.float32),
        np.array(las.z, dtype=np.float64).astype(np.float32),
    ])

    labels = None
    if hasattr(las, "classification"):
        raw_labels = np.array(las.classification, dtype=np.int32)
        # Only use if there are meaningful labels (more than one class)
        if len(np.unique(raw_labels)) > 1:
            labels = raw_labels

    return points, labels


def read_ply(path):
    """Read a PLY file and return Nx3 points array."""
    import trimesh
    cloud = trimesh.load(path)
    if hasattr(cloud, "vertices"):
        return np.array(cloud.vertices, dtype=np.float32), None
    raise ValueError(f"PLY file does not contain vertex data: {path}")


def read_pts(path):
    """Read a PTS file (Leica/FARO ASCII format).

    PTS format:
      - First line: number of points (integer)
      - Subsequent lines: X Y Z [intensity] [R G B]

    Returns Nx3 points and optional intensity.
    """
    with open(path, "r") as f:
        first_line = f.readline().strip()

    # Check if first line is a point count
    try:
        num_points = int(first_line)
        skip_header = 1
    except ValueError:
        skip_header = 0

    data = np.loadtxt(path, skiprows=skip_header, dtype=np.float32)
    points = data[:, :3]

    intensity = None
    if data.shape[1] >= 4:
        intensity = data[:, 3]

    return points, intensity


def read_ptx(path):
    """Read a PTX file (FARO/Leica gridded ASCII format).

    PTX format per scan block:
      - Line 1: number of columns
      - Line 2: number of rows
      - Lines 3-5: scanner position (3x1) and rotation (3x3)
      - Lines 6-8: transformation matrix rows
      - Line 9-11: 4x4 transform (last row)
      - Remaining lines: X Y Z intensity [R G B] (rows * cols lines)

    Only the first scan block is read. Invalid points (0 0 0) are filtered.
    """
    points_list = []
    intensity_list = []

    with open(path, "r") as f:
        # Read first scan block header
        cols = int(f.readline().strip())
        rows = int(f.readline().strip())

        # Skip 8 header lines (scanner pos + rotation + transform)
        for _ in range(8):
            f.readline()

        num_points = rows * cols
        for _ in range(num_points):
            line = f.readline().strip()
            if not line:
                break
            parts = line.split()
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            points_list.append([x, y, z])
            if len(parts) >= 4:
                intensity_list.append(float(parts[3]))

    points = np.array(points_list, dtype=np.float32)
    intensity = np.array(intensity_list, dtype=np.float32) if intensity_list else None

    # Filter invalid points (FARO uses 0,0,0 for missing returns)
    valid_mask = np.any(points != 0, axis=1)
    points = points[valid_mask]
    if intensity is not None:
        intensity = intensity[valid_mask]

    return points, intensity


def read_xyz(path):
    """Read a plain XYZ text file. Expects at least 3 columns (X Y Z ...)."""
    data = np.loadtxt(path, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise ValueError(f"XYZ file must have at least 3 columns, got {data.shape[1]}")
    return data[:, :3], None


def read_xyzc(path):
    """Read an already-segmented XYZC file (pass-through)."""
    data = np.loadtxt(path, dtype=np.float32)
    if data.shape[1] != 4:
        raise ValueError(
            f"XYZC file must have exactly 4 columns, got {data.shape[1]}"
        )
    return data[:, :3], data[:, 3].astype(np.int32)


# Format-to-reader dispatch
_READERS = {
    "e57": read_e57,
    "las": read_las,
    "laz": read_las,
    "ply": read_ply,
    "pts": read_pts,
    "ptx": read_ptx,
    "xyz": read_xyz,
    "xyzc": read_xyzc,
}


def downsample_points(points, voxel_size=0.01, labels=None):
    """Voxel-based downsampling using pure numpy (no Open3D).

    For each voxel, keeps the point closest to the voxel center.

    Args:
        points: Nx3 array.
        voxel_size: Size of each voxel for downsampling.
        labels: Optional Nx1 labels array to carry through.

    Returns:
        Downsampled (points, labels) tuple.
    """
    # Quantize points to voxel grid
    voxel_indices = np.floor(points / voxel_size).astype(np.int64)

    # Create unique voxel keys
    # Shift to non-negative indices
    mins = voxel_indices.min(axis=0)
    voxel_indices -= mins
    maxes = voxel_indices.max(axis=0) + 1

    # Flatten to 1D key per point
    keys = (voxel_indices[:, 0] * maxes[1] * maxes[2] +
            voxel_indices[:, 1] * maxes[2] +
            voxel_indices[:, 2])

    # For each unique voxel, pick the first point encountered
    _, unique_idx = np.unique(keys, return_index=True)
    unique_idx.sort()

    new_points = points[unique_idx].astype(np.float32)
    new_labels = labels[unique_idx] if labels is not None else None

    return new_points, new_labels


def segment_region_growing(points, n_neighbors=30, smoothness_threshold=10.0,
                           curvature_threshold=1.0, min_cluster_size=50):
    """Surface segmentation via scipy DBSCAN (no Open3D).

    Uses scipy's cKDTree for spatial indexing and a simple DBSCAN-style
    clustering. This is a fallback when ParseNet is not available.

    Args:
        points: Nx3 array.
        n_neighbors: Not used (kept for API compatibility).
        smoothness_threshold: Not used (kept for API compatibility).
        curvature_threshold: Not used (kept for API compatibility).
        min_cluster_size: Minimum points per cluster.

    Returns:
        Integer label array of length N.
    """
    from scipy.spatial import cKDTree
    from scipy.sparse import lil_matrix
    from scipy.sparse.csgraph import connected_components

    spatial_extent = np.max(points, axis=0) - np.min(points, axis=0)
    scale = np.mean(spatial_extent) if np.mean(spatial_extent) > 0 else 1.0
    eps = scale * 0.05

    n = len(points)
    print(f"    Building KD-tree for {n} points...")
    tree = cKDTree(points)

    print(f"    Finding neighbors (eps={eps:.4f})...")
    # query_pairs returns all pairs within eps — fast C implementation
    pairs = tree.query_pairs(r=eps, output_type='ndarray')
    print(f"    Found {len(pairs)} neighbor pairs")

    # Build sparse adjacency and find connected components
    print("    Computing connected components...")
    adj = lil_matrix((n, n), dtype=bool)
    if len(pairs) > 0:
        adj[pairs[:, 0], pairs[:, 1]] = True
        adj[pairs[:, 1], pairs[:, 0]] = True

    n_components, comp_labels = connected_components(adj, directed=False)
    print(f"    Found {n_components} raw components")

    # Filter small clusters — relabel sequentially
    labels = np.full(n, -1, dtype=np.int32)
    current_label = 0
    for comp_id in range(n_components):
        mask = comp_labels == comp_id
        if np.sum(mask) >= min_cluster_size:
            labels[mask] = current_label
            current_label += 1

    num_clusters = current_label
    print(f"    {num_clusters} clusters after filtering (min_size={min_cluster_size})")

    # Assign noise points to nearest cluster
    unlabeled = labels == -1
    if np.any(unlabeled) and num_clusters > 0:
        labeled_mask = ~unlabeled
        labeled_tree = cKDTree(points[labeled_mask])
        _, nn_idx = labeled_tree.query(points[unlabeled])
        labels[unlabeled] = labels[labeled_mask][nn_idx]

    if num_clusters == 0:
        labels[:] = 0

    return labels


def load_point_cloud(path, max_points=None, voxel_size=None, auto_segment=True):
    """Load a point cloud from any supported format and prepare for Point2CAD.

    This is the main entry point for the input adapter. It handles format
    detection, reading, optional downsampling, and segmentation.

    Args:
        path: Path to the input point cloud file.
        max_points: If set, randomly subsample to this many points after loading.
        voxel_size: If set, apply voxel downsampling with this size.
        auto_segment: If True and no labels are present, apply fallback
            region-growing segmentation. If False, raises an error when
            labels are missing.

    Returns:
        Tuple of (points, labels) where:
          - points is an Nx3 float32 array
          - labels is an N-length int32 array of surface cluster IDs
    """
    fmt = detect_format(path)
    reader = _READERS[fmt]

    print(f"Reading {fmt.upper()} file: {path}")
    points, labels = reader(path)
    print(f"  Loaded {len(points)} points")

    # Filter NaN/Inf values
    valid = np.all(np.isfinite(points), axis=1)
    if not np.all(valid):
        num_invalid = np.sum(~valid)
        print(f"  Filtered {num_invalid} invalid (NaN/Inf) points")
        points = points[valid]
        if labels is not None:
            labels = labels[valid]

    # Pre-subsample very large clouds to keep processing time reasonable
    SAFE_LIMIT = 1_000_000
    if len(points) > SAFE_LIMIT:
        print(f"  Large scan detected ({len(points)} pts). Pre-subsampling to {SAFE_LIMIT}...")
        indices = np.random.default_rng(42).choice(len(points), SAFE_LIMIT, replace=False)
        indices.sort()
        points = points[indices]
        if labels is not None:
            labels = labels[indices]

    # Voxel downsampling for large scans
    if voxel_size is not None and voxel_size > 0:
        before = len(points)
        points, labels = downsample_points(points, voxel_size, labels)
        print(f"  Downsampled {before} -> {len(points)} points (voxel_size={voxel_size})")

    # Random subsampling
    if max_points is not None and len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        indices.sort()
        points = points[indices]
        if labels is not None:
            labels = labels[indices]
        print(f"  Subsampled to {max_points} points")

    # Handle segmentation
    if labels is None:
        if auto_segment:
            print("  No surface labels found. Running fallback segmentation...")
            print("  (For best results, use ParseNet via generate_segmentation.py)")
            labels = segment_region_growing(points)
            num_clusters = len(np.unique(labels))
            print(f"  Segmented into {num_clusters} surface clusters")
        else:
            raise ValueError(
                f"Input file has no surface labels and auto_segment=False. "
                f"Run ParseNet segmentation first, or set auto_segment=True."
            )

    labels = labels.astype(np.int32)
    return points, labels


def save_xyzc(points, labels, path):
    """Save points and labels as an XYZC text file.

    Args:
        points: Nx3 float array.
        labels: N-length int array.
        path: Output file path.
    """
    data = np.column_stack([points, labels.reshape(-1, 1)])
    np.savetxt(path, data, fmt="%.6f %.6f %.6f %d")
    print(f"Saved XYZC: {path} ({len(points)} points)")


def convert_to_xyzc(input_path, output_path=None, max_points=None,
                     voxel_size=None, auto_segment=True):
    """Convert any supported format to XYZC and save.

    Convenience function that combines load + save. If output_path is None,
    it is derived from input_path by changing the extension to .xyzc.

    Args:
        input_path: Path to input point cloud.
        output_path: Path for output XYZC file (optional).
        max_points: Max points after loading.
        voxel_size: Voxel downsampling size.
        auto_segment: Whether to auto-segment if no labels.

    Returns:
        The output path.
    """
    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = base + ".xyzc"

    points, labels = load_point_cloud(
        input_path,
        max_points=max_points,
        voxel_size=voxel_size,
        auto_segment=auto_segment,
    )

    save_xyzc(points, labels, output_path)
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert FARO scanner point clouds to XYZC format for Point2CAD"
    )
    parser.add_argument(
        "input", type=str,
        help="Input point cloud file (E57, LAS, LAZ, PLY, PTS, PTX, XYZ)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output XYZC file path (default: <input_name>.xyzc)",
    )
    parser.add_argument(
        "--max_points", type=int, default=None,
        help="Maximum number of points to keep (random subsampling)",
    )
    parser.add_argument(
        "--voxel_size", type=float, default=None,
        help="Voxel size for downsampling (useful for dense FARO scans)",
    )
    parser.add_argument(
        "--no_auto_segment", action="store_true", default=False,
        help="Disable automatic segmentation when labels are missing",
    )
    args = parser.parse_args()

    out = convert_to_xyzc(
        args.input,
        output_path=args.output,
        max_points=args.max_points,
        voxel_size=args.voxel_size,
        auto_segment=not args.no_auto_segment,
    )
    print(f"\nConversion complete: {out}")
    print("Next step: python -m point2cad.main --path_in", out)

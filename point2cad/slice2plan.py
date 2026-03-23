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

def _otsu_threshold(image):
    """Compute Otsu's optimal threshold for a grayscale image.

    Finds the threshold that minimizes intra-class variance between
    foreground (walls) and background (everything else). Only considers
    non-zero pixels so the vast black background doesn't skew the result.

    Returns:
        Optimal threshold value (int, 0-255).
    """
    # Only consider non-zero pixels
    nonzero = image[image > 0].ravel()
    if len(nonzero) == 0:
        return 128

    # Build histogram of non-zero values
    hist = np.zeros(256, dtype=np.float64)
    for v in nonzero:
        hist[v] += 1
    hist /= hist.sum()

    # Otsu's method: maximize between-class variance
    best_t = 0
    best_var = 0.0
    w0 = 0.0  # weight of background class
    mu0 = 0.0  # mean of background class
    mu_total = np.sum(np.arange(256) * hist)

    for t in range(1, 256):
        w0 += hist[t - 1]
        if w0 == 0:
            continue
        w1 = 1.0 - w0
        if w1 == 0:
            break
        mu0 += (t - 1) * hist[t - 1]
        mu1 = (mu_total - mu0) / w1
        mu0_norm = mu0 / w0

        var_between = w0 * w1 * (mu0_norm - mu1) ** 2
        if var_between > best_var:
            best_var = var_between
            best_t = t

    return best_t


def detect_walls(image, origin, resolution, min_density=3,
                 min_line_length=0.3, simplify_tolerance=None,
                 morph_iterations=None):
    """Detect wall segments from a scanner orthoimage.

    Returns both centreline segments and their measured thicknesses so that
    the DXF exporter can draw double-line walls (both faces).

    Args:
        image: 2D uint8 density image (bright = walls, dark = empty).
        origin: (x_min, y_min) world coordinate origin.
        resolution: Grid cell size in metres.
        min_density: Minimum pixel value to consider as wall.
        min_line_length: Minimum wall segment length in metres.
        simplify_tolerance: RDP tolerance in metres (default: 3*resolution).
        morph_iterations: Unused, kept for CLI compatibility.

    Returns:
        List of ((x1,y1), (x2,y2), thickness) tuples in world coordinates.
        thickness is the measured wall width in metres.
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
    # Use Otsu's method on non-zero pixels to separate wall signal from
    # background noise. Don't force the threshold too high — scanner
    # orthoimages have walls at varying brightness depending on scan
    # density and angle. A too-high threshold shatters walls into pieces.
    threshold = _otsu_threshold(stretched)
    # Only enforce a modest floor — the contrast stretch already
    # normalised the range, so Otsu on non-zero pixels is reliable.
    threshold = max(threshold, min_density, 30)
    print(f"  Threshold: {threshold} (walls must be >= this brightness)")
    binary = (stretched >= threshold).astype(np.uint8)
    n_wall_px = binary.sum()
    print(f"  Binary mask: {n_wall_px:,} wall pixels "
          f"({100*n_wall_px/binary.size:.1f}% of image)")

    if n_wall_px == 0:
        print("  WARNING: No wall pixels found. Try lowering --min_density.")
        return []

    # --- 3. Morphological cleanup ---
    struct8 = ndimage.generate_binary_structure(2, 2)  # 8-connected
    struct4 = ndimage.generate_binary_structure(2, 1)  # 4-connected

    # Close gaps in wall lines — use 8-connected structuring element and
    # 2 iterations to bridge gaps up to ~2px (caused by threshold cutting
    # through thin or dim wall sections).
    binary = ndimage.binary_closing(
        binary, structure=struct8, iterations=2
    ).astype(np.uint8)

    # Gentle dilation to reconnect wall fragments that are 1px apart
    binary = ndimage.binary_dilation(
        binary, structure=struct4, iterations=1
    ).astype(np.uint8)

    # Remove small isolated noise blobs
    labeled_noise, n_noise = ndimage.label(binary, structure=struct8)
    if n_noise > 0:
        sizes = ndimage.sum(binary, labeled_noise, range(1, n_noise + 1))
        min_blob = max(10, int(0.15 / resolution))  # 15cm minimum blob
        for i, sz in enumerate(sizes):
            if sz < min_blob:
                binary[labeled_noise == (i + 1)] = 0
        n_removed = sum(1 for s in sizes if s < min_blob)
        if n_removed > 0:
            print(f"  Removed {n_removed} noise blobs (< {min_blob}px)")

    # --- 4. Distance transform (for wall thickness measurement) ---
    # The distance transform gives the distance from each wall pixel to
    # the nearest background pixel. At the skeleton (centreline), this
    # equals half the wall thickness.
    dist_transform = ndimage.distance_transform_edt(binary)

    # --- 5. Skeletonize to 1px centrelines ---
    skeleton = _skeletonize_proper(binary)
    n_skel = skeleton.sum()
    print(f"  Skeleton: {n_skel:,} pixels")

    if n_skel == 0:
        return []

    # Measure half-thickness at each skeleton pixel
    skel_thickness = dist_transform * skeleton  # zero except on skeleton

    # --- 6. Chain skeleton pixels into polylines ---
    polylines = _chain_skeleton(skeleton)
    print(f"  Chained into {len(polylines)} polylines")

    # --- 7. Convert to world coords, simplify, merge collinear ---
    min_px_len = max(3, int(min_line_length / resolution))
    # Use a shorter minimum for small features (window casings, etc.)
    min_short = min_line_length * 0.4
    all_segments = []  # will hold ((x1,y1),(x2,y2), thickness)

    for chain in polylines:
        if len(chain) < min_px_len:
            continue

        # Measure thickness along this chain (median half-width * 2)
        chain_rows = chain[:, 0].astype(np.intp)
        chain_cols = chain[:, 1].astype(np.intp)
        half_widths = skel_thickness[chain_rows, chain_cols]
        # Filter out zeros (shouldn't happen on skeleton, but be safe)
        valid = half_widths[half_widths > 0]
        if len(valid) > 0:
            median_half = np.median(valid)
            chain_thickness = median_half * 2.0 * resolution  # in metres
        else:
            chain_thickness = resolution * 4  # fallback: ~2 pixels wide

        # Convert (row, col) to world (x, y)
        world_pts = np.empty((len(chain), 2), dtype=np.float64)
        world_pts[:, 0] = chain[:, 1] * resolution + origin[0]  # col -> x
        world_pts[:, 1] = chain[:, 0] * resolution + origin[1]  # row -> y

        simplified = _rdp_simplify(world_pts, simplify_tolerance)
        if len(simplified) < 2:
            continue

        # Build raw segments from this polyline
        raw_segs = []
        for i in range(len(simplified) - 1):
            p1, p2 = simplified[i], simplified[i + 1]
            seg_len = np.linalg.norm(p2 - p1)
            if seg_len >= min_short:
                raw_segs.append((p1, p2))

        # Merge consecutive collinear segments into longer walls
        # Use generous tolerances to prevent walls from being fragmented
        merged = _merge_collinear(raw_segs, angle_tol=8.0,
                                   gap_tol=resolution * 8)

        for p1, p2 in merged:
            seg_len = np.linalg.norm(p2 - p1)
            if seg_len >= min_short:
                all_segments.append(
                    ((p1[0], p1[1]), (p2[0], p2[1]), chain_thickness)
                )

    # --- 8. Orthogonal snapping ---
    # Extract just the segment pairs for snapping, then reattach thickness
    seg_pairs = [(s[0], s[1]) for s in all_segments]
    thicknesses = [s[2] for s in all_segments]
    snapped_pairs = _snap_to_orthogonal(seg_pairs, resolution)

    # Reattach thickness (snapping may have removed degenerate segments)
    # Build a mapping from original to snapped
    if len(snapped_pairs) == len(seg_pairs):
        all_segments = [(p1, p2, t)
                        for (p1, p2), t in zip(snapped_pairs, thicknesses)]
    else:
        # Snapping removed some segments; use median thickness as fallback
        med_t = np.median(thicknesses) if thicknesses else resolution * 4
        all_segments = [(p1, p2, med_t) for p1, p2 in snapped_pairs]

    # Clamp wall thickness to reasonable bounds
    min_t = resolution * 2   # at least 2 pixels
    max_t = resolution * 30  # at most ~60cm at 0.02 res
    all_segments = [(p1, p2, max(min_t, min(t, max_t)))
                    for p1, p2, t in all_segments]

    # --- 9. Global collinear merge ---
    # After snapping, segments from different polyline chains that ended
    # up on the same axis-aligned line should be merged into longer walls.
    all_segments = _merge_global_collinear(all_segments, resolution)

    # Report typical wall thickness
    if all_segments:
        ts = [s[2] for s in all_segments]
        print(f"  Wall thickness: median {np.median(ts)*100:.0f}cm, "
              f"range {min(ts)*100:.0f}-{max(ts)*100:.0f}cm")

    print(f"  Final: {len(all_segments)} wall segments")
    return all_segments


def _merge_global_collinear(segments, resolution, angle_tol=3.0,
                             lateral_tol=None, gap_tol=None):
    """Merge collinear segments across different polyline chains.

    After orthogonal snapping, many segments from separate chains may lie
    on the same line. This groups segments by direction and lateral offset,
    then merges overlapping/nearby segments within each group.

    Args:
        segments: List of ((x1,y1),(x2,y2), thickness) tuples.
        resolution: Grid resolution (metres).
        angle_tol: Max angle difference to consider same direction (degrees).
        lateral_tol: Max perpendicular distance to consider same line (metres).
            Default: 3 * resolution.
        gap_tol: Max gap along the line to bridge when merging (metres).
            Default: 10 * resolution.

    Returns:
        New list of merged ((x1,y1),(x2,y2), thickness) tuples.
    """
    if len(segments) < 2:
        return segments

    if lateral_tol is None:
        lateral_tol = resolution * 3
    if gap_tol is None:
        gap_tol = resolution * 10

    # Group segments by angle (quantised to nearest axis)
    # For each segment, compute angle in [0, 180) and a lateral offset
    # (signed distance from origin along the perpendicular direction).
    entries = []  # (angle, lateral, t_min, t_max, thickness, index)
    for i, seg in enumerate(segments):
        (x1, y1), (x2, y2) = seg[0], seg[1]
        thickness = seg[2]
        dx, dy = x2 - x1, y2 - y1
        angle = np.degrees(np.arctan2(dy, dx)) % 180.0
        length = np.sqrt(dx*dx + dy*dy)
        if length < 1e-12:
            continue

        # Unit direction and perpendicular
        ux, uy = dx / length, dy / length
        # Lateral offset = perpendicular distance from origin
        # For a line through (x1,y1) with direction (ux,uy),
        # lateral = x1 * (-uy) + y1 * ux
        lateral = x1 * (-uy) + y1 * ux

        # Project endpoints along the line direction
        t1 = x1 * ux + y1 * uy
        t2 = x2 * ux + y2 * uy
        t_min, t_max = min(t1, t2), max(t1, t2)

        entries.append((angle, lateral, t_min, t_max, thickness, ux, uy))

    if not entries:
        return segments

    # Sort by angle, then lateral offset
    entries.sort(key=lambda e: (round(e[0] / angle_tol) * angle_tol, e[1]))

    # Greedy merge: walk through sorted entries, merge compatible ones
    merged = []
    used = [False] * len(entries)

    for i in range(len(entries)):
        if used[i]:
            continue
        a_i, lat_i, tmin_i, tmax_i, thick_i, ux_i, uy_i = entries[i]
        used[i] = True

        # Find all compatible segments
        for j in range(i + 1, len(entries)):
            if used[j]:
                continue
            a_j, lat_j, tmin_j, tmax_j, thick_j, ux_j, uy_j = entries[j]

            # Check angle compatibility
            if _angle_distance(a_i, a_j) > angle_tol:
                # Since sorted by angle, no more matches in this group
                # (but there could be wraparound, so don't break)
                continue

            # Check lateral offset (are they on the same line?)
            if abs(lat_i - lat_j) > lateral_tol:
                continue

            # Check gap along the line
            gap = max(0, max(tmin_j - tmax_i, tmin_i - tmax_j))
            if gap > gap_tol:
                continue

            # Merge: extend the span
            tmin_i = min(tmin_i, tmin_j)
            tmax_i = max(tmax_i, tmax_j)
            thick_i = max(thick_i, thick_j)  # use thicker measurement
            used[j] = True

        # Reconstruct segment from merged span
        x1 = ux_i * tmin_i - (-uy_i) * lat_i
        y1 = uy_i * tmin_i - ux_i * (-lat_i)
        # Simpler: use parametric form
        # Point on line = t * (ux, uy) + lateral * (-uy, ux) ... wait
        # lateral = x*(-uy) + y*ux, so the line is:
        # x = t*ux + lateral*(-uy) ... no. Let me think.
        # Actually: given lateral = -uy*x + ux*y, and t = ux*x + uy*y
        # Then: x = ux*t - uy*lateral, y = uy*t + ux*lateral
        # (inverse of the rotation)
        p1x = ux_i * tmin_i - uy_i * lat_i
        p1y = uy_i * tmin_i + ux_i * lat_i
        p2x = ux_i * tmax_i - uy_i * lat_i
        p2y = uy_i * tmax_i + ux_i * lat_i

        merged.append(((p1x, p1y), (p2x, p2y), thick_i))

    n_before = len(segments)
    n_after = len(merged)
    if n_before > n_after:
        print(f"  Global merge: {n_before} -> {n_after} segments "
              f"({n_before - n_after} merged)")

    return merged


def _merge_collinear(segments, angle_tol=5.0, gap_tol=0.1):
    """Merge consecutive nearly-collinear segments into longer walls.

    Walks the segment list and greedily fuses segments whose direction
    differs by less than angle_tol degrees. This prevents straight walls
    from being broken into many short pieces by pixel-level jitter.

    Args:
        segments: List of (p1, p2) numpy array pairs (ordered from polyline).
        angle_tol: Maximum angle difference (degrees) to consider collinear.
        gap_tol: Maximum gap between segment endpoints to allow merging.

    Returns:
        New list of (p1, p2) segments with collinear runs merged.
    """
    if len(segments) <= 1:
        return segments

    def _seg_angle(p1, p2):
        d = p2 - p1
        return np.degrees(np.arctan2(d[1], d[0])) % 180.0

    merged = []
    # Start accumulating from the first segment
    cur_start = segments[0][0]
    cur_end = segments[0][1]
    cur_angle = _seg_angle(cur_start, cur_end)

    for i in range(1, len(segments)):
        seg_start, seg_end = segments[i]
        seg_angle = _seg_angle(seg_start, seg_end)

        # Angle difference (handle wraparound at 180)
        angle_diff = abs(cur_angle - seg_angle)
        if angle_diff > 90:
            angle_diff = 180 - angle_diff

        # Gap between current end and next segment start
        gap = np.linalg.norm(seg_start - cur_end)

        if angle_diff <= angle_tol and gap <= gap_tol:
            # Extend current segment to the end of this one
            cur_end = seg_end
            # Update running angle using the full merged span
            cur_angle = _seg_angle(cur_start, cur_end)
        else:
            # Emit current merged segment, start a new one
            merged.append((cur_start, cur_end))
            cur_start = seg_start
            cur_end = seg_end
            cur_angle = seg_angle

    # Emit the last segment
    merged.append((cur_start, cur_end))
    return merged


def _snap_to_orthogonal(segments, resolution, long_threshold=1.0,
                         snap_tol_long=10.0, snap_tol_short=20.0):
    """Snap segments to the building's dominant orthogonal axes.

    Three-pass approach:
      1. **Axis detection**: find the dominant building angle with sub-degree
         precision using a length-weighted circular mean of long segments.
      2. **Angle snapping**: rotate each segment to the nearest cardinal
         direction (dominant or dominant+90°) around its midpoint.
      3. **Endpoint alignment**: snap nearby endpoints together so walls
         meet cleanly at corners and T-junctions. Parallel walls at the
         same offset get their endpoints aligned onto a common grid line.

    Args:
        segments: List of ((x1,y1),(x2,y2)) tuples.
        resolution: Grid resolution (metres) for context.
        long_threshold: Length above which a segment is "long" (metres).
        snap_tol_long: Max angle deviation to snap long segments (degrees).
        snap_tol_short: Max angle deviation to snap short segments (degrees).

    Returns:
        New list of snapped ((x1,y1),(x2,y2)) segments.
    """
    if len(segments) < 2:
        return segments

    # ------------------------------------------------------------------
    # Pass 1: Sub-degree dominant axis detection
    # ------------------------------------------------------------------
    # Compute angle in [0, 180) for each segment
    seg_data = []  # (angle_deg, length, midx, midy, (x1,y1), (x2,y2))
    for (x1, y1), (x2, y2) in segments:
        dx, dy = x2 - x1, y2 - y1
        angle = np.degrees(np.arctan2(dy, dx)) % 180.0
        length = np.sqrt(dx * dx + dy * dy)
        seg_data.append((angle, length, (x1+x2)/2, (y1+y2)/2,
                         (x1, y1), (x2, y2)))

    angles = np.array([s[0] for s in seg_data])
    lengths = np.array([s[1] for s in seg_data])

    # Use only long segments for axis detection — they're the most
    # reliable indicator of building orientation.
    long_mask = lengths >= long_threshold
    if long_mask.sum() < 3:
        long_mask = lengths >= np.percentile(lengths, 50)

    long_angles = angles[long_mask]
    long_lengths = lengths[long_mask]

    # Length-weighted circular mean in the angle-doubled domain
    # (doubling maps [0,180) to [0,360) so 0° and 179° are neighbours)
    theta2 = np.radians(long_angles * 2)
    wx = np.sum(long_lengths * np.cos(theta2))
    wy = np.sum(long_lengths * np.sin(theta2))
    dominant_angle = (np.degrees(np.arctan2(wy, wx)) / 2) % 180.0

    # Refine: weighted circular mean of segments within ±15° of the
    # coarse peak, giving sub-degree accuracy.
    near_mask = _angle_distance(long_angles, dominant_angle) < 15.0
    if near_mask.sum() >= 2:
        near_a = long_angles[near_mask]
        near_l = long_lengths[near_mask]
        theta2n = np.radians(near_a * 2)
        wx2 = np.sum(near_l * np.cos(theta2n))
        wy2 = np.sum(near_l * np.sin(theta2n))
        dominant_angle = (np.degrees(np.arctan2(wy2, wx2)) / 2) % 180.0

    secondary_angle = (dominant_angle + 90) % 180
    axis_angles = [dominant_angle, secondary_angle]

    # Stats
    near_dom = _angle_distance(angles, dominant_angle) < snap_tol_long
    near_sec = _angle_distance(angles, secondary_angle) < snap_tol_long
    pct = 100 * lengths[near_dom | near_sec].sum() / lengths.sum()
    print(f"  Dominant axis: {dominant_angle:.1f}° "
          f"({pct:.0f}% of wall length near axes), "
          f"secondary at {secondary_angle:.1f}°")

    # ------------------------------------------------------------------
    # Pass 2: Angle snapping — rotate segments to exact axis angles
    # ------------------------------------------------------------------
    snapped = []
    n_snapped = 0

    for seg_angle, seg_len, mid_x, mid_y, (x1, y1), (x2, y2) in seg_data:
        is_long = seg_len >= long_threshold
        tol = snap_tol_long if is_long else snap_tol_short

        # Find nearest axis
        best_axis = None
        best_diff = 999.0
        for ax in axis_angles:
            diff = _angle_distance(seg_angle, ax)
            if diff < best_diff:
                best_diff = diff
                best_axis = ax

        if best_diff <= tol:
            half_len = seg_len / 2.0
            # Preserve direction sense
            orig_rad = np.radians(seg_angle)
            ax_rad = np.radians(best_axis)
            dot = (np.cos(orig_rad) * np.cos(ax_rad) +
                   np.sin(orig_rad) * np.sin(ax_rad))
            if dot < 0:
                ax_rad += np.pi

            dx = np.cos(ax_rad) * half_len
            dy = np.sin(ax_rad) * half_len

            snapped.append((
                (mid_x - dx, mid_y - dy),
                (mid_x + dx, mid_y + dy),
            ))
            n_snapped += 1
        else:
            snapped.append(((x1, y1), (x2, y2)))

    print(f"  Snapped {n_snapped}/{len(segments)} segments to "
          f"{dominant_angle:.1f}°/{secondary_angle:.1f}° axes")

    # ------------------------------------------------------------------
    # Pass 3: Endpoint alignment — merge nearby endpoints so corners meet
    # ------------------------------------------------------------------
    snapped = _align_endpoints(snapped, snap_radius=resolution * 5)

    return snapped


def _angle_distance(a, b):
    """Shortest angular distance in [0,180) space. Works on scalars or arrays."""
    d = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    d = np.where(d > 90, 180 - d, d)
    return d


def _align_endpoints(segments, snap_radius=0.10):
    """Snap nearby endpoints to a common location so walls meet cleanly.

    Groups all segment endpoints that are within snap_radius of each other,
    replaces each group with the length-weighted centroid, then projects
    endpoints of axis-aligned segments onto a common grid line so parallel
    walls share exact coordinates.

    Args:
        segments: List of ((x1,y1),(x2,y2)) tuples (already angle-snapped).
        snap_radius: Maximum distance (metres) to merge endpoints.

    Returns:
        New list of segments with aligned endpoints.
    """
    if not segments:
        return segments

    # Collect all endpoints
    pts = []  # (x, y, seg_idx, end_idx)
    for i, ((x1, y1), (x2, y2)) in enumerate(segments):
        pts.append((x1, y1, i, 0))
        pts.append((x2, y2, i, 1))

    n_pts = len(pts)
    coords = np.array([(p[0], p[1]) for p in pts])

    # --- Greedy clustering of nearby endpoints ---
    # For each point, find all neighbours within snap_radius.
    # Use a simple O(n²) approach — fine for typical counts (<10k segments).
    assigned = np.full(n_pts, -1, dtype=int)
    clusters = []  # list of lists of point indices

    for i in range(n_pts):
        if assigned[i] >= 0:
            continue
        # Find all unassigned points within radius
        dists = np.sqrt((coords[:, 0] - coords[i, 0])**2 +
                        (coords[:, 1] - coords[i, 1])**2)
        mask = (dists <= snap_radius) & (assigned < 0)
        members = np.where(mask)[0].tolist()

        cluster_id = len(clusters)
        clusters.append(members)
        for m in members:
            assigned[m] = cluster_id

    # --- Compute cluster centroids (length-weighted) ---
    seg_lengths = np.array([
        np.sqrt((x2-x1)**2 + (y2-y1)**2)
        for (x1, y1), (x2, y2) in segments
    ])
    new_coords = coords.copy()

    n_merged = 0
    for cluster in clusters:
        if len(cluster) <= 1:
            continue
        # Weighted average — longer walls get more influence
        weights = np.array([seg_lengths[pts[j][2]] for j in cluster])
        total_w = weights.sum()
        if total_w < 1e-12:
            total_w = 1.0
        cx = sum(coords[j, 0] * weights[k] for k, j in enumerate(cluster)) / total_w
        cy = sum(coords[j, 1] * weights[k] for k, j in enumerate(cluster)) / total_w
        for j in cluster:
            new_coords[j] = (cx, cy)
        n_merged += len(cluster)

    if n_merged > 0:
        print(f"  Aligned {n_merged} endpoints in {len([c for c in clusters if len(c)>1])} groups")

    # Rebuild segments
    result = []
    seg_pts = {}  # seg_idx -> {0: (x,y), 1: (x,y)}
    for i in range(n_pts):
        seg_idx = pts[i][2]
        end_idx = pts[i][3]
        if seg_idx not in seg_pts:
            seg_pts[seg_idx] = {}
        seg_pts[seg_idx][end_idx] = (new_coords[i, 0], new_coords[i, 1])

    for i in range(len(segments)):
        p1 = seg_pts[i][0]
        p2 = seg_pts[i][1]
        # Skip degenerate segments (endpoints merged to same point)
        if np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2) > 0.01:
            result.append((p1, p2))

    return result


def _skeletonize_proper(binary):
    """Zhang-Suen thinning — fully vectorized with numpy.

    Produces clean 1px-wide skeletons that preserve connectivity,
    suitable for pixel chaining. Runs efficiently on large images
    (tested on 8000x20000+ scanner orthoimages).
    """
    img = np.pad(binary.astype(np.uint8), 1, mode='constant',
                 constant_values=0)
    changed = True
    iteration = 0

    while changed:
        changed = False
        for step in (0, 1):
            # Extract 8 neighbours using array slicing (P2..P9)
            # P2=N, P3=NE, P4=E, P5=SE, P6=S, P7=SW, P8=W, P9=NW
            p2 = img[:-2, 1:-1]   # north
            p3 = img[:-2, 2:]     # northeast
            p4 = img[1:-1, 2:]    # east
            p5 = img[2:, 2:]      # southeast
            p6 = img[2:, 1:-1]    # south
            p7 = img[2:, :-2]     # southwest
            p8 = img[1:-1, :-2]   # west
            p9 = img[:-2, :-2]    # northwest

            center = img[1:-1, 1:-1]

            # B: number of non-zero neighbours (2 <= B <= 6)
            B = (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9).astype(np.int16)

            # A: number of 0->1 transitions in clockwise order
            # Order: P2,P3,P4,P5,P6,P7,P8,P9,P2
            A = (((p2 == 0) & (p3 == 1)).astype(np.int16) +
                 ((p3 == 0) & (p4 == 1)).astype(np.int16) +
                 ((p4 == 0) & (p5 == 1)).astype(np.int16) +
                 ((p5 == 0) & (p6 == 1)).astype(np.int16) +
                 ((p6 == 0) & (p7 == 1)).astype(np.int16) +
                 ((p7 == 0) & (p8 == 1)).astype(np.int16) +
                 ((p8 == 0) & (p9 == 1)).astype(np.int16) +
                 ((p9 == 0) & (p2 == 1)).astype(np.int16))

            # Common conditions
            cond = (center == 1) & (B >= 2) & (B <= 6) & (A == 1)

            if step == 0:
                # Step 1: P2*P4*P6==0 and P4*P6*P8==0
                cond &= (p2 * p4 * p6 == 0)
                cond &= (p4 * p6 * p8 == 0)
            else:
                # Step 2: P2*P4*P8==0 and P2*P6*P8==0
                cond &= (p2 * p4 * p8 == 0)
                cond &= (p2 * p6 * p8 == 0)

            if cond.any():
                img[1:-1, 1:-1][cond] = 0
                changed = True

        iteration += 1
        if iteration % 10 == 0:
            remaining = img[1:-1, 1:-1].sum()
            print(f"    Thinning iteration {iteration}: {remaining:,} pixels remaining")

    # Remove padding
    return img[1:-1, 1:-1]


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

def _line_intersection(p1, d1, p2, d2):
    """Find intersection of two infinite lines defined by point + direction.

    Line 1: p1 + t*d1,  Line 2: p2 + s*d2

    Returns:
        (x, y) intersection point, or None if lines are parallel.
    """
    # Solve: p1 + t*d1 = p2 + s*d2
    # d1x*t - d2x*s = p2x - p1x
    # d1y*t - d2y*s = p2y - p1y
    det = d1[0] * (-d2[1]) - d1[1] * (-d2[0])
    if abs(det) < 1e-12:
        return None  # parallel
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    t = (-d2[1] * dx + d2[0] * dy) / det
    return (p1[0] + t * d1[0], p1[1] + t * d1[1])


def _build_wall_geometry(segments, shift):
    """Build double-line wall geometry with proper corner joins.

    For each wall segment:
      - Compute left/right offset lines (wall faces)
      - At endpoints where walls meet, compute miter intersections
        so the wall faces join cleanly
      - At free endpoints, draw end caps

    Args:
        segments: List of ((x1,y1),(x2,y2), thickness) tuples.
        shift: (sx, sy) coordinate shift applied to all points.

    Returns:
        List of dicts with keys:
          'left': ((x1,y1),(x2,y2))  — left wall face
          'right': ((x1,y1),(x2,y2)) — right wall face
          'centre': ((x1,y1),(x2,y2)) — centreline
          'caps': list of ((x1,y1),(x2,y2)) end cap lines
    """
    if not segments:
        return []

    # --- 1. Prepare shifted centrelines and normals ---
    walls = []  # per-wall data
    for seg in segments:
        (x1, y1), (x2, y2) = seg[0], seg[1]
        thickness = seg[2] if len(seg) > 2 else 0.15
        half_t = thickness / 2.0

        x1s, y1s = x1 - shift[0], y1 - shift[1]
        x2s, y2s = x2 - shift[0], y2 - shift[1]

        dx, dy = x2s - x1s, y2s - y1s
        length = np.sqrt(dx*dx + dy*dy)
        if length < 1e-12:
            continue

        ux, uy = dx / length, dy / length  # unit direction
        nx, ny = -uy * half_t, ux * half_t  # normal * half_t

        walls.append({
            'p1': np.array([x1s, y1s]),
            'p2': np.array([x2s, y2s]),
            'dir': np.array([ux, uy]),
            'normal': np.array([nx, ny]),
            'half_t': half_t,
            'thickness': thickness,
            # Offset endpoints (will be modified by miter joins)
            'L1': np.array([x1s + nx, y1s + ny]),
            'L2': np.array([x2s + nx, y2s + ny]),
            'R1': np.array([x1s - nx, y1s - ny]),
            'R2': np.array([x2s - nx, y2s - ny]),
            'cap_start': True,  # draw end cap at p1?
            'cap_end': True,    # draw end cap at p2?
        })

    # --- 2. Build endpoint adjacency ---
    # For each wall endpoint, find other walls that share approximately
    # the same centreline endpoint (= junction).
    n = len(walls)
    join_radius = max(w['half_t'] for w in walls) * 2.5 if walls else 0.2

    # Collect all endpoints: (x, y, wall_idx, end: 0=start, 1=end)
    eps = []
    for i, w in enumerate(walls):
        eps.append((*w['p1'], i, 0))
        eps.append((*w['p2'], i, 1))

    ep_coords = np.array([(e[0], e[1]) for e in eps])

    # Group into clusters
    assigned = np.full(len(eps), -1, dtype=int)
    clusters = []
    for i in range(len(eps)):
        if assigned[i] >= 0:
            continue
        dists = np.sqrt((ep_coords[:, 0] - ep_coords[i, 0])**2 +
                        (ep_coords[:, 1] - ep_coords[i, 1])**2)
        members = np.where((dists <= join_radius) & (assigned < 0))[0]
        cid = len(clusters)
        clusters.append(members.tolist())
        for m in members:
            assigned[m] = cid

    # --- 3. Process each junction cluster ---
    for cluster in clusters:
        if len(cluster) < 2:
            continue  # isolated endpoint — will get an end cap

        # Get the walls and which end participates
        participants = [(eps[j][2], eps[j][3]) for j in cluster]

        # For each pair of walls meeting at this junction, compute miter
        for a_idx in range(len(participants)):
            for b_idx in range(a_idx + 1, len(participants)):
                wi, ei = participants[a_idx]  # wall index, end (0/1)
                wj, ej = participants[b_idx]
                if wi == wj:
                    continue

                wa = walls[wi]
                wb = walls[wj]

                # Direction vectors pointing AWAY from the junction
                da = wa['dir'] if ei == 0 else -wa['dir']
                db = wb['dir'] if ej == 0 else -wb['dir']

                # Skip near-parallel walls (handled by collinear merge)
                cross = abs(da[0]*db[1] - da[1]*db[0])
                if cross < 0.1:
                    continue

                # Compute miter intersections for L and R faces
                # Wall A's left face at the junction end
                na = wa['normal']
                nb = wb['normal']

                # The left offset line of wall A: point + t*dir
                # At end 0: starts at L1, direction = wa['dir']
                # At end 1: starts at L2, direction = -wa['dir']
                a_L_pt = wa['L1'] if ei == 0 else wa['L2']
                a_R_pt = wa['R1'] if ei == 0 else wa['R2']

                # For wall B
                b_L_pt = wb['L1'] if ej == 0 else wb['L2']
                b_R_pt = wb['R1'] if ej == 0 else wb['R2']

                # Try all four combinations and pick the ones that
                # create clean intersections
                for a_side, a_pt_key in [('L', ei), ('R', ei)]:
                    a_pt = (wa[f'{a_side}1'] if a_pt_key == 0
                            else wa[f'{a_side}2'])
                    for b_side, b_pt_key in [('L', ej), ('R', ej)]:
                        b_pt = (wb[f'{b_side}1'] if b_pt_key == 0
                                else wb[f'{b_side}2'])

                        ix = _line_intersection(
                            a_pt, wa['dir'], b_pt, wb['dir']
                        )
                        if ix is None:
                            continue

                        ix = np.array(ix)
                        # Only accept if the intersection is reasonably
                        # close to the junction (within a few wall widths)
                        junction_pt = wa['p1'] if ei == 0 else wa['p2']
                        dist = np.linalg.norm(ix - junction_pt)
                        max_dist = max(wa['half_t'], wb['half_t']) * 4
                        if dist > max_dist:
                            continue

                        # Update the wall endpoints
                        key_a = f'{a_side}{1 if a_pt_key == 0 else 2}'
                        key_b = f'{b_side}{1 if b_pt_key == 0 else 2}'
                        wa[key_a] = ix
                        wb[key_b] = ix

                # Mark this end as joined (no cap needed)
                if ei == 0:
                    wa['cap_start'] = False
                else:
                    wa['cap_end'] = False
                if ej == 0:
                    wb['cap_start'] = False
                else:
                    wb['cap_end'] = False

    # --- 3b. T-junction detection ---
    # Find endpoints that land near the MIDDLE of another wall (not at
    # its endpoints). These are T-junctions — a wall or casing that
    # terminates against the side of another wall.
    for i, w in enumerate(walls):
        for end_idx in (0, 1):
            # Skip if already joined at this end
            if end_idx == 0 and not w['cap_start']:
                continue
            if end_idx == 1 and not w['cap_end']:
                continue

            ep = w['p1'] if end_idx == 0 else w['p2']

            for j, other in enumerate(walls):
                if i == j:
                    continue

                # Project ep onto other wall's centreline
                v = ep - other['p1']
                d = other['dir']
                t = np.dot(v, d)
                seg_len = np.linalg.norm(other['p2'] - other['p1'])

                # Must be within the segment span (not at the very ends)
                if t < other['half_t'] * 0.5 or t > seg_len - other['half_t'] * 0.5:
                    continue

                # Perpendicular distance
                proj = other['p1'] + t * d
                perp_dist = np.linalg.norm(ep - proj)
                max_perp = (w['half_t'] + other['half_t']) * 1.5

                if perp_dist > max_perp:
                    continue

                # T-junction found! Extend wall i's faces to meet
                # wall j's faces
                n_other = other['normal']  # already scaled by half_t

                # Wall i's left and right offset lines at this end
                for side in ('L', 'R'):
                    key = f'{side}{1 if end_idx == 0 else 2}'
                    face_pt = w[key]

                    # Find which face of the other wall is closer
                    other_L_line_pt = other['p1'] + n_other
                    other_R_line_pt = other['p1'] - n_other

                    # Intersect wall i's face line with each face of other
                    best_ix = None
                    best_dist = 999
                    for other_face_pt in [other_L_line_pt, other_R_line_pt]:
                        ix = _line_intersection(
                            face_pt, w['dir'], other_face_pt, other['dir']
                        )
                        if ix is None:
                            continue
                        ix = np.array(ix)
                        d_ix = np.linalg.norm(ix - ep)
                        if d_ix < best_dist and d_ix < max_perp * 3:
                            best_dist = d_ix
                            best_ix = ix

                    if best_ix is not None:
                        w[key] = best_ix

                # Mark end as joined
                if end_idx == 0:
                    w['cap_start'] = False
                else:
                    w['cap_end'] = False
                break  # only join to one wall per endpoint

    # --- 4. Build output geometry ---
    result = []
    for w in walls:
        geom = {
            'left': ((w['L1'][0], w['L1'][1]), (w['L2'][0], w['L2'][1])),
            'right': ((w['R1'][0], w['R1'][1]), (w['R2'][0], w['R2'][1])),
            'centre': ((w['p1'][0], w['p1'][1]), (w['p2'][0], w['p2'][1])),
            'caps': [],
        }
        if w['cap_start']:
            geom['caps'].append((
                (w['L1'][0], w['L1'][1]),
                (w['R1'][0], w['R1'][1]),
            ))
        if w['cap_end']:
            geom['caps'].append((
                (w['L2'][0], w['L2'][1]),
                (w['R2'][0], w['R2'][1]),
            ))
        result.append(geom)

    return result


def export_dxf(segments, out_path, origin_offset=True):
    """Export wall segments as double-line walls to a DXF file.

    Each wall segment is drawn as two parallel lines (both faces) with
    miter joins at corners and end caps at free endpoints.

    Args:
        segments: List of ((x1,y1), (x2,y2), thickness) tuples.
        out_path: Output DXF file path.
        origin_offset: If True, shift geometry so bottom-left is near origin.
    """
    import ezdxf
    from ezdxf import units

    doc = ezdxf.new("R2010")
    doc.units = units.M

    doc.layers.add("WALLS", color=7)
    doc.layers.add("WALL_CAPS", color=7)
    doc.layers.add("CENTRELINE", color=8)
    doc.layers.add("DIMENSIONS", color=6)

    cl_layer = doc.layers.get("CENTRELINE")
    cl_layer.off()

    msp = doc.modelspace()

    if not segments:
        print("  Warning: No wall segments to export.")
        doc.saveas(out_path)
        return

    # Compute shift
    all_pts = np.array([(p[0], p[1])
                        for seg in segments
                        for p in (seg[0], seg[1])])
    if origin_offset:
        shift = all_pts.min(axis=0) - 1.0
    else:
        shift = np.zeros(2)

    # Build wall geometry with proper corner joins
    wall_geoms = _build_wall_geometry(segments, shift)

    n_joins = sum(1 for g in wall_geoms
                  if not any(g['caps']))  # walls with no caps = fully joined
    print(f"  Corner joins: {n_joins} walls fully joined at both ends")

    for geom in wall_geoms:
        # Wall faces
        msp.add_line(geom['left'][0], geom['left'][1],
                     dxfattribs={"layer": "WALLS"})
        msp.add_line(geom['right'][0], geom['right'][1],
                     dxfattribs={"layer": "WALLS"})
        # Centreline
        msp.add_line(geom['centre'][0], geom['centre'][1],
                     dxfattribs={"layer": "CENTRELINE"})
        # End caps
        for cap in geom['caps']:
            msp.add_line(cap[0], cap[1],
                         dxfattribs={"layer": "WALL_CAPS"})

    # Add bounding dimensions
    shifted = all_pts - shift
    min_x, min_y = shifted.min(axis=0)
    max_x, max_y = shifted.max(axis=0)
    width = max_x - min_x
    height = max_y - min_y

    dim_offset = max(width, height) * 0.05 + 0.5

    dim_style = doc.dimstyles.new("PLAN")
    dim_style.dxf.dimtxt = max(width, height) * 0.015
    dim_style.dxf.dimasz = max(width, height) * 0.01

    msp.add_linear_dim(
        base=(min_x, min_y - dim_offset),
        p1=(min_x, min_y), p2=(max_x, min_y),
        dimstyle="PLAN",
        dxfattribs={"layer": "DIMENSIONS"},
    ).render()

    msp.add_linear_dim(
        base=(min_x - dim_offset, min_y),
        p1=(min_x, min_y), p2=(min_x, max_y),
        angle=90, dimstyle="PLAN",
        dxfattribs={"layer": "DIMENSIONS"},
    ).render()

    doc.saveas(out_path)
    print(f"  Saved DXF: {out_path}")
    print(f"  Extents: {width:.2f}m x {height:.2f}m")


def export_dxf_with_image(segments, image_path, image_size, resolution,
                           origin, out_path, origin_offset=True):
    """Export wall segments + the source orthoimage as a DXF underlay.

    Creates a DXF with:
      - SCAN_IMAGE layer: the PNG orthoimage as a raster background
      - WALLS layer: traced wall line segments on top
      - DIMENSIONS layer: bounding dimensions

    Args:
        segments: List of ((x1,y1), (x2,y2)) wall segments.
        image_path: Absolute path to the source PNG image file.
        image_size: (width_px, height_px) of the image.
        resolution: Metres per pixel.
        origin: (x_min, y_min) world coordinate origin of the image.
        out_path: Output DXF file path.
        origin_offset: If True, shift geometry so bottom-left is near origin.
    """
    import ezdxf
    from ezdxf import units

    doc = ezdxf.new("R2010")
    doc.units = units.M

    doc.layers.add("SCAN_IMAGE", color=8)   # Dark grey
    doc.layers.add("WALLS", color=7)        # White/black
    doc.layers.add("DIMENSIONS", color=6)   # Magenta

    msp = doc.modelspace()

    # Compute the shift (same as export_dxf)
    if segments:
        all_pts = np.array([(p[0], p[1])
                            for seg in segments
                            for p in (seg[0], seg[1])])
        if origin_offset:
            shift = all_pts.min(axis=0) - 1.0
        else:
            shift = np.zeros(2)
    else:
        shift = np.array([origin[0], origin[1]]) - 1.0

    # --- Insert the raster image ---
    img_w, img_h = image_size
    # Image insert point = origin shifted
    insert_x = origin[0] - shift[0]
    insert_y = origin[1] - shift[1]

    # Image size in world units
    world_w = img_w * resolution
    world_h = img_h * resolution

    # Use an absolute path for the image reference
    abs_image_path = os.path.abspath(image_path)

    try:
        # Define the image
        image_def = doc.add_image_def(
            filename=abs_image_path,
            size_in_pixel=(img_w, img_h),
        )

        # Insert into modelspace on the SCAN_IMAGE layer
        # The image is inserted at (insert_x, insert_y) with the
        # size scaled to match world coordinates.
        # DXF IMAGE entity uses a size vector (u, v) for pixel scaling.
        msp.add_image(
            insert=(insert_x, insert_y),
            size_in_units=(world_w, world_h),
            image_def=image_def,
            rotation=0,
            dxfattribs={"layer": "SCAN_IMAGE"},
        )
        print(f"  Added image underlay: {img_w}x{img_h}px "
              f"({world_w:.1f}x{world_h:.1f}m)")
    except Exception as e:
        print(f"  WARNING: Could not embed image: {e}")
        print(f"  (Wall lines will still be exported)")

    # --- Add wall segments (double-line with corner joins) ---
    wall_geoms = _build_wall_geometry(segments, shift)
    for geom in wall_geoms:
        msp.add_line(geom['left'][0], geom['left'][1],
                     dxfattribs={"layer": "WALLS"})
        msp.add_line(geom['right'][0], geom['right'][1],
                     dxfattribs={"layer": "WALLS"})
        for cap in geom['caps']:
            msp.add_line(cap[0], cap[1],
                         dxfattribs={"layer": "WALLS"})

    # --- Add dimensions ---
    if segments:
        shifted = all_pts - shift
        min_x, min_y = shifted.min(axis=0)
        max_x, max_y = shifted.max(axis=0)
        width = max_x - min_x
        height = max_y - min_y

        dim_offset = max(width, height) * 0.05 + 0.5

        dim_style = doc.dimstyles.new("PLAN")
        dim_style.dxf.dimtxt = max(width, height) * 0.015
        dim_style.dxf.dimasz = max(width, height) * 0.01

        msp.add_linear_dim(
            base=(min_x, min_y - dim_offset),
            p1=(min_x, min_y), p2=(max_x, min_y),
            dimstyle="PLAN",
            dxfattribs={"layer": "DIMENSIONS"},
        ).render()

        msp.add_linear_dim(
            base=(min_x - dim_offset, min_y),
            p1=(min_x, min_y), p2=(min_x, max_y),
            angle=90, dimstyle="PLAN",
            dxfattribs={"layer": "DIMENSIONS"},
        ).render()

    doc.saveas(out_path)
    print(f"  Saved DXF+image: {out_path}")


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

    Handles alpha channels correctly: if the image has transparency,
    composites onto a black background so transparent areas become black
    (empty) and opaque wall pixels are preserved.

    Returns a 2D uint8 numpy array (grayscale).
    """
    try:
        from PIL import Image
        # Increase the decompression bomb limit for large scanner images
        Image.MAX_IMAGE_PIXELS = 300_000_000

        img = Image.open(path)
        print(f"  Image mode: {img.mode}, size: {img.size}")

        if img.mode in ("RGBA", "LA", "PA"):
            # Composite onto black background — transparent = black (empty)
            background = Image.new("L", img.size, 0)
            # Split to get alpha; convert RGB/LA to grayscale
            if img.mode == "RGBA":
                gray = img.convert("LA")  # Luminance + Alpha
                l_channel, alpha = gray.split()
            elif img.mode == "LA":
                l_channel, alpha = img.split()
            else:  # PA (palette + alpha)
                img = img.convert("LA")
                l_channel, alpha = img.split()
            background.paste(l_channel, mask=alpha)
            return np.array(background, dtype=np.uint8)
        else:
            return np.array(img.convert("L"), dtype=np.uint8)
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
    img_path = None
    if save_image:
        img_path = os.path.join(output_dir, f"{basename}_ortho.png")
        save_orthoimage(image, img_path)

    # 8. Detect walls
    print("  Detecting walls...")
    segments = detect_walls(image, origin, res, min_density=min_density)

    # 9. Export DXF (walls only)
    export_dxf(segments, output_dxf)

    # 10. Export DXF with orthoimage underlay
    if img_path and os.path.isfile(img_path):
        overlay_path = os.path.join(output_dir, f"{basename}_plan_overlay.dxf")
        export_dxf_with_image(
            segments,
            image_path=os.path.abspath(img_path),
            image_size=(image.shape[1], image.shape[0]),
            resolution=res,
            origin=origin,
            out_path=overlay_path,
        )

    return output_dxf


def run_pipeline_from_image(image_path, output_dxf=None, resolution=0.02,
                            min_density=3, invert=False, max_pixels=20_000_000):
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
        max_pixels: Maximum image size in pixels. Larger images are
            downscaled to this size (default 20M pixels). The resolution
            is adjusted proportionally so the DXF output is still correct.

    Returns:
        Path to the output DXF file.
    """
    basename = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.dirname(image_path) or "."

    if output_dxf is None:
        output_dxf = os.path.join(output_dir, f"{basename}_plan.dxf")

    print(f"Reading image: {image_path}")
    image = load_image(image_path)
    print(f"  Image size: {image.shape[1]} x {image.shape[0]} pixels "
          f"({image.shape[0] * image.shape[1]:,} total)")

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

    # Remember original image info for the overlay DXF
    # (overlay references the full-res source image, not the downscaled copy)
    orig_image_path = os.path.abspath(image_path)
    orig_image_size = (image.shape[1], image.shape[0])
    orig_resolution = resolution

    # Downscale if the image is very large
    total_pixels = image.shape[0] * image.shape[1]
    if total_pixels > max_pixels:
        scale_factor = np.sqrt(max_pixels / total_pixels)
        new_h = max(1, int(image.shape[0] * scale_factor))
        new_w = max(1, int(image.shape[1] * scale_factor))
        print(f"  Downscaling: {image.shape[1]}x{image.shape[0]} -> "
              f"{new_w}x{new_h} ({scale_factor:.2f}x)")

        try:
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(image)
            pil_img = pil_img.resize((new_w, new_h), PILImage.LANCZOS)
            image = np.array(pil_img, dtype=np.uint8)
        except ImportError:
            bh = image.shape[0] // new_h
            bw = image.shape[1] // new_w
            image = image[:new_h * bh, :new_w * bw]
            image = image.reshape(new_h, bh, new_w, bw).max(axis=(1, 3)).astype(np.uint8)

        # Adjust resolution for wall detection on the downscaled image
        resolution = resolution / scale_factor
        print(f"  Adjusted resolution: {resolution:.4f} m/px")

    print(f"  Real-world extents: "
          f"{image.shape[1]*resolution:.1f}m x {image.shape[0]*resolution:.1f}m")

    print("  Detecting walls...")
    segments = detect_walls(image, origin, resolution, min_density=min_density)

    # 1. Walls-only DXF
    export_dxf(segments, output_dxf)

    # 2. DXF with image underlay (secondary output)
    overlay_path = os.path.join(
        output_dir, f"{basename}_plan_overlay.dxf"
    )
    export_dxf_with_image(
        segments,
        image_path=orig_image_path,
        image_size=orig_image_size,
        resolution=orig_resolution,
        origin=origin,
        out_path=overlay_path,
    )

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

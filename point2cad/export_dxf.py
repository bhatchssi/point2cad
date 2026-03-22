"""
DXF export for 2D projected views and cross-sections.

Exports Point2CAD 2D projections and cross-section slices to DXF format,
suitable for import into standard CAD software (AutoCAD, LibreCAD, etc.).
"""

import numpy as np

try:
    import ezdxf
    from ezdxf import units
except ImportError:
    raise ImportError(
        "ezdxf is required for DXF export. Install it with: pip install ezdxf"
    )


# DXF layer names
LAYER_EDGES = "EDGES"
LAYER_CORNERS = "CORNERS"
LAYER_TOPOLOGY = "TOPOLOGY"
LAYER_SECTIONS = "SECTIONS"
LAYER_BOUNDARY = "BOUNDARY"
LAYER_DIMENSIONS = "DIMENSIONS"

# DXF color indices (AutoCAD Color Index)
COLOR_EDGE = 7       # White/Black (depends on background)
COLOR_CORNER = 1     # Red
COLOR_TOPOLOGY = 3   # Green
COLOR_SECTION = 5    # Blue
COLOR_BOUNDARY = 4   # Cyan
COLOR_DIMENSION = 6  # Magenta


def create_dxf_document():
    """Create a new DXF document with standard layers.

    Returns:
        ezdxf.document.Drawing object.
    """
    doc = ezdxf.new("R2010")
    doc.units = units.MM

    doc.layers.add(LAYER_EDGES, color=COLOR_EDGE)
    doc.layers.add(LAYER_CORNERS, color=COLOR_CORNER)
    doc.layers.add(LAYER_TOPOLOGY, color=COLOR_TOPOLOGY)
    doc.layers.add(LAYER_SECTIONS, color=COLOR_SECTION)
    doc.layers.add(LAYER_BOUNDARY, color=COLOR_BOUNDARY)
    doc.layers.add(LAYER_DIMENSIONS, color=COLOR_DIMENSION)

    # Create a dimension style for measurements
    dim_style = doc.dimstyles.new("POINT2CAD")
    dim_style.dxf.dimtxt = 2.5      # Text height
    dim_style.dxf.dimasz = 2.0      # Arrow size
    dim_style.dxf.dimexe = 1.0      # Extension line extension
    dim_style.dxf.dimexo = 0.5      # Extension line offset from origin
    dim_style.dxf.dimgap = 0.5      # Gap between dimension line and text
    dim_style.dxf.dimclrd = COLOR_DIMENSION  # Dimension line color
    dim_style.dxf.dimclre = COLOR_DIMENSION  # Extension line color
    dim_style.dxf.dimclrt = COLOR_DIMENSION  # Text color
    dim_style.dxf.dimdec = 1        # Decimal places

    return doc


def _add_topology_curves(msp, curves_2d, layer, scale=1.0, offset=(0.0, 0.0)):
    """Add projected topology curves as polylines to DXF modelspace.

    Args:
        msp: DXF modelspace object.
        curves_2d: List of curve dicts with "points" and "connectivity" keys.
        layer: DXF layer name.
        scale: Scale factor to apply to coordinates.
        offset: (x, y) offset to apply after scaling.
    """
    for curve in curves_2d:
        pts = np.array(curve["points"]) * scale
        pts[:, 0] += offset[0]
        pts[:, 1] += offset[1]
        connectivity = curve["connectivity"]

        for edge in connectivity:
            if len(edge) == 2:
                p0 = pts[edge[0]]
                p1 = pts[edge[1]]
                msp.add_line(
                    (p0[0], p0[1]),
                    (p1[0], p1[1]),
                    dxfattribs={"layer": layer},
                )


def _add_corners(msp, corners_2d, layer, scale=1.0, offset=(0.0, 0.0),
                 marker_size=0.5):
    """Add projected corners as point markers to DXF modelspace.

    Args:
        msp: DXF modelspace object.
        corners_2d: Nx2 array of 2D corner positions.
        layer: DXF layer name.
        scale: Scale factor.
        offset: (x, y) offset.
        marker_size: Size of the cross marker at each corner.
    """
    corners = np.array(corners_2d)
    if corners.ndim != 2 or len(corners) == 0:
        return

    corners = corners * scale
    corners[:, 0] += offset[0]
    corners[:, 1] += offset[1]

    half = marker_size / 2.0
    for cx, cy in corners:
        # Draw a small cross at each corner
        msp.add_line((cx - half, cy), (cx + half, cy),
                     dxfattribs={"layer": layer})
        msp.add_line((cx, cy - half), (cx, cy + half),
                     dxfattribs={"layer": layer})
        # Also add a DXF point entity
        msp.add_point((cx, cy), dxfattribs={"layer": layer})


def _add_mesh_boundary_edges(msp, mesh_edges_list, layer, scale=1.0,
                              offset=(0.0, 0.0)):
    """Add projected mesh boundary edges to DXF modelspace.

    Args:
        msp: DXF modelspace object.
        mesh_edges_list: List of dicts with "points" (Nx2) and "edges" keys.
        layer: DXF layer name.
        scale: Scale factor.
        offset: (x, y) offset.
    """
    for edge_data in mesh_edges_list:
        pts = np.array(edge_data["points"]) * scale
        pts[:, 0] += offset[0]
        pts[:, 1] += offset[1]
        for edge in edge_data["edges"]:
            if len(edge) >= 2:
                p0 = pts[edge[0]]
                p1 = pts[edge[1]]
                msp.add_line(
                    (p0[0], p0[1]),
                    (p1[0], p1[1]),
                    dxfattribs={"layer": layer},
                )


def _add_section_contours(msp, contours, layer, scale=1.0, offset=(0.0, 0.0)):
    """Add cross-section contours to DXF modelspace.

    Args:
        msp: DXF modelspace object.
        contours: List of contour dicts with "vertices" and "edges" keys.
        layer: DXF layer name.
        scale: Scale factor.
        offset: (x, y) offset.
    """
    for contour in contours:
        verts = np.array(contour["vertices"]) * scale
        verts[:, 0] += offset[0]
        verts[:, 1] += offset[1]
        for edge in contour["edges"]:
            if len(edge) >= 2:
                p0 = verts[edge[0]]
                p1 = verts[edge[1]]
                msp.add_line(
                    (p0[0], p0[1]),
                    (p1[0], p1[1]),
                    dxfattribs={"layer": layer},
                )


def _add_view_label(msp, label, position):
    """Add a text label for a view in the DXF drawing.

    Args:
        msp: DXF modelspace object.
        label: Text string.
        position: (x, y) position for the label.
    """
    msp.add_text(
        label,
        dxfattribs={
            "height": 3.0,
            "layer": LAYER_EDGES,
        },
    ).set_placement(position)


def _compute_bounding_box(projection, scale=1.0, offset=(0.0, 0.0)):
    """Compute the 2D bounding box of all geometry in a projection.

    Returns:
        (min_x, min_y, max_x, max_y) or None if no geometry found.
    """
    all_points = []

    for curve in projection.get("curves_2d", []):
        pts = np.array(curve["points"]) * scale
        pts[:, 0] += offset[0]
        pts[:, 1] += offset[1]
        all_points.append(pts)

    corners = projection.get("corners_2d", [])
    if len(corners) > 0:
        c = np.array(corners) * scale
        c[:, 0] += offset[0]
        c[:, 1] += offset[1]
        all_points.append(c)

    for edge_data in projection.get("mesh_edges_2d", []):
        pts = np.array(edge_data["points"]) * scale
        pts[:, 0] += offset[0]
        pts[:, 1] += offset[1]
        all_points.append(pts)

    if not all_points:
        return None

    combined = np.vstack(all_points)
    return (
        float(combined[:, 0].min()),
        float(combined[:, 1].min()),
        float(combined[:, 0].max()),
        float(combined[:, 1].max()),
    )


def _add_bounding_dimensions(msp, bbox, dim_offset=8.0):
    """Add overall width and height dimensions around the bounding box.

    Args:
        msp: DXF modelspace object.
        bbox: (min_x, min_y, max_x, max_y) bounding box.
        dim_offset: Distance to offset the dimension lines from the geometry.
    """
    min_x, min_y, max_x, max_y = bbox

    # Horizontal dimension (width) along the bottom
    msp.add_linear_dim(
        base=(min_x, min_y - dim_offset),
        p1=(min_x, min_y),
        p2=(max_x, min_y),
        dimstyle="POINT2CAD",
        override={"dimtad": 1},
        dxfattribs={"layer": LAYER_DIMENSIONS},
    ).render()

    # Vertical dimension (height) along the left side
    msp.add_linear_dim(
        base=(min_x - dim_offset, min_y),
        p1=(min_x, min_y),
        p2=(min_x, max_y),
        angle=90,
        dimstyle="POINT2CAD",
        override={"dimtad": 1},
        dxfattribs={"layer": LAYER_DIMENSIONS},
    ).render()


def _add_corner_dimensions(msp, corners_2d, scale=1.0, offset=(0.0, 0.0),
                           max_dims=10, dim_offset=5.0):
    """Add dimensions between adjacent corner points.

    Dimensions are added between pairs of corners that are closest to each
    other, up to max_dims pairs, to avoid cluttering the drawing.

    Args:
        msp: DXF modelspace object.
        corners_2d: Nx2 array of corner positions.
        scale: Scale factor.
        offset: (x, y) offset.
        max_dims: Maximum number of corner-to-corner dimensions to add.
        dim_offset: Offset distance for dimension lines.
    """
    corners = np.array(corners_2d)
    if corners.ndim != 2 or len(corners) < 2:
        return

    corners = corners * scale
    corners[:, 0] += offset[0]
    corners[:, 1] += offset[1]

    # Find nearest-neighbor pairs
    from scipy.spatial import cKDTree
    tree = cKDTree(corners)
    pairs_added = set()
    dims_added = 0

    for i in range(len(corners)):
        if dims_added >= max_dims:
            break
        # Query 2 nearest (first is self)
        dists, indices = tree.query(corners[i], k=min(2, len(corners)))
        if len(indices) < 2:
            continue
        j = indices[1]
        pair = (min(i, j), max(i, j))
        if pair in pairs_added:
            continue
        pairs_added.add(pair)

        p1 = corners[i]
        p2 = corners[j]
        dist = np.linalg.norm(p2 - p1)
        if dist < 0.1:
            continue

        # Determine if dimension is more horizontal or vertical
        dx = abs(p2[0] - p1[0])
        dy = abs(p2[1] - p1[1])
        mid_y = (p1[1] + p2[1]) / 2.0
        mid_x = (p1[0] + p2[0]) / 2.0

        if dx >= dy:
            # Horizontal-ish: place dimension below
            msp.add_linear_dim(
                base=(mid_x, min(p1[1], p2[1]) - dim_offset),
                p1=(p1[0], p1[1]),
                p2=(p2[0], p2[1]),
                dimstyle="POINT2CAD",
                override={"dimtad": 1},
                dxfattribs={"layer": LAYER_DIMENSIONS},
            ).render()
        else:
            # Vertical-ish: place dimension to the right
            msp.add_linear_dim(
                base=(max(p1[0], p2[0]) + dim_offset, mid_y),
                p1=(p1[0], p1[1]),
                p2=(p2[0], p2[1]),
                angle=90,
                dimstyle="POINT2CAD",
                override={"dimtad": 1},
                dxfattribs={"layer": LAYER_DIMENSIONS},
            ).render()

        dims_added += 1


def _add_curve_length_dimensions(msp, curves_2d, scale=1.0, offset=(0.0, 0.0),
                                  max_dims=8, min_length=2.0):
    """Add length dimensions along the longest topology curves.

    Measures the straight-line distance between curve endpoints for the
    longest curves in the projection.

    Args:
        msp: DXF modelspace object.
        curves_2d: List of curve dicts with "points" and "connectivity".
        scale: Scale factor.
        offset: (x, y) offset.
        max_dims: Max number of curve dimensions.
        min_length: Minimum curve endpoint distance to annotate.
    """
    # Compute endpoint distances for all curves and sort by length
    curve_info = []
    for curve in curves_2d:
        pts = np.array(curve["points"]) * scale
        pts[:, 0] += offset[0]
        pts[:, 1] += offset[1]
        if len(pts) < 2:
            continue
        p_start = pts[0]
        p_end = pts[-1]
        dist = np.linalg.norm(p_end - p_start)
        if dist >= min_length:
            curve_info.append((dist, p_start, p_end, pts))

    # Sort by length descending, take top N
    curve_info.sort(key=lambda x: -x[0])

    dim_offset_base = 4.0
    for k, (dist, p_start, p_end, pts) in enumerate(curve_info[:max_dims]):
        # Offset each successive dimension a bit further out
        dim_offset = dim_offset_base + k * 3.0

        dx = abs(p_end[0] - p_start[0])
        dy = abs(p_end[1] - p_start[1])
        mid = (p_start + p_end) / 2.0

        if dx >= dy:
            msp.add_linear_dim(
                base=(mid[0], max(p_start[1], p_end[1]) + dim_offset),
                p1=(p_start[0], p_start[1]),
                p2=(p_end[0], p_end[1]),
                dimstyle="POINT2CAD",
                override={"dimtad": 1},
                dxfattribs={"layer": LAYER_DIMENSIONS},
            ).render()
        else:
            msp.add_linear_dim(
                base=(min(p_start[0], p_end[0]) - dim_offset, mid[1]),
                p1=(p_start[0], p_start[1]),
                p2=(p_end[0], p_end[1]),
                angle=90,
                dimstyle="POINT2CAD",
                override={"dimtad": 1},
                dxfattribs={"layer": LAYER_DIMENSIONS},
            ).render()


def _add_dimensions_to_view(msp, projection, scale=1.0, offset=(0.0, 0.0)):
    """Add all automatic dimensions to a projected view.

    Args:
        msp: DXF modelspace object.
        projection: Projection dict from generate_2d_views().
        scale: Scale factor.
        offset: (x, y) offset.
    """
    # Bounding box dimensions
    bbox = _compute_bounding_box(projection, scale=scale, offset=offset)
    if bbox is not None:
        _add_bounding_dimensions(msp, bbox)

    # Corner-to-corner dimensions
    if len(projection.get("corners_2d", [])) > 0:
        _add_corner_dimensions(
            msp, projection["corners_2d"], scale=scale, offset=offset
        )

    # Topology curve length dimensions
    _add_curve_length_dimensions(
        msp, projection["curves_2d"], scale=scale, offset=offset
    )


def export_single_view(views_2d, view_name, out_path, scale=100.0,
                       add_dimensions=True):
    """Export a single 2D view to a DXF file.

    Args:
        views_2d: Output dict from generate_2d_views().
        view_name: Name of the view to export (e.g., "top", "front", "right").
        out_path: Path for the output DXF file.
        scale: Scale factor (default 100 maps normalized coords to mm).
        add_dimensions: If True, add automatic measurement annotations.
    """
    doc = create_dxf_document()
    msp = doc.modelspace()

    projection = views_2d["projections"].get(view_name)
    if projection is None:
        print(f"Warning: View '{view_name}' not found in projections.")
        return

    _add_topology_curves(msp, projection["curves_2d"], LAYER_TOPOLOGY, scale=scale)

    if len(projection.get("corners_2d", [])) > 0:
        _add_corners(msp, projection["corners_2d"], LAYER_CORNERS, scale=scale)

    for edge_data in projection.get("mesh_edges_2d", []):
        _add_mesh_boundary_edges(msp, [edge_data], LAYER_BOUNDARY, scale=scale)

    if add_dimensions:
        _add_dimensions_to_view(msp, projection, scale=scale)

    _add_view_label(msp, view_name.upper(), (0, -10))

    doc.saveas(out_path)
    print(f"Saved DXF: {out_path}")


def export_all_views(views_2d, out_path, scale=100.0, view_spacing=150.0,
                     add_dimensions=True):
    """Export all 2D views into a single multi-view DXF drawing.

    Views are arranged side by side with labels. Cross-sections are placed
    below the main views.

    Args:
        views_2d: Output dict from generate_2d_views().
        out_path: Path for the output DXF file.
        scale: Scale factor (default 100 maps normalized coords to mm).
        view_spacing: Horizontal spacing between views in output units.
        add_dimensions: If True, add automatic measurement annotations.
    """
    doc = create_dxf_document()
    msp = doc.modelspace()

    # Arrange projection views side by side
    view_names = list(views_2d.get("projections", {}).keys())
    for i, view_name in enumerate(view_names):
        offset_x = i * view_spacing
        offset = (offset_x, 0.0)
        projection = views_2d["projections"][view_name]

        _add_topology_curves(
            msp, projection["curves_2d"], LAYER_TOPOLOGY,
            scale=scale, offset=offset,
        )

        if len(projection.get("corners_2d", [])) > 0:
            _add_corners(
                msp, projection["corners_2d"], LAYER_CORNERS,
                scale=scale, offset=offset,
            )

        for edge_data in projection.get("mesh_edges_2d", []):
            _add_mesh_boundary_edges(
                msp, [edge_data], LAYER_BOUNDARY,
                scale=scale, offset=offset,
            )

        if add_dimensions:
            _add_dimensions_to_view(msp, projection, scale=scale, offset=offset)

        _add_view_label(msp, view_name.upper(), (offset_x, -10))

    # Arrange cross-sections below projections
    sections = views_2d.get("sections", {})
    for j, (section_key, section_data) in enumerate(sections.items()):
        offset_x = j * view_spacing
        offset = (offset_x, -view_spacing)

        _add_section_contours(
            msp, section_data["contours"], LAYER_SECTIONS,
            scale=scale, offset=offset,
        )

        label = f"SECTION {section_data['axis'].upper()}={section_data['position']:.3f}"
        _add_view_label(msp, label, (offset_x, -view_spacing - 10))

    doc.saveas(out_path)
    print(f"Saved multi-view DXF: {out_path}")


def export_sections_only(views_2d, out_path, scale=100.0, view_spacing=150.0):
    """Export only cross-section slices to a DXF file.

    Args:
        views_2d: Output dict from generate_2d_views().
        out_path: Path for the output DXF file.
        scale: Scale factor.
        view_spacing: Spacing between section views.
    """
    doc = create_dxf_document()
    msp = doc.modelspace()

    sections = views_2d.get("sections", {})
    if not sections:
        print("Warning: No cross-sections available to export.")
        return

    for j, (section_key, section_data) in enumerate(sections.items()):
        offset_x = j * view_spacing
        offset = (offset_x, 0.0)

        _add_section_contours(
            msp, section_data["contours"], LAYER_SECTIONS,
            scale=scale, offset=offset,
        )

        label = f"SECTION {section_data['axis'].upper()}={section_data['position']:.3f}"
        _add_view_label(msp, label, (offset_x, -10))

    doc.saveas(out_path)
    print(f"Saved sections DXF: {out_path}")

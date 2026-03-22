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

# DXF color indices (AutoCAD Color Index)
COLOR_EDGE = 7       # White/Black (depends on background)
COLOR_CORNER = 1     # Red
COLOR_TOPOLOGY = 3   # Green
COLOR_SECTION = 5    # Blue
COLOR_BOUNDARY = 4   # Cyan


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


def export_single_view(views_2d, view_name, out_path, scale=100.0):
    """Export a single 2D view to a DXF file.

    Args:
        views_2d: Output dict from generate_2d_views().
        view_name: Name of the view to export (e.g., "top", "front", "right").
        out_path: Path for the output DXF file.
        scale: Scale factor (default 100 maps normalized coords to mm).
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

    _add_view_label(msp, view_name.upper(), (0, -10))

    doc.saveas(out_path)
    print(f"Saved DXF: {out_path}")


def export_all_views(views_2d, out_path, scale=100.0, view_spacing=150.0):
    """Export all 2D views into a single multi-view DXF drawing.

    Views are arranged side by side with labels. Cross-sections are placed
    below the main views.

    Args:
        views_2d: Output dict from generate_2d_views().
        out_path: Path for the output DXF file.
        scale: Scale factor (default 100 maps normalized coords to mm).
        view_spacing: Horizontal spacing between views in output units.
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

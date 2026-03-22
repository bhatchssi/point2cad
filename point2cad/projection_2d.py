"""
2D projection and cross-section slicing for Point2CAD meshes.

Provides orthographic projection of 3D topology (edges/corners) onto 2D planes,
and cross-section slicing of 3D meshes at specified heights/positions.
"""

import itertools
import json
import numpy as np
import trimesh


# Axis name to vector mapping
AXIS_VECTORS = {
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
}

# Standard engineering view definitions: (projection_normal, up_vector)
STANDARD_VIEWS = {
    "top": ("z", "y"),      # Plan view - looking down Z axis
    "front": ("y", "z"),    # Front elevation - looking along Y axis
    "right": ("x", "z"),    # Right elevation - looking along X axis
}


def get_projection_axes(normal_axis, up_axis=None):
    """Return the two axes to keep when projecting along normal_axis.

    Args:
        normal_axis: Axis to project along ("x", "y", or "z").
        up_axis: Optional up direction hint for consistent orientation.

    Returns:
        Tuple of (horizontal_index, vertical_index) into XYZ coordinates.
    """
    axis_map = {"x": 0, "y": 1, "z": 2}
    normal_idx = axis_map[normal_axis]
    remaining = [i for i in range(3) if i != normal_idx]

    if up_axis is not None:
        up_idx = axis_map[up_axis]
        if up_idx in remaining:
            # vertical axis = up, horizontal = the other one
            remaining.remove(up_idx)
            return remaining[0], up_idx

    return remaining[0], remaining[1]


def project_points_orthographic(points_3d, normal_axis, up_axis=None):
    """Project 3D points onto a 2D plane by dropping one axis.

    Args:
        points_3d: Nx3 array of 3D points.
        normal_axis: Axis perpendicular to the projection plane ("x", "y", "z").
        up_axis: Optional axis for vertical direction in the 2D output.

    Returns:
        Nx2 array of projected 2D points.
    """
    points_3d = np.asarray(points_3d)
    h_idx, v_idx = get_projection_axes(normal_axis, up_axis)
    return points_3d[:, [h_idx, v_idx]]


def slice_mesh_at_position(mesh, axis, position):
    """Slice a trimesh at a given position along an axis.

    Args:
        mesh: A trimesh.Trimesh object.
        axis: Axis to slice along ("x", "y", or "z").
        position: Position along the axis where the slice is taken.

    Returns:
        List of trimesh.path.Path2D contours (may be empty if no intersection).
    """
    plane_origin = np.zeros(3)
    plane_origin[{"x": 0, "y": 1, "z": 2}[axis]] = position
    plane_normal = AXIS_VECTORS[axis]

    try:
        sliced = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
        if sliced is None:
            return []
        # Get the 2D path by projecting onto the slice plane
        path_2d, _ = sliced.to_planar()
        return [path_2d]
    except Exception:
        return []


def slice_meshes_at_position(meshes, axis, position):
    """Slice multiple meshes at a given position and collect all contours.

    Args:
        meshes: List of trimesh.Trimesh objects.
        axis: Axis to slice along.
        position: Position along the axis.

    Returns:
        List of (vertices_2d, edges) tuples for each contour found.
    """
    contours = []
    for mesh in meshes:
        paths = slice_mesh_at_position(mesh, axis, position)
        for path in paths:
            if hasattr(path, "vertices") and len(path.vertices) > 0:
                edges = []
                for entity in path.entities:
                    points = entity.points
                    for i in range(len(points) - 1):
                        edges.append((points[i], points[i + 1]))
                contours.append({
                    "vertices": path.vertices,
                    "edges": edges,
                })
    return contours


def auto_slice_positions(meshes, axis, num_slices=5):
    """Compute evenly spaced slice positions spanning the mesh extents.

    Args:
        meshes: List of trimesh.Trimesh objects.
        axis: Axis along which to compute slice positions.
        num_slices: Number of slices to generate.

    Returns:
        1D array of positions along the axis.
    """
    axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
    all_min = float("inf")
    all_max = float("-inf")
    for mesh in meshes:
        bounds = mesh.bounds  # 2x3 array: [min_corner, max_corner]
        all_min = min(all_min, bounds[0, axis_idx])
        all_max = max(all_max, bounds[1, axis_idx])

    if all_min == float("inf") or all_max == float("-inf"):
        return np.array([])

    margin = (all_max - all_min) * 0.05
    return np.linspace(all_min + margin, all_max - margin, num_slices)


def project_topology_to_2d(topo_path, normal_axis, up_axis=None):
    """Project 3D topology (edges and corners) from a topo.json file to 2D.

    Args:
        topo_path: Path to the topology JSON file from save_topology().
        normal_axis: Axis to project along.
        up_axis: Optional up direction.

    Returns:
        Dict with "curves_2d" and "corners_2d" keys containing projected data.
    """
    with open(topo_path, "r") as f:
        topo = json.load(f)

    result = {"curves_2d": [], "corners_2d": []}

    for curve in topo.get("curves", []):
        pts_3d = np.array(curve["pv_points"])
        pts_2d = project_points_orthographic(pts_3d, normal_axis, up_axis)
        connectivity = curve["pv_lines"]
        result["curves_2d"].append({
            "points": pts_2d,
            "connectivity": connectivity,
        })

    corners_3d = topo.get("corners", [])
    if corners_3d:
        corners_3d = np.array(corners_3d)
        corners_2d = project_points_orthographic(corners_3d, normal_axis, up_axis)
        result["corners_2d"] = corners_2d

    return result


def generate_2d_views(clipped_meshes, topo_path, views=None, slice_axis=None,
                      slice_positions=None, num_slices=5):
    """Generate a complete set of 2D views from 3D reconstruction results.

    This is the main entry point that orchestrates projection and slicing.

    Args:
        clipped_meshes: List of trimesh.Trimesh objects from save_clipped_meshes().
        topo_path: Path to the topology JSON file.
        views: List of view names from STANDARD_VIEWS (default: all three).
        slice_axis: Optional axis for cross-section slicing.
        slice_positions: Optional explicit slice positions. If None but slice_axis
            is set, positions are auto-computed.
        num_slices: Number of auto slices if slice_positions is None.

    Returns:
        Dict with "projections" and "sections" keys containing all 2D data.
    """
    if views is None:
        views = list(STANDARD_VIEWS.keys())

    output = {"projections": {}, "sections": {}}

    # Generate orthographic projections from topology
    for view_name in views:
        if view_name not in STANDARD_VIEWS:
            print(f"Warning: Unknown view '{view_name}', skipping.")
            continue

        normal_axis, up_axis = STANDARD_VIEWS[view_name]
        projected = project_topology_to_2d(topo_path, normal_axis, up_axis)

        # Also project mesh edges (boundary edges visible in silhouette)
        mesh_edges_2d = []
        for mesh in clipped_meshes:
            boundary_edges = _extract_boundary_edges(mesh)
            if len(boundary_edges) > 0:
                mesh_edges_2d.append({
                    "points": project_points_orthographic(
                        mesh.vertices, normal_axis, up_axis
                    ),
                    "edges": boundary_edges,
                })
        projected["mesh_edges_2d"] = mesh_edges_2d
        output["projections"][view_name] = projected

    # Generate cross-sections
    if slice_axis is not None:
        if slice_positions is None:
            slice_positions = auto_slice_positions(
                clipped_meshes, slice_axis, num_slices
            )
        combined_mesh = trimesh.util.concatenate(clipped_meshes)
        for i, pos in enumerate(slice_positions):
            contours = slice_meshes_at_position(clipped_meshes, slice_axis, pos)
            section_key = f"section_{slice_axis}_{i}"
            output["sections"][section_key] = {
                "axis": slice_axis,
                "position": float(pos),
                "contours": contours,
            }

    return output


def _extract_boundary_edges(mesh):
    """Extract boundary (non-manifold or open) edges from a trimesh.

    Returns:
        List of (vertex_idx_a, vertex_idx_b) tuples.
    """
    edges = mesh.edges_unique
    edge_face_count = trimesh.grouping.group_rows(
        mesh.edges_sorted, require_count=1
    )
    # Boundary edges appear in exactly one face
    boundary_mask = np.array([len(g) == 1 for g in edge_face_count])
    if not np.any(boundary_mask):
        return []
    return edges[boundary_mask].tolist()

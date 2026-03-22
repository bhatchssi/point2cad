"""
Backend-agnostic mesh operations for Point2CAD.

Tries to use PyMesh (fast, C++), falls back to trimesh (pure Python, works
everywhere including macOS). The six operations needed by io_utils.py are:

  - form_mesh(vertices, faces)
  - merge_meshes(mesh_list) -> mesh with face_sources attribute
  - detect_self_intersection(mesh)
  - resolve_self_intersection(mesh)
  - separate_mesh(mesh)
  - remove_duplicated_vertices(mesh, tol)
"""

import warnings
import numpy as np

try:
    import pymesh as _pymesh
    BACKEND = "pymesh"
except ImportError:
    _pymesh = None
    BACKEND = "trimesh"

import trimesh


# ---------------------------------------------------------------------------
# Lightweight mesh wrapper so both backends share the same attribute API
# ---------------------------------------------------------------------------

class MeshWrapper:
    """Thin wrapper around vertices/faces with named attributes.

    Provides ``get_attribute(name)`` so call-sites written for PyMesh's API
    continue to work with the trimesh backend.
    """

    def __init__(self, vertices, faces, attributes=None):
        self.vertices = np.asarray(vertices)
        self.faces = np.asarray(faces)
        self._attributes = attributes or {}

    def get_attribute(self, name):
        if name not in self._attributes:
            raise KeyError(f"Attribute '{name}' not found on mesh")
        return self._attributes[name]

    def set_attribute(self, name, value):
        self._attributes[name] = np.asarray(value)

    @property
    def num_vertices(self):
        return len(self.vertices)

    @property
    def num_faces(self):
        return len(self.faces)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def form_mesh(vertices, faces):
    """Create a mesh from vertices and faces.

    Args:
        vertices: Nx3 array of vertex positions.
        faces: Mx3 array of triangle indices.

    Returns:
        MeshWrapper (trimesh backend) or pymesh.Mesh (pymesh backend).
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)

    if BACKEND == "pymesh":
        return _pymesh.form_mesh(vertices, faces)

    return MeshWrapper(vertices, faces)


def merge_meshes(mesh_list):
    """Merge multiple meshes into one, tracking which face came from which mesh.

    The returned mesh has a ``face_sources`` attribute: an integer array of
    length num_faces where each entry is the index of the source mesh.

    Args:
        mesh_list: List of mesh objects (MeshWrapper or pymesh.Mesh).

    Returns:
        Merged mesh with ``face_sources`` attribute.
    """
    if BACKEND == "pymesh":
        return _pymesh.merge_meshes(mesh_list)

    all_verts = []
    all_faces = []
    face_sources = []
    vertex_offset = 0

    for idx, mesh in enumerate(mesh_list):
        v = np.asarray(mesh.vertices)
        f = np.asarray(mesh.faces)
        all_verts.append(v)
        all_faces.append(f + vertex_offset)
        face_sources.append(np.full(len(f), idx, dtype=np.int32))
        vertex_offset += len(v)

    merged_verts = np.vstack(all_verts)
    merged_faces = np.vstack(all_faces)
    sources = np.concatenate(face_sources)

    merged = MeshWrapper(merged_verts, merged_faces)
    merged.set_attribute("face_sources", sources)
    return merged


def detect_self_intersection(mesh):
    """Detect pairs of self-intersecting faces.

    With the trimesh backend this returns an empty array — the result is only
    used for logging and is not consumed by downstream pipeline steps.

    Args:
        mesh: Mesh object.

    Returns:
        Nx2 array of intersecting face-index pairs (may be empty).
    """
    if BACKEND == "pymesh":
        return _pymesh.detect_self_intersection(mesh)

    return np.empty((0, 2), dtype=np.int32)


def resolve_self_intersection(mesh):
    """Resolve self-intersections by splitting overlapping triangles.

    With PyMesh this is an exact CSG-based resolution. With the trimesh
    backend, a best-effort approach is used:

    1. If ``manifold3d`` is installed, use it for boolean self-union.
    2. Otherwise, pass the mesh through unchanged (the downstream
       connected-component + proximity logic in io_utils.py tolerates this).

    The returned mesh carries a ``face_sources`` attribute that maps each
    output face back to a face in the input mesh.

    Args:
        mesh: Mesh object (must have ``face_sources`` attribute from merge).

    Returns:
        Mesh with resolved intersections and ``face_sources`` attribute.
    """
    if BACKEND == "pymesh":
        return _pymesh.resolve_self_intersection(mesh)

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    input_sources = mesh.get_attribute("face_sources")

    # Try manifold3d for high-quality boolean self-union
    try:
        import manifold3d
        manifold = manifold3d.Manifold.of_trimesh(vertices, faces)
        result_verts, result_faces = manifold.to_trimesh()
        result_verts = np.asarray(result_verts, dtype=np.float64)
        result_faces = np.asarray(result_faces, dtype=np.int32)

        # Map new faces back to input faces by centroid proximity
        new_sources = _map_face_sources(
            vertices, faces, input_sources, result_verts, result_faces
        )

        resolved = MeshWrapper(result_verts, result_faces)
        resolved.set_attribute("face_sources", new_sources)
        return resolved
    except (ImportError, Exception) as e:
        if isinstance(e, ImportError):
            warnings.warn(
                "manifold3d not installed — skipping self-intersection resolution. "
                "Mesh clipping quality may be reduced. Install with: pip install manifold3d"
            )
        else:
            warnings.warn(
                f"manifold3d self-union failed ({e}), passing mesh through unchanged."
            )

    # Fallback: return mesh unchanged with identity face_sources mapping
    resolved = MeshWrapper(vertices, faces)
    resolved.set_attribute("face_sources", input_sources.copy())
    return resolved


def separate_mesh(mesh):
    """Split a mesh into connected components.

    Args:
        mesh: Mesh object.

    Returns:
        List of mesh objects, one per connected component.
    """
    if BACKEND == "pymesh":
        return _pymesh.separate_mesh(mesh)

    tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    components = tm.split()

    result = []
    for comp in components:
        result.append(MeshWrapper(comp.vertices, comp.faces))
    return result


def remove_duplicated_vertices(mesh, tol=1e-6, importance=None):
    """Merge duplicate vertices within a tolerance.

    Args:
        mesh: Mesh object.
        tol: Distance tolerance for merging.
        importance: Unused, kept for API compatibility with PyMesh.

    Returns:
        Tuple of (cleaned_mesh, info_dict). The cleaned mesh has a
        ``face_sources`` attribute carried through from the input if present.
    """
    if BACKEND == "pymesh":
        return _pymesh.remove_duplicated_vertices(mesh, tol=tol, importance=importance)

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    # Quantize vertices to merge within tolerance
    quantized = np.round(vertices / tol) * tol
    _, unique_indices, inverse_indices = np.unique(
        quantized, axis=0, return_index=True, return_inverse=True
    )

    new_vertices = vertices[unique_indices]
    new_faces = inverse_indices[faces]

    # Remove degenerate faces (where two or more vertices collapsed)
    valid = (new_faces[:, 0] != new_faces[:, 1]) & \
            (new_faces[:, 1] != new_faces[:, 2]) & \
            (new_faces[:, 0] != new_faces[:, 2])
    new_faces = new_faces[valid]

    cleaned = MeshWrapper(new_vertices, new_faces)

    # Carry through face_sources if present
    if hasattr(mesh, '_attributes') and "face_sources" in mesh._attributes:
        cleaned.set_attribute("face_sources", mesh._attributes["face_sources"][valid])
    elif hasattr(mesh, 'get_attribute'):
        try:
            src = mesh.get_attribute("face_sources")
            cleaned.set_attribute("face_sources", src[valid])
        except (KeyError, AttributeError):
            pass

    info_dict = {
        "num_vertex_merged": len(vertices) - len(new_vertices),
        "num_face_removed": np.sum(~valid),
    }
    return cleaned, info_dict


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _map_face_sources(old_verts, old_faces, old_sources, new_verts, new_faces):
    """Map face sources from an old mesh to a new mesh via centroid proximity.

    For each face in the new mesh, find the closest face (by centroid) in the
    old mesh and inherit its source label.
    """
    from scipy.spatial import cKDTree

    old_centroids = old_verts[old_faces].mean(axis=1)
    new_centroids = new_verts[new_faces].mean(axis=1)

    tree = cKDTree(old_centroids)
    _, nearest = tree.query(new_centroids)

    return old_sources[nearest]

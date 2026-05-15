"""CGNS standard-element connectivity tables.

The face decompositions below are taken from the CGNS Standard
Interface Data Structures (SIDS) document, section "Element Types".
Vertex indices are 0-based here; the SIDS document uses 1-based, so
each table is the SIDS face list minus 1.

Each face is listed in the CGNS convention: vertices ordered so that
the right-hand rule yields an outward-pointing normal. This matches
what OpenFOAM expects on boundary faces and on internal faces owned
by the cell being decomposed; for internal faces shared between two
cells the mesh builder picks the orientation from the owner cell
(lower cell index), which gives a normal pointing from owner to
neighbour.

Only the linear element types Fluent emits for unstructured volume
meshes are supported. Quadratic types (TETRA_10, HEXA_20, ...) are
not in scope — they would require either degree reduction or per-
element subdivision, neither of which the dose tracker benefits from.
Polyhedral cells use the NGON_n/NFACE_n path and are handled in
`cgns_reader.py`, not here.
"""

from __future__ import annotations

import numpy as np

# CGNS element-type codes (SIDS section 7).
ELEMENT_TYPE_CODES: dict[str, int] = {
    "NODE": 2,
    "BAR_2": 3,
    "TRI_3": 5,
    "QUAD_4": 7,
    "TETRA_4": 10,
    "PYRA_5": 12,
    "PENTA_6": 14,
    "HEXA_8": 17,
    "MIXED": 20,
    "NGON_n": 22,
    "NFACE_n": 23,
}

ELEMENT_TYPE_NAMES: dict[int, str] = {v: k for k, v in ELEMENT_TYPE_CODES.items()}


# Number of vertices per element type.
N_VERTS: dict[str, int] = {
    "TRI_3": 3,
    "QUAD_4": 4,
    "TETRA_4": 4,
    "PYRA_5": 5,
    "PENTA_6": 6,
    "HEXA_8": 8,
}


# Face tables for the 3-D volume element types. Each entry is a list
# of (vertex-index, ...) tuples; one tuple per face, vertices in CGNS
# outward-normal order. All indices are 0-based local.

TETRA_FACES: list[tuple[int, ...]] = [
    (0, 2, 1),     # F1: base opposite apex
    (0, 1, 3),     # F2
    (1, 2, 3),     # F3
    (2, 0, 3),     # F4
]

HEXA_FACES: list[tuple[int, ...]] = [
    (0, 3, 2, 1),  # F1: bottom
    (0, 1, 5, 4),  # F2
    (1, 2, 6, 5),  # F3
    (2, 3, 7, 6),  # F4
    (0, 4, 7, 3),  # F5
    (4, 5, 6, 7),  # F6: top
]

PYRA_FACES: list[tuple[int, ...]] = [
    (0, 3, 2, 1),  # F1: square base
    (0, 1, 4),     # F2
    (1, 2, 4),     # F3
    (2, 3, 4),     # F4
    (3, 0, 4),     # F5
]

# PENTA_6 (wedge). Bottom triangle 0,1,2; top triangle 3,4,5 with
# node 3 above 0, 4 above 1, 5 above 2.
PENTA_FACES: list[tuple[int, ...]] = [
    (0, 1, 4, 3),  # F1: quad side
    (1, 2, 5, 4),  # F2: quad side
    (2, 0, 3, 5),  # F3: quad side
    (0, 2, 1),     # F4: bottom triangle
    (3, 4, 5),     # F5: top triangle
]

FACE_TABLE: dict[str, list[tuple[int, ...]]] = {
    "TETRA_4": TETRA_FACES,
    "HEXA_8": HEXA_FACES,
    "PYRA_5": PYRA_FACES,
    "PENTA_6": PENTA_FACES,
}


SUPPORTED_VOLUME_ELEMENTS: tuple[str, ...] = tuple(FACE_TABLE.keys())
SUPPORTED_BOUNDARY_ELEMENTS: tuple[str, ...] = ("TRI_3", "QUAD_4")


def cell_faces(element_type: str, connectivity: np.ndarray) -> list[tuple[int, ...]]:
    """Return the faces of a single cell.

    `connectivity` is the cell's vertex indices (length ``N_VERTS[element_type]``,
    arbitrary 0-based global indexing into the mesh's point array). The
    return value is a list of tuples of global vertex indices, one per
    face, in CGNS outward-normal order.
    """
    if element_type not in FACE_TABLE:
        raise ValueError(
            f"Unsupported volume element type: {element_type!r}. "
            f"Supported: {SUPPORTED_VOLUME_ELEMENTS}"
        )
    table = FACE_TABLE[element_type]
    conn = np.asarray(connectivity)
    if conn.shape[-1] != N_VERTS[element_type]:
        raise ValueError(
            f"{element_type} expects {N_VERTS[element_type]} vertices, "
            f"got {conn.shape[-1]}"
        )
    return [tuple(int(conn[i]) for i in local_face) for local_face in table]


def n_faces_per_cell(element_type: str) -> int:
    return len(FACE_TABLE[element_type])

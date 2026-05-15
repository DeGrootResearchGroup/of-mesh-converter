"""Unit tests for ``elements.py``: face decomposition of standard CGNS
element types.

Each element type's face table is checked against three invariants
that the CGNS SIDS guarantees, without needing to look up the SIDS
tables directly:

1. Vertex count per face is right for the element (tet: 3 each;
   hex: 4 each; pyramid: 1×4 + 4×3; wedge: 3×4 + 2×3).
2. Every local vertex appears in the union of the cell's faces
   (no vertex orphaned).
3. Every face vertex pair (an edge of the cell) is shared by exactly
   two faces — i.e., the faces of a single cell, together, form a
   closed surface (Euler characteristic = 2 for a topological sphere).

These are necessary conditions for the face table to be a valid
boundary representation of the cell. They wouldn't catch a swap
between two faces, but they catch the most likely
transcription error (wrong vertex count, wrong vertex index,
duplicated vertex).
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from of_mesh_converter import elements


@pytest.mark.parametrize(
    "type_name, expected_face_sizes",
    [
        ("TETRA_4", (3, 3, 3, 3)),
        ("PYRA_5",  (4, 3, 3, 3, 3)),
        ("PENTA_6", (4, 4, 4, 3, 3)),
        ("HEXA_8",  (4, 4, 4, 4, 4, 4)),
    ],
)
def test_face_vertex_counts(type_name, expected_face_sizes):
    faces = elements.FACE_TABLE[type_name]
    assert tuple(len(f) for f in faces) == expected_face_sizes


@pytest.mark.parametrize("type_name", elements.SUPPORTED_VOLUME_ELEMENTS)
def test_every_vertex_appears_in_some_face(type_name):
    nv = elements.N_VERTS[type_name]
    used = set()
    for face in elements.FACE_TABLE[type_name]:
        used.update(face)
    assert used == set(range(nv))


@pytest.mark.parametrize("type_name", elements.SUPPORTED_VOLUME_ELEMENTS)
def test_each_edge_shared_by_exactly_two_faces(type_name):
    """Every edge of the cell must lie on exactly two faces. If it
    appeared on one, the cell would have a hole; on three or more,
    the cell would be non-manifold."""
    counter: Counter = Counter()
    for face in elements.FACE_TABLE[type_name]:
        for i in range(len(face)):
            a = face[i]
            b = face[(i + 1) % len(face)]
            key = tuple(sorted((a, b)))
            counter[key] += 1
    bad = {k: v for k, v in counter.items() if v != 2}
    assert not bad, (
        f"{type_name}: edges with non-2 face count: {bad}"
    )


def test_cell_faces_uses_global_indices():
    """cell_faces must apply the local face table to the cell's
    global vertex indices, not return local indices."""
    # A hex with non-trivial global vertex ids.
    global_ids = np.array([100, 101, 102, 103, 200, 201, 202, 203])
    faces = elements.cell_faces("HEXA_8", global_ids)
    # F1 (bottom) in CGNS local order is (0, 3, 2, 1):
    assert faces[0] == (100, 103, 102, 101)
    # F6 (top) is (4, 5, 6, 7):
    assert faces[5] == (200, 201, 202, 203)


def test_cell_faces_rejects_wrong_vertex_count():
    with pytest.raises(ValueError, match="HEXA_8 expects 8 vertices"):
        elements.cell_faces("HEXA_8", np.array([0, 1, 2, 3]))


def test_cell_faces_rejects_unsupported_type():
    with pytest.raises(ValueError, match="Unsupported volume element type"):
        elements.cell_faces("NGON_n", np.array([0, 1, 2]))

"""Unit tests for ``mesh_builder.py``: cell-vertex -> face-based mesh."""

from __future__ import annotations

import numpy as np
import pytest

from of_mesh_converter.mesh_builder import (
    BoundaryFaceGroup,
    CellBlock,
    build_mesh,
)


def _single_hex_inputs():
    """One HEXA_8 cell, vertex ordering matches CGNS convention."""
    pts = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=np.float64,
    )
    cells = CellBlock(element_type="HEXA_8", connectivity=np.array([[0, 1, 2, 3, 4, 5, 6, 7]]))
    # The six external faces of the single hex, sorted-vertex-key
    # match independent of orientation.
    boundary_groups = [
        BoundaryFaceGroup(
            name="all_boundary",
            type="wall",
            faces=[
                (0, 3, 2, 1),  # z=0
                (4, 5, 6, 7),  # z=1
                (0, 1, 5, 4),  # y=0
                (3, 2, 6, 7),  # y=1
                (0, 4, 7, 3),  # x=0
                (1, 5, 6, 2),  # x=1
            ],
        )
    ]
    return pts, cells, boundary_groups


def test_single_hex_has_six_boundary_faces_no_internal():
    pts, cells, groups = _single_hex_inputs()
    mesh = build_mesh(pts, [cells], groups)
    assert mesh.n_cells == 1
    assert mesh.n_internal_faces == 0
    assert mesh.n_boundary_faces == 6
    assert len(mesh.patches) == 1
    assert mesh.patches[0].name == "all_boundary"
    assert mesh.patches[0].n_faces == 6
    # All boundary faces own the only cell.
    np.testing.assert_array_equal(mesh.owner, np.zeros(6, dtype=np.int32))


def test_two_stacked_hexes_share_internal_face():
    """Two hexes stacked along x produce one internal face and ten
    boundary faces (five on each cell)."""
    pts = np.array(
        [
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
            [2, 0, 0], [2, 1, 0], [2, 0, 1], [2, 1, 1],
        ],
        dtype=np.float64,
    )
    cells = CellBlock(
        element_type="HEXA_8",
        connectivity=np.array([
            [0, 1, 2, 3, 4, 5, 6, 7],
            [1, 8, 9, 2, 5, 10, 11, 6],
        ]),
    )
    # Wrap all 10 boundary faces in one patch.
    boundary_groups = [
        BoundaryFaceGroup(
            name="all",
            type="wall",
            faces=[
                # cell 0 outer
                (0, 3, 2, 1), (0, 1, 5, 4), (3, 2, 6, 7), (0, 4, 7, 3),
                (4, 5, 6, 7),
                # cell 1 outer
                (1, 8, 9, 2), (5, 10, 11, 6), (1, 5, 10, 8), (2, 6, 11, 9),
                (8, 10, 11, 9),
            ],
        )
    ]
    mesh = build_mesh(pts, [cells], boundary_groups)
    assert mesh.n_cells == 2
    assert mesh.n_internal_faces == 1
    assert mesh.n_boundary_faces == 10
    # Internal face: owner is lower-index cell (0); neighbour is 1.
    assert int(mesh.owner[0]) == 0
    assert int(mesh.neighbour[0]) == 1


def test_unassigned_boundary_face_raises():
    """If we forget to put a boundary face in some patch, the builder
    must reject the mesh (rather than silently inventing one)."""
    pts, cells, groups = _single_hex_inputs()
    # Drop the last face from the patch.
    groups[0].faces = groups[0].faces[:5]
    with pytest.raises(ValueError, match="were not assigned to any patch"):
        build_mesh(pts, [cells], groups)


def test_boundary_face_not_in_mesh_raises():
    pts, cells, groups = _single_hex_inputs()
    # Add a face referencing a vertex that exists but isn't part of
    # any cell face combination.
    groups[0].faces.append((0, 2, 4, 6))
    with pytest.raises(ValueError, match="does not match any cell face"):
        build_mesh(pts, [cells], groups)


def test_internal_face_ordering_is_upper_triangular():
    """Internal faces must be ordered so that owner is ascending and,
    within an owner, neighbour is ascending. This is what OpenFOAM's
    addressing.C assumes for matrix assembly."""
    # 1x1x3 stack of hexes → 2 internal faces.
    pts = np.array(
        [
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
            [0, 0, 2], [1, 0, 2], [1, 1, 2], [0, 1, 2],
            [0, 0, 3], [1, 0, 3], [1, 1, 3], [0, 1, 3],
        ],
        dtype=np.float64,
    )
    cells = CellBlock(
        element_type="HEXA_8",
        connectivity=np.array([
            [0, 1, 2, 3,  4, 5, 6, 7],
            [4, 5, 6, 7,  8, 9, 10, 11],
            [8, 9, 10, 11, 12, 13, 14, 15],
        ]),
    )
    # External faces: 2 bottom-ish + 2 top-ish + 4 sides per cell
    # but the internal ones we want to ignore. We construct an
    # all-boundary patch with the outer faces and rely on the
    # builder to identify internal faces by hash.
    boundary_faces = [
        # bottom of cell 0
        (0, 3, 2, 1),
        # top of cell 2
        (12, 13, 14, 15),
        # side faces of all three stacked cells
        (0, 1, 5, 4),  (1, 2, 6, 5),  (2, 3, 7, 6),  (0, 4, 7, 3),
        (4, 5, 9, 8),  (5, 6, 10, 9), (6, 7, 11, 10),(4, 8, 11, 7),
        (8, 9, 13, 12),(9, 10, 14, 13),(10, 11, 15, 14),(8, 12, 15, 11),
    ]
    groups = [BoundaryFaceGroup(name="wall", type="wall", faces=boundary_faces)]
    mesh = build_mesh(pts, [cells], groups)
    assert mesh.n_internal_faces == 2
    # owner ascending, neighbour ascending.
    assert int(mesh.owner[0]) == 0 and int(mesh.neighbour[0]) == 1
    assert int(mesh.owner[1]) == 1 and int(mesh.neighbour[1]) == 2

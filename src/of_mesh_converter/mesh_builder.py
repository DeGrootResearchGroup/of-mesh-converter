"""Build a face-based ``Mesh`` from cell-vertex CGNS input.

Algorithm:

1. For every cell, enumerate the cell's faces using
   ``elements.cell_faces`` (CGNS outward-normal orientation).
2. Hash each face by its sorted vertex tuple. Faces that appear
   twice are internal; faces that appear once are boundary.
3. Match CGNS boundary 2-D elements (TRI_3 / QUAD_4) against the
   boundary faces by sorted vertex tuple and tag each boundary
   face with its patch name.
4. Order faces: internal first (sorted by ``(owner, neighbour)`` —
   OpenFOAM's upper-triangular convention), then each boundary
   patch in turn, in the order patches were given.
5. For internal faces, the owner is the lower-indexed cell and the
   face orientation is the one generated when decomposing the owner.
   That guarantees the face normal points from owner to neighbour.

The output is a ``Mesh`` ready to hand to ``foam_writer.write_mesh``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .elements import cell_faces
from .mesh_ir import Mesh, Patch


@dataclass
class BoundaryFaceGroup:
    """A run of CGNS 2-D boundary elements that belong to one patch.

    ``faces`` is a list of vertex-index tuples (one per face). Order
    inside a group is preserved on output, so the user's CGNS face
    ordering survives the round-trip.
    """

    name: str
    type: str  # OF patch type: "patch" / "wall" / "symmetry" / ...
    faces: list[tuple[int, ...]]


@dataclass
class CellBlock:
    """A homogeneous block of CGNS volume elements.

    ``connectivity`` is shape ``(n_cells, n_verts_per_cell)`` of
    global 0-based point indices.
    """

    element_type: str
    connectivity: np.ndarray


def _face_key(verts: tuple[int, ...]) -> tuple[int, ...]:
    """Order-independent key for matching opposite-orientation copies
    of the same face."""
    return tuple(sorted(verts))


def build_mesh(
    points: np.ndarray,
    cell_blocks: list[CellBlock],
    boundary_groups: list[BoundaryFaceGroup],
) -> Mesh:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (n_points, 3); got {points.shape}")

    # 1. Enumerate every cell's faces. cell_index runs across all
    # blocks in the order they were given (block 0 cells first, then
    # block 1, etc.).
    face_by_key: dict[tuple[int, ...], list[tuple[int, tuple[int, ...]]]] = {}
    cell_index = 0
    for block in cell_blocks:
        conn = np.asarray(block.connectivity)
        if conn.ndim != 2:
            raise ValueError(
                f"block.connectivity must be 2-D; got shape {conn.shape}"
            )
        for row in conn:
            for face in cell_faces(block.element_type, row):
                face_by_key.setdefault(_face_key(face), []).append(
                    (cell_index, face)
                )
            cell_index += 1
    n_cells = cell_index

    # 2. Partition into internal / boundary.
    internal: list[tuple[int, int, tuple[int, ...]]] = []  # (owner, neigh, verts)
    boundary_by_key: dict[tuple[int, ...], tuple[int, tuple[int, ...]]] = {}
    for key, hits in face_by_key.items():
        if len(hits) == 1:
            cell_idx, verts = hits[0]
            boundary_by_key[key] = (cell_idx, verts)
        elif len(hits) == 2:
            (c1, v1), (c2, v2) = hits
            if c1 < c2:
                owner, neigh, verts = c1, c2, v1
            else:
                owner, neigh, verts = c2, c1, v2
            internal.append((owner, neigh, verts))
        else:
            raise ValueError(
                f"Face with key {key} is shared by {len(hits)} cells "
                f"(expected 1 or 2). Mesh is non-manifold."
            )

    # OF's upper-triangular convention: internal faces sorted by
    # (owner, neighbour). This is what addressing.C in OF assumes
    # when building the LDU matrix.
    internal.sort(key=lambda t: (t[0], t[1]))

    # 3. Walk boundary groups in user order and consume entries from
    # boundary_by_key. Anything left over after the loop is an
    # unassigned boundary face (a mesh error or a missing patch).
    patches: list[Patch] = []
    boundary_faces_ordered: list[tuple[int, tuple[int, ...]]] = []
    start = len(internal)
    for group in boundary_groups:
        n_in_patch = 0
        for cgns_face in group.faces:
            key = _face_key(cgns_face)
            entry = boundary_by_key.pop(key, None)
            if entry is None:
                raise ValueError(
                    f"Boundary face {cgns_face} in patch {group.name!r} "
                    f"does not match any cell face."
                )
            cell_idx, owner_oriented_verts = entry
            # We keep the owner's outward orientation, which is what
            # OF wants on boundary faces (normal pointing out of the
            # domain).
            boundary_faces_ordered.append((cell_idx, owner_oriented_verts))
            n_in_patch += 1
        patches.append(
            Patch(
                name=group.name,
                type=group.type,
                start_face=start,
                n_faces=n_in_patch,
            )
        )
        start += n_in_patch

    if boundary_by_key:
        n_left = len(boundary_by_key)
        raise ValueError(
            f"{n_left} boundary face(s) were not assigned to any patch. "
            f"Every external face must belong to a CGNS BC."
        )

    # 4. Materialise the final arrays.
    n_internal = len(internal)
    n_total = n_internal + len(boundary_faces_ordered)

    faces: list[list[int]] = []
    owner = np.empty(n_total, dtype=np.int32)
    neighbour = np.empty(n_internal, dtype=np.int32)

    for i, (own, nei, verts) in enumerate(internal):
        faces.append(list(verts))
        owner[i] = own
        neighbour[i] = nei

    for j, (own, verts) in enumerate(boundary_faces_ordered):
        faces.append(list(verts))
        owner[n_internal + j] = own

    return Mesh(
        points=points,
        faces=faces,
        owner=owner,
        neighbour=neighbour,
        patches=patches,
        n_cells=n_cells,
    )

"""Build a face-based ``Mesh`` from CGNS cell input.

Two entry points:

- ``build_mesh`` for standard (cell-vertex) elements, described below.
- ``build_polyhedral_mesh`` for ``NGON_n`` / ``NFACE_n`` input, which
  is already face-based: each face is listed once and each cell lists
  its faces with a sign giving the face orientation. That path only
  has to pick owner / neighbour, flip faces whose stored orientation
  points into the owner, and reorder.

Standard-element algorithm:

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

from dataclasses import dataclass, field

import numpy as np

from .elements import cell_faces
from .mesh_ir import Mesh, Patch, faces_to_csr


@dataclass
class BoundaryFaceGroup:
    """A run of CGNS 2-D boundary elements that belong to one patch.

    ``faces`` is a list of vertex-index tuples (one per face), used by
    ``build_mesh``. ``face_ids`` is used by ``build_polyhedral_mesh``
    instead: 0-based indices into the ``PolyhedralCells`` face list.
    Order inside a group is preserved on output, so the user's CGNS
    face ordering survives the round-trip.
    """

    name: str
    type: str  # OF patch type: "patch" / "wall" / "symmetry" / ...
    faces: list[tuple[int, ...]] = field(default_factory=list)
    face_ids: np.ndarray | None = None


@dataclass
class CellBlock:
    """A homogeneous block of CGNS volume elements.

    ``connectivity`` is shape ``(n_cells, n_verts_per_cell)`` of
    global 0-based point indices.
    """

    element_type: str
    connectivity: np.ndarray


@dataclass
class PolyhedralCells:
    """Polyhedral cells in the CGNS ``NGON_n`` / ``NFACE_n`` layout.

    Face ``i`` is ``face_vertices[face_offsets[i]:face_offsets[i + 1]]``
    (0-based point indices). Cell ``c`` is
    ``cell_faces[cell_offsets[c]:cell_offsets[c + 1]]``: signed,
    1-based indices into the face list. ``+f`` means face ``f``'s
    right-hand-rule normal points out of the cell, ``-f`` into it.
    """

    face_offsets: np.ndarray
    face_vertices: np.ndarray
    cell_offsets: np.ndarray
    cell_faces: np.ndarray

    @property
    def n_faces(self) -> int:
        return int(self.face_offsets.shape[0]) - 1

    @property
    def n_cells(self) -> int:
        return int(self.cell_offsets.shape[0]) - 1


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

    faces: list[tuple[int, ...]] = []
    owner = np.empty(n_total, dtype=np.int32)
    neighbour = np.empty(n_internal, dtype=np.int32)

    for i, (own, nei, verts) in enumerate(internal):
        faces.append(verts)
        owner[i] = own
        neighbour[i] = nei

    for j, (own, verts) in enumerate(boundary_faces_ordered):
        faces.append(verts)
        owner[n_internal + j] = own

    face_offsets, face_vertices = faces_to_csr(faces)
    return Mesh(
        points=points,
        face_offsets=face_offsets,
        face_vertices=face_vertices,
        owner=owner,
        neighbour=neighbour,
        patches=patches,
        n_cells=n_cells,
    )


def build_polyhedral_mesh(
    points: np.ndarray,
    cells: PolyhedralCells,
    boundary_groups: list[BoundaryFaceGroup],
) -> Mesh:
    """Build a ``Mesh`` from ``NGON_n`` / ``NFACE_n`` input.

    Every face must be referenced by one cell (boundary) or two cells
    of opposite sign (internal). The owner of an internal face is the
    lower-indexed cell; a face whose stored orientation points into
    its owner is reversed, OpenFOAM-style (first vertex kept). Faces
    no cell references are dropped. Each boundary face must belong
    to exactly one group, via ``BoundaryFaceGroup.face_ids``.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (n_points, 3); got {points.shape}")

    n_faces = cells.n_faces
    n_cells = cells.n_cells
    face_offsets = np.asarray(cells.face_offsets, dtype=np.int64)
    face_vertices = np.asarray(cells.face_vertices, dtype=np.int64)
    signed = np.asarray(cells.cell_faces, dtype=np.int64)

    if signed.size and (np.any(signed == 0) or np.abs(signed).max() > n_faces):
        raise ValueError(
            f"NFACE_n references a face outside 1..{n_faces} (or face 0)."
        )
    if face_vertices.size and (
        face_vertices.min() < 0 or face_vertices.max() >= points.shape[0]
    ):
        raise ValueError("NGON_n references a point outside the coordinate arrays.")

    ref_cell = np.repeat(
        np.arange(n_cells, dtype=np.int64), np.diff(cells.cell_offsets)
    )
    ref_face = np.abs(signed) - 1
    ref_out = signed > 0

    counts = np.bincount(ref_face, minlength=n_faces)
    if np.any(counts > 2):
        bad = int(np.flatnonzero(counts > 2)[0])
        raise ValueError(
            f"Face {bad} is referenced by {int(counts[bad])} cells "
            "(expected 1 or 2). Mesh is non-manifold."
        )

    # First and (for internal faces) second reference of every face.
    order = np.argsort(ref_face, kind="stable")
    first = np.searchsorted(ref_face[order], np.arange(n_faces))
    used = counts > 0
    internal = counts == 2
    boundary = counts == 1

    c1 = np.full(n_faces, -1, dtype=np.int64)
    o1 = np.zeros(n_faces, dtype=bool)
    c1[used] = ref_cell[order[first[used]]]
    o1[used] = ref_out[order[first[used]]]
    c2 = np.full(n_faces, -1, dtype=np.int64)
    o2 = np.zeros(n_faces, dtype=bool)
    c2[internal] = ref_cell[order[first[internal] + 1]]
    o2[internal] = ref_out[order[first[internal] + 1]]

    same_sign = internal & (o1 == o2)
    if np.any(same_sign):
        bad = int(np.flatnonzero(same_sign)[0])
        raise ValueError(
            f"{int(same_sign.sum())} internal face(s) have the same sign in "
            f"both cells (e.g. face {bad}); NFACE_n orientation is inconsistent."
        )
    self_ref = internal & (c1 == c2)
    if np.any(self_ref):
        raise ValueError(
            f"{int(self_ref.sum())} face(s) are listed twice by the same cell."
        )

    owner_all = np.where(internal, np.minimum(c1, c2), c1)
    neigh_all = np.where(internal, np.maximum(c1, c2), -1)
    # Orientation of the face as stored, relative to its owner.
    owner_out = np.where(c1 == owner_all, o1, o2)
    flip_all = ~owner_out

    # Internal faces in upper-triangular order.
    internal_ids = np.flatnonzero(internal)
    internal_ids = internal_ids[
        np.lexsort((neigh_all[internal_ids], owner_all[internal_ids]))
    ]

    # Boundary faces, patch by patch.
    assigned = np.zeros(n_faces, dtype=bool)
    patches: list[Patch] = []
    boundary_chunks: list[np.ndarray] = []
    start = internal_ids.size
    for group in boundary_groups:
        if group.face_ids is None:
            raise ValueError(
                f"Patch {group.name!r} has no face_ids; the polyhedral "
                "builder needs face indices, not vertex tuples."
            )
        ids = np.asarray(group.face_ids, dtype=np.int64).ravel()
        if ids.size and (ids.min() < 0 or ids.max() >= n_faces):
            raise ValueError(f"Patch {group.name!r} references a face out of range.")
        not_boundary = ~boundary[ids]
        if np.any(not_boundary):
            bad = int(ids[not_boundary][0])
            raise ValueError(
                f"Patch {group.name!r}: face {bad} is referenced by "
                f"{int(counts[bad])} cells, so it is not a boundary face."
            )
        if np.any(assigned[ids]) or np.unique(ids).size != ids.size:
            raise ValueError(
                f"Patch {group.name!r} contains faces already assigned to a patch."
            )
        assigned[ids] = True
        boundary_chunks.append(ids)
        patches.append(
            Patch(name=group.name, type=group.type, start_face=start, n_faces=ids.size)
        )
        start += ids.size

    unassigned = boundary & ~assigned
    if np.any(unassigned):
        raise ValueError(
            f"{int(unassigned.sum())} boundary face(s) were not assigned to "
            "any patch. Every external face must belong to a CGNS BC."
        )

    final = np.concatenate([internal_ids, *boundary_chunks]).astype(np.int64)

    # Gather the vertices of the final face list, reversing flipped
    # faces as (v0, v[n-1], ..., v1).
    sizes = np.diff(face_offsets)[final]
    new_offsets = np.zeros(final.size + 1, dtype=np.int64)
    np.cumsum(sizes, out=new_offsets[1:])
    local = np.arange(new_offsets[-1], dtype=np.int64) - np.repeat(
        new_offsets[:-1], sizes
    )
    flip = np.repeat(flip_all[final], sizes)
    rep_sizes = np.repeat(sizes, sizes)
    src_local = np.where(flip & (local > 0), rep_sizes - local, local)
    new_vertices = face_vertices[np.repeat(face_offsets[final], sizes) + src_local]

    return Mesh(
        points=points,
        face_offsets=new_offsets,
        face_vertices=new_vertices,
        owner=owner_all[final].astype(np.int32),
        neighbour=neigh_all[internal_ids].astype(np.int32),
        patches=patches,
        n_cells=n_cells,
    )

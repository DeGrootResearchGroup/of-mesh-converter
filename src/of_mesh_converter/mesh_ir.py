"""Format-neutral intermediate representation produced by the reader
and consumed by the writer.

The reader is responsible for filling these in from CGNS. The writer
consumes them and knows nothing about CGNS. This split lets each side
be tested in isolation and leaves a hook for adding other readers
(STAR-CCM+ CGNS, EnSight, ...) later without touching the writer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Patch:
    """A boundary patch: a contiguous run of boundary faces sharing
    a name and a type (used by OpenFOAM to pick a default BC class)."""

    name: str
    type: str  # "patch", "wall", "empty", "symmetry", ...
    start_face: int
    n_faces: int


@dataclass
class Mesh:
    """Face-based OpenFOAM-style mesh.

    Face ordering invariant: internal faces first, then each boundary
    patch in turn. `patches[i].start_face` is an index into the face
    list / `owner`. Internal faces have a `neighbour` entry; boundary
    faces do not, so `len(neighbour) == n_internal_faces`.

    Faces are stored compressed: face ``i`` is
    ``face_vertices[face_offsets[i]:face_offsets[i + 1]]``. Production
    polyhedral meshes have millions of faces, which as Python lists
    would cost an order of magnitude more memory.
    """

    points: np.ndarray  # (n_points, 3) float64
    face_offsets: np.ndarray  # (n_faces + 1,) int64, starts at 0
    face_vertices: np.ndarray  # (face_offsets[-1],) int64 point indices
    owner: np.ndarray  # (n_faces,) int32 cell index
    neighbour: np.ndarray  # (n_internal_faces,) int32 cell index
    patches: list[Patch] = field(default_factory=list)
    n_cells: int = 0

    @property
    def faces(self) -> list[list[int]]:
        """The faces as a list of point-index lists. Convenience for
        small meshes and tests; large-mesh code should use the
        compressed arrays directly."""
        verts = self.face_vertices.tolist()
        offs = self.face_offsets.tolist()
        return [verts[offs[i]:offs[i + 1]] for i in range(len(offs) - 1)]

    @property
    def n_faces(self) -> int:
        return int(self.face_offsets.shape[0]) - 1

    @property
    def n_internal_faces(self) -> int:
        return int(self.neighbour.shape[0])

    @property
    def n_boundary_faces(self) -> int:
        return self.n_faces - self.n_internal_faces


@dataclass
class CaseData:
    """Mesh + per-cell field arrays, keyed by OpenFOAM field name."""

    mesh: Mesh

    # Scalar fields (n_cells,) and vector fields (n_cells, 3),
    # keyed by their OpenFOAM names ("U", "k", "epsilon", "G", ...).
    scalar_fields: dict[str, np.ndarray] = field(default_factory=dict)
    vector_fields: dict[str, np.ndarray] = field(default_factory=dict)

    # Diagnostics carried through from the reader / sanitiser so the
    # sanity report can print them. Free-form by design — anything
    # interesting the reader noticed goes here.
    notes: list[str] = field(default_factory=list)


def faces_to_csr(faces) -> tuple[np.ndarray, np.ndarray]:
    """Pack a sequence of point-index sequences into
    ``(face_offsets, face_vertices)``."""
    sizes = np.fromiter((len(f) for f in faces), dtype=np.int64, count=len(faces))
    offsets = np.zeros(len(faces) + 1, dtype=np.int64)
    np.cumsum(sizes, out=offsets[1:])
    vertices = np.fromiter(
        (int(v) for f in faces for v in f), dtype=np.int64, count=int(offsets[-1])
    )
    return offsets, vertices

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
    patch in turn. `patches[i].start_face` is an index into `faces` /
    `owner`. Internal faces have a `neighbour` entry; boundary faces
    do not, so `len(neighbour) == n_internal_faces`.
    """

    points: np.ndarray  # (n_points, 3) float64
    faces: list[list[int]]  # list of point-index lists, one per face
    owner: np.ndarray  # (n_faces,) int32 cell index
    neighbour: np.ndarray  # (n_internal_faces,) int32 cell index
    patches: list[Patch] = field(default_factory=list)
    n_cells: int = 0

    @property
    def n_faces(self) -> int:
        return len(self.faces)

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

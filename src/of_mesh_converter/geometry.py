"""Face and cell geometry of a ``Mesh``, computed the way OpenFOAM does.

Face centre and area vector: triangles from each edge to the average of
the face's points, area-weighted (``primitiveMeshFaceCentresAndAreas``).
Cell centre and volume: pyramids from each face to the average of the
cell's face centres, volume-weighted with the pyramid centroid at 3/4 of
the way from apex to face (``primitiveMeshCellCentresAndVols``).

Matching OpenFOAM's definitions matters for ``transfer``: it compares
these centres with the ``C`` field OpenFOAM writes for the target mesh.
"""

from __future__ import annotations

import numpy as np

from .mesh_ir import Mesh


def face_centres_and_areas(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(Cf, Sf)``, each ``(n_faces, 3)``."""
    off = mesh.face_offsets
    verts = mesh.face_vertices
    n_faces = mesh.n_faces
    sizes = np.diff(off)
    face_of = np.repeat(np.arange(n_faces), sizes)
    pts = mesh.points[verts]

    p_avg = np.add.reduceat(pts, off[:-1], axis=0) / sizes[:, None]
    pos = np.arange(verts.size)
    is_last = pos == off[1:][face_of] - 1
    nxt = np.where(is_last, off[:-1][face_of], pos + 1)

    a = pts
    b = mesh.points[verts[nxt]]
    c = p_avg[face_of]
    tri_n = 0.5 * np.cross(b - a, c - a)
    tri_c = (a + b + c) / 3.0

    sf = np.zeros((n_faces, 3))
    np.add.at(sf, face_of, tri_n)
    mag = np.linalg.norm(sf, axis=1)
    n_hat = sf / np.where(mag > 0, mag, 1.0)[:, None]
    w = np.einsum("ij,ij->i", tri_n, n_hat[face_of])
    sum_w = np.bincount(face_of, weights=w, minlength=n_faces)
    cf = np.column_stack([
        np.bincount(face_of, weights=w * tri_c[:, k], minlength=n_faces)
        for k in range(3)
    ]) / np.where(sum_w > 0, sum_w, 1.0)[:, None]
    degenerate = sum_w <= 0
    cf[degenerate] = p_avg[degenerate]
    return cf, sf


def cell_centres_and_volumes(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(C, V)``: ``(n_cells, 3)`` centres and ``(n_cells,)``
    volumes."""
    cf, sf = face_centres_and_areas(mesh)
    n_cells = mesh.n_cells
    n_int = mesh.n_internal_faces
    own = mesh.owner.astype(np.int64)
    nei = mesh.neighbour.astype(np.int64)

    cells = np.concatenate([own, nei])
    face_cf = np.concatenate([cf, cf[:n_int]])
    count = np.bincount(cells, minlength=n_cells)
    c_est = np.column_stack([
        np.bincount(cells, weights=face_cf[:, k], minlength=n_cells)
        for k in range(3)
    ]) / np.maximum(count, 1)[:, None]

    # Three times each pyramid's volume; positive for a face pointing
    # out of the cell.
    vol3_own = np.einsum("ij,ij->i", sf, cf - c_est[own])
    vol3_nei = np.einsum("ij,ij->i", sf[:n_int], c_est[nei] - cf[:n_int])
    vol3 = np.maximum(np.concatenate([vol3_own, vol3_nei]), 0.0)
    pyr_c = 0.75 * face_cf + 0.25 * c_est[cells]

    sum_v = np.bincount(cells, weights=vol3, minlength=n_cells)
    centres = np.column_stack([
        np.bincount(cells, weights=vol3 * pyr_c[:, k], minlength=n_cells)
        for k in range(3)
    ]) / np.where(sum_v > 0, sum_v, 1.0)[:, None]
    flat = sum_v <= 0
    centres[flat] = c_est[flat]
    return centres, sum_v / 3.0

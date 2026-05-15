"""Printable end-of-run audit for a converted case.

This is the artefact the user shows their QA lead — confirms the
mesh actually came out of the converter with sensible bounds, every
patch maps to a recognisable name, and the field statistics look
like a real flow solution rather than e.g. all zeros.
"""

from __future__ import annotations

from io import StringIO

import numpy as np

from .mesh_ir import CaseData


def format_report(
    case: CaseData,
    *,
    patch_name_mapping: dict[str, str] | None = None,
    n_clipped_k: int = 0,
    n_clipped_epsilon: int = 0,
) -> str:
    buf = StringIO()
    p = lambda s="": buf.write(s + "\n")  # noqa: E731

    p("=" * 72)
    p("of-mesh-converter — conversion sanity report")
    p("=" * 72)

    mesh = case.mesh
    pts = mesh.points
    p("")
    p("Mesh")
    p(f"  points          : {pts.shape[0]}")
    p(f"  cells           : {mesh.n_cells}")
    p(f"  faces           : {mesh.n_faces} "
      f"(internal {mesh.n_internal_faces}, boundary {mesh.n_boundary_faces})")
    if pts.size:
        mn = pts.min(axis=0)
        mx = pts.max(axis=0)
        p(f"  bounding box    : "
          f"[{mn[0]:.6g}, {mn[1]:.6g}, {mn[2]:.6g}] x "
          f"[{mx[0]:.6g}, {mx[1]:.6g}, {mx[2]:.6g}]")
        extents = mx - mn
        p(f"  extents (LxWxH) : {extents[0]:.6g} x {extents[1]:.6g} x {extents[2]:.6g}")

    p("")
    p("Patches")
    if not mesh.patches:
        p("  (none)")
    for patch in mesh.patches:
        p(f"  {patch.name:24s} type={patch.type:10s} nFaces={patch.n_faces}")

    if patch_name_mapping:
        renamed = {k: v for k, v in patch_name_mapping.items() if k != v}
        if renamed:
            p("")
            p("Patch-name sanitisation (CGNS source -> OF identifier)")
            for src, dst in renamed.items():
                p(f"  {src!r} -> {dst!r}")

    p("")
    p("Fields")
    if not case.scalar_fields and not case.vector_fields:
        p("  (none)")
    for name, arr in case.scalar_fields.items():
        a = np.asarray(arr)
        p(f"  {name:10s} (scalar)  "
          f"min={a.min():+.6g} max={a.max():+.6g} mean={a.mean():+.6g}")
    for name, arr in case.vector_fields.items():
        a = np.asarray(arr)
        mag = np.linalg.norm(a, axis=1)
        p(f"  {name:10s} (vector)  "
          f"|.| min={mag.min():.6g} max={mag.max():.6g} mean={mag.mean():.6g}")

    clipped_lines: list[str] = []
    if n_clipped_k:
        clipped_lines.append(f"  k       clipped in {n_clipped_k} cell(s)")
    if n_clipped_epsilon:
        clipped_lines.append(f"  epsilon clipped in {n_clipped_epsilon} cell(s)")
    if clipped_lines:
        p("")
        p("Turbulence sanitisation (values <= 0 floored to 1e-12)")
        for line in clipped_lines:
            p(line)

    if case.notes:
        p("")
        p("Notes")
        for note in case.notes:
            p(f"  - {note}")

    p("=" * 72)
    return buf.getvalue()

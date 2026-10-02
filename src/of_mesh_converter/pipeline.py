"""End-to-end convert: CGNS path in, OpenFOAM case directory out.

The pipeline is the only place that wires the reader, sanitiser,
mesh builder, writer, and sanity report together. Each step is in
its own module and individually unit-testable; the pipeline is the
integration layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import cgns_reader, foam_writer, sanitise, sanity_report
from .mesh_builder import PolyhedralCells, build_mesh, build_polyhedral_mesh
from .mesh_ir import CaseData


@dataclass
class LoadedCase:
    """A CGNS file read, sanitised and built into a ``Mesh``, before
    anything is written."""

    case: CaseData
    patch_name_mapping: dict[str, str]
    n_clipped_k: int
    n_clipped_epsilon: int


def load(cgns_path: Path | str) -> LoadedCase:
    """Read a CGNS file into a ``CaseData``: mesh built, patch names
    sanitised, ``k`` / ``epsilon`` floored. Shared by ``convert`` and
    ``transfer.transfer``."""
    notes: list[str] = []
    points, cells, boundary_groups, flow_solution = cgns_reader.read_cgns(
        Path(cgns_path), notes
    )

    # Sanitise patch names and resolve collisions before the builder
    # ever sees them — that way the Mesh.patches list and the field BC
    # blocks both speak the final OF-side names.
    original_names = [g.name for g in boundary_groups]
    sanitised, mapping = sanitise.sanitise_patch_names(original_names)
    for group, new_name in zip(boundary_groups, sanitised):
        group.name = new_name

    if isinstance(cells, PolyhedralCells):
        mesh = build_polyhedral_mesh(points, cells, boundary_groups)
    else:
        mesh = build_mesh(points, cells, boundary_groups)

    scalars = dict(flow_solution.get("scalars", {}))
    vectors = dict(flow_solution.get("vectors", {}))
    n_clip_k = 0
    n_clip_eps = 0
    if "k" in scalars:
        scalars["k"], n_clip_k = sanitise.clip_nonpositive(scalars["k"])
    if "epsilon" in scalars:
        scalars["epsilon"], n_clip_eps = sanitise.clip_nonpositive(scalars["epsilon"])

    return LoadedCase(
        case=CaseData(
            mesh=mesh, scalar_fields=scalars, vector_fields=vectors, notes=notes
        ),
        patch_name_mapping=mapping,
        n_clipped_k=n_clip_k,
        n_clipped_epsilon=n_clip_eps,
    )


def convert(
    cgns_path: Path | str,
    out_dir: Path | str,
    *,
    write_report: bool = True,
) -> tuple[CaseData, str]:
    """Convert one CGNS file to an OpenFOAM case directory.

    Returns ``(CaseData, report_text)``. The report is also printed
    to stdout when invoked from the CLI; library callers get it back
    as a string and can decide what to do with it.
    """
    out_dir = Path(out_dir)
    loaded = load(cgns_path)
    case = loaded.case
    scalars, vectors, notes = case.scalar_fields, case.vector_fields, case.notes

    if "U" not in vectors:
        notes.append(
            "U missing from FlowSolution — radiationDose needs a velocity "
            "field. Supply U in 0/ yourself, or, if an OpenFOAM flow case "
            "on the same mesh exists, use `of-mesh-converter transfer` to "
            "put these fields into it instead."
        )
    if "k" not in scalars or "epsilon" not in scalars:
        notes.append(
            "k or epsilon missing from FlowSolution — "
            "radiationDose's DRW dispersion model requires both. "
            "Either re-export with a k-epsilon turbulence model or "
            "switch the dispersion model in postProcess.dict to 'none'."
        )
    if "G" not in scalars:
        notes.append(
            "G (fluence rate) not present in CGNS. The dose tracker "
            "needs G in the 0/ directory; supply it via "
            "setFluenceRate, the DOM solver, or a user-written field."
        )

    foam_writer.write_case(case, out_dir)

    # The sanity report is also written next to the case so the user
    # can re-read it after the run.
    report = sanity_report.format_report(
        case,
        patch_name_mapping=loaded.patch_name_mapping,
        n_clipped_k=loaded.n_clipped_k,
        n_clipped_epsilon=loaded.n_clipped_epsilon,
    )
    if write_report:
        (out_dir / "conversion_report.txt").write_text(report)

    return case, report

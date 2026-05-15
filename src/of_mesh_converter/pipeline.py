"""End-to-end convert: CGNS path in, OpenFOAM case directory out.

The pipeline is the only place that wires the reader, sanitiser,
mesh builder, writer, and sanity report together. Each step is in
its own module and individually unit-testable; the pipeline is the
integration layer.
"""

from __future__ import annotations

from pathlib import Path

from . import cgns_reader, foam_writer, sanitise, sanity_report
from .mesh_builder import build_mesh
from .mesh_ir import CaseData


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
    cgns_path = Path(cgns_path)
    out_dir = Path(out_dir)

    points, cell_blocks, boundary_groups, flow_solution = cgns_reader.read_cgns(
        cgns_path
    )

    # 1. Sanitise patch names and resolve collisions before the
    # builder ever sees them — that way the Mesh.patches list and
    # the field BC blocks both speak the final OF-side names.
    original_names = [g.name for g in boundary_groups]
    sanitised, mapping = sanitise.sanitise_patch_names(original_names)
    for group, new_name in zip(boundary_groups, sanitised):
        group.name = new_name

    # 2. Build the face-based mesh from cell-vertex CGNS input.
    mesh = build_mesh(points, cell_blocks, boundary_groups)

    # 3. Assemble fields and run turbulence-floor clipping.
    scalars = dict(flow_solution.get("scalars", {}))
    vectors = dict(flow_solution.get("vectors", {}))
    n_clip_k = 0
    n_clip_eps = 0
    if "k" in scalars:
        scalars["k"], n_clip_k = sanitise.clip_nonpositive(scalars["k"])
    if "epsilon" in scalars:
        scalars["epsilon"], n_clip_eps = sanitise.clip_nonpositive(scalars["epsilon"])

    notes: list[str] = []
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

    case = CaseData(
        mesh=mesh,
        scalar_fields=scalars,
        vector_fields=vectors,
        notes=notes,
    )

    # 4. Write everything.
    foam_writer.write_case(case, out_dir)

    # 5. Build the sanity report and (optionally) write it next to
    # the case so the user can re-read it after the run.
    report = sanity_report.format_report(
        case,
        patch_name_mapping=mapping,
        n_clipped_k=n_clip_k,
        n_clipped_epsilon=n_clip_eps,
    )
    if write_report:
        (out_dir / "conversion_report.txt").write_text(report)

    return case, report

"""Write the IR to disk in OpenFOAM ASCII format.

Produces:

  case_dir/
    constant/polyMesh/{points,faces,owner,neighbour,boundary}
    0/{U,k,epsilon,G}                 (only those present in IR)
    system/{controlDict,fvSchemes,fvSolution,postProcess.dict}

The system stubs are the bare minimum to run ``foamPostProcess`` with
the radiationDose function object: a small ``controlDict`` with the
function object loaded, ``fvSchemes`` and ``fvSolution`` placeholders
(no equations are solved on these fields — they're frozen inputs),
and a ``postProcess.dict`` template with the imported patch names
plugged into the ``escapePatches`` / ``patches`` slots, leaving the
seeding model and kInact list as ``TODO`` markers for the user.

The writer does not know about CGNS. It consumes a ``CaseData``
object and nothing else.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .field_mapping import OF_DIMENSIONS
from .mesh_ir import CaseData, Mesh

# OpenFOAM ASCII file header used at the top of every output file.
# ``CLASS`` and ``OBJECT`` are formatted in per-file.
_HEADER = """\
/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Website:  https://openfoam.org                  |
|   \\\\  /    A nd           | Version:  13                                    |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       {cls};
    location    "{loc}";
    object      {obj};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""

_FOOTER = "// ************************************************************************* //\n"


def _header(cls: str, loc: str, obj: str) -> str:
    return _HEADER.format(cls=cls, loc=loc, obj=obj)


def _format_vector(v: np.ndarray) -> str:
    return f"({float(v[0])} {float(v[1])} {float(v[2])})"


def _format_face(verts: list[int]) -> str:
    inner = " ".join(str(int(x)) for x in verts)
    return f"{len(verts)}({inner})"


def _format_dimensions(of_name: str) -> str:
    if of_name not in OF_DIMENSIONS:
        raise KeyError(f"No dimension set defined for field {of_name!r}")
    dims = OF_DIMENSIONS[of_name]
    return "[" + " ".join(str(d) for d in dims) + "]"


def write_case(case: CaseData, out_dir: Path | str) -> None:
    out_dir = Path(out_dir)
    (out_dir / "constant" / "polyMesh").mkdir(parents=True, exist_ok=True)
    (out_dir / "0").mkdir(parents=True, exist_ok=True)
    (out_dir / "system").mkdir(parents=True, exist_ok=True)

    write_mesh(case.mesh, out_dir)
    write_fields(case, out_dir)
    write_system_stubs(case, out_dir)


def write_mesh(mesh: Mesh, out_dir: Path | str) -> None:
    out_dir = Path(out_dir)
    pm = out_dir / "constant" / "polyMesh"
    pm.mkdir(parents=True, exist_ok=True)

    _write_points(mesh, pm / "points")
    _write_faces(mesh, pm / "faces")
    _write_labels(mesh.owner, pm / "owner", obj="owner",
                  note=f"nPoints:{len(mesh.points)} "
                       f"nCells:{mesh.n_cells} "
                       f"nFaces:{mesh.n_faces} "
                       f"nInternalFaces:{mesh.n_internal_faces}")
    _write_labels(mesh.neighbour, pm / "neighbour", obj="neighbour",
                  note=f"nInternalFaces:{mesh.n_internal_faces}")
    _write_boundary(mesh, pm / "boundary")


def _write_points(mesh: Mesh, path: Path) -> None:
    n = len(mesh.points)
    lines = [_header("vectorField", "constant/polyMesh", "points"), ""]
    lines.append(f"{n}")
    lines.append("(")
    for p in mesh.points:
        lines.append(_format_vector(p))
    lines.append(")")
    lines.append("")
    lines.append(_FOOTER)
    path.write_text("\n".join(lines))


def _write_faces(mesh: Mesh, path: Path) -> None:
    n = len(mesh.faces)
    lines = [_header("faceList", "constant/polyMesh", "faces"), ""]
    lines.append(f"{n}")
    lines.append("(")
    for face in mesh.faces:
        lines.append(_format_face(face))
    lines.append(")")
    lines.append("")
    lines.append(_FOOTER)
    path.write_text("\n".join(lines))


def _write_labels(arr: np.ndarray, path: Path, obj: str, note: str) -> None:
    n = int(arr.shape[0])
    lines = [_header("labelList", "constant/polyMesh", obj), ""]
    lines.append(f"// {note}")
    lines.append(f"{n}")
    lines.append("(")
    for x in arr:
        lines.append(str(int(x)))
    lines.append(")")
    lines.append("")
    lines.append(_FOOTER)
    path.write_text("\n".join(lines))


def _write_boundary(mesh: Mesh, path: Path) -> None:
    lines = [_header("polyBoundaryMesh", "constant/polyMesh", "boundary"), ""]
    lines.append(f"{len(mesh.patches)}")
    lines.append("(")
    for patch in mesh.patches:
        lines.append(f"    {patch.name}")
        lines.append("    {")
        lines.append(f"        type            {patch.type};")
        if patch.type == "wall":
            lines.append("        inGroups        List<word> 1(wall);")
        lines.append(f"        nFaces          {patch.n_faces};")
        lines.append(f"        startFace       {patch.start_face};")
        lines.append("    }")
    lines.append(")")
    lines.append("")
    lines.append(_FOOTER)
    path.write_text("\n".join(lines))


def write_fields(case: CaseData, out_dir: Path | str) -> None:
    out_dir = Path(out_dir)
    zero_dir = out_dir / "0"
    zero_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in case.scalar_fields.items():
        _write_scalar_field(name, arr, case.mesh, zero_dir / name)
    for name, arr in case.vector_fields.items():
        _write_vector_field(name, arr, case.mesh, zero_dir / name)


def _patch_bc_block(mesh: Mesh) -> list[str]:
    """zeroGradient on every patch. The fields are frozen inputs to a
    Lagrangian tracker; we never solve an equation on them, so BC
    type fidelity is not load-bearing."""
    out = ["boundaryField", "{"]
    for patch in mesh.patches:
        out.append(f"    {patch.name}")
        out.append("    {")
        out.append("        type            zeroGradient;")
        out.append("    }")
    out.append("}")
    return out


def _write_scalar_field(name: str, arr: np.ndarray, mesh: Mesh, path: Path) -> None:
    if arr.shape[0] != mesh.n_cells:
        raise ValueError(
            f"Scalar field {name!r} has {arr.shape[0]} values but mesh "
            f"has {mesh.n_cells} cells"
        )
    dims = _format_dimensions(name)
    lines = [_header("volScalarField", "0", name), ""]
    lines.append(f"dimensions      {dims};")
    lines.append("")
    lines.append(f"internalField   nonuniform List<scalar>")
    lines.append(f"{arr.shape[0]}")
    lines.append("(")
    for v in arr:
        lines.append(repr(float(v)))
    lines.append(")")
    lines.append(";")
    lines.append("")
    lines.extend(_patch_bc_block(mesh))
    lines.append("")
    lines.append(_FOOTER)
    path.write_text("\n".join(lines))


def _write_vector_field(name: str, arr: np.ndarray, mesh: Mesh, path: Path) -> None:
    if arr.shape != (mesh.n_cells, 3):
        raise ValueError(
            f"Vector field {name!r} has shape {arr.shape} but mesh has "
            f"{mesh.n_cells} cells (expected ({mesh.n_cells}, 3))"
        )
    dims = _format_dimensions(name)
    lines = [_header("volVectorField", "0", name), ""]
    lines.append(f"dimensions      {dims};")
    lines.append("")
    lines.append("internalField   nonuniform List<vector>")
    lines.append(f"{arr.shape[0]}")
    lines.append("(")
    for v in arr:
        lines.append(_format_vector(v))
    lines.append(")")
    lines.append(";")
    lines.append("")
    lines.extend(_patch_bc_block(mesh))
    lines.append("")
    lines.append(_FOOTER)
    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# system/ stubs

_CONTROL_DICT = """\
application     foamPostProcess;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         0;
deltaT          1;
writeControl    timeStep;
writeInterval   1;
writeFormat     ascii;
writePrecision  9;
runTimeModifiable yes;

functions
{
    #includeFunc "postProcess.dict"
}
"""

_FV_SCHEMES = """\
ddtSchemes      { default steadyState; }
gradSchemes     { default Gauss linear; }
divSchemes      { default none; }
laplacianSchemes{ default none; }
interpolationSchemes{ default linear; }
snGradSchemes   { default corrected; }
"""

_FV_SOLUTION = """\
solvers {}
"""


def _post_process_dict(case: CaseData) -> str:
    patch_names = [p.name for p in case.mesh.patches]
    patches_inline = " ".join(patch_names)
    return f"""\
//
// radiationDose function object stub.
//
// The patch names below were imported from the source CGNS file by
// of-mesh-converter and are guaranteed to match constant/polyMesh/boundary.
// Fill in the seeding model, escape patches, and inactivation rate
// constants (kInact list) for your case before running:
//
//   foamPostProcess -dict system/postProcess.dict -latestTime
//
// All imported patches: ( {patches_inline} )

radiationDose
{{
    type            radiationDose;
    libs            ("libradiationDose.so");

    U               U;
    fluenceRate     G;
    seed            42;

    seeding
    {{
        // TODO: pick a seedingModel suitable for your case. Example:
        type        patchInjection;
        patches     ( /* inlet patch name from list above */ );
        nParticles  10000;
    }}

    dispersion
    {{
        type        discreteRandomWalk;
        k           k;
        epsilon     epsilon;
        Cl          0.15;
    }}

    termination
    {{
        // TODO: list escape patches (typically the outlet).
        escapePatches   ( /* outlet patch name from list above */ );
        maxTime         300;
        maxDose         0;
        wallReflection  true;
    }}

    integration
    {{
        dtMax           0.005;
        cflMax          0.5;
        maxOuterSteps   200000;
    }}

    output
    {{
        // TODO: list inactivation rate constants (cm^2/mJ) for the
        // organisms of interest.
        kInact          ( 0.1 );
    }}
}}
"""


def write_system_stubs(case: CaseData, out_dir: Path | str) -> None:
    out_dir = Path(out_dir)
    sys_dir = out_dir / "system"
    sys_dir.mkdir(parents=True, exist_ok=True)
    (sys_dir / "controlDict").write_text(
        _header("dictionary", "system", "controlDict") + "\n" +
        _CONTROL_DICT + "\n" + _FOOTER
    )
    (sys_dir / "fvSchemes").write_text(
        _header("dictionary", "system", "fvSchemes") + "\n" +
        _FV_SCHEMES + "\n" + _FOOTER
    )
    (sys_dir / "fvSolution").write_text(
        _header("dictionary", "system", "fvSolution") + "\n" +
        _FV_SOLUTION + "\n" + _FOOTER
    )
    (sys_dir / "postProcess.dict").write_text(
        _header("dictionary", "system", "postProcess") + "\n" +
        _post_process_dict(case) + "\n" + _FOOTER
    )

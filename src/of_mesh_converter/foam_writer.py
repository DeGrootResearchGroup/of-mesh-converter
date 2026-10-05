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


def _write_list(path: Path, header: str, lines, n: int, note: str | None = None) -> None:
    """Stream an OpenFOAM ASCII list: header, count, one entry per
    line. ``lines`` is any iterable of preformatted entries."""
    with open(path, "w") as fh:
        fh.write(header)
        fh.write("\n")
        if note is not None:
            fh.write(f"// {note}\n")
        fh.write(f"{n}\n(\n")
        for line in lines:
            fh.write(line)
            fh.write("\n")
        fh.write(")\n\n")
        fh.write(_FOOTER)


def _write_points(mesh: Mesh, path: Path) -> None:
    _write_list(
        path,
        _header("vectorField", "constant/polyMesh", "points"),
        (f"({x!r} {y!r} {z!r})" for x, y, z in mesh.points.tolist()),
        len(mesh.points),
    )


def _write_faces(mesh: Mesh, path: Path) -> None:
    verts = mesh.face_vertices.tolist()
    offs = mesh.face_offsets.tolist()

    def lines():
        for i in range(len(offs) - 1):
            face = verts[offs[i]:offs[i + 1]]
            yield f"{len(face)}({' '.join(map(str, face))})"

    _write_list(
        path,
        _header("faceList", "constant/polyMesh", "faces"),
        lines(),
        mesh.n_faces,
    )


def _write_labels(arr: np.ndarray, path: Path, obj: str, note: str) -> None:
    _write_list(
        path,
        _header("labelList", "constant/polyMesh", obj),
        map(str, np.asarray(arr).tolist()),
        int(arr.shape[0]),
        note=note,
    )


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
    write_field_file(path, name, arr, _patch_bc_block(mesh), location="0")


def _write_vector_field(name: str, arr: np.ndarray, mesh: Mesh, path: Path) -> None:
    if arr.shape != (mesh.n_cells, 3):
        raise ValueError(
            f"Vector field {name!r} has shape {arr.shape} but mesh has "
            f"{mesh.n_cells} cells (expected ({mesh.n_cells}, 3))"
        )
    write_field_file(path, name, arr, _patch_bc_block(mesh), location="0")


def write_field_file(
    path: Path,
    name: str,
    arr: np.ndarray,
    boundary_field: list[str],
    *,
    location: str,
) -> None:
    """Write a ``volScalarField`` ((n,) array) or ``volVectorField``
    ((n, 3) array) with a nonuniform internal field. ``boundary_field``
    is the complete ``boundaryField { ... }`` block, one line per item."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        cls, kind = "volScalarField", "scalar"
        entries = map(repr, arr.tolist())
    elif arr.ndim == 2 and arr.shape[1] == 3:
        cls, kind = "volVectorField", "vector"
        entries = (f"({x!r} {y!r} {z!r})" for x, y, z in arr.tolist())
    else:
        raise ValueError(f"Field {name!r}: unsupported shape {arr.shape}")
    with open(path, "w") as fh:
        fh.write(_header(cls, location, name))
        fh.write(f"\ndimensions      {_format_dimensions(name)};\n\n")
        fh.write(f"internalField   nonuniform List<{kind}>\n{arr.shape[0]}\n(\n")
        for line in entries:
            fh.write(line)
            fh.write("\n")
        fh.write(")\n;\n\n")
        fh.write("\n".join(boundary_field))
        fh.write("\n\n")
        fh.write(_FOOTER)


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

"""``transfer``: copy CGNS fields into an existing OpenFOAM case on the
same mesh, and ``geometry``: the cell centres it matches on."""

from __future__ import annotations

import numpy as np
import pytest

from of_mesh_converter import geometry, pipeline, transfer
from of_mesh_converter.__main__ import main
from of_mesh_converter.foam_writer import _header, _FOOTER

from .cgns_fixture import hex_box_cgns, poly_box_cgns


def _write_vector_field(path, values, *, binary=False):
    n = len(values)
    head = _header("volVectorField", path.parent.name, path.name)
    if binary:
        head = head.replace("format      ascii;", 'format      binary;\n    arch        "LSB;label=32;scalar=64";')
        body = (
            f"\ndimensions      [0 1 0 0 0 0 0];\n\ninternalField   nonuniform List<vector> {n}(".encode()
            + np.asarray(values, dtype="<f8").tobytes()
            + b");\n\nboundaryField\n{\n}\n"
        )
        path.write_bytes(head.encode() + body)
        return
    lines = "\n".join(f"({x!r} {y!r} {z!r})" for x, y, z in np.asarray(values).tolist())
    path.write_text(
        head + f"\ndimensions      [0 1 0 0 0 0 0];\n\ninternalField   nonuniform List<vector>\n"
        f"{n}\n(\n{lines}\n)\n;\n\nboundaryField\n{{\n}}\n\n" + _FOOTER
    )


def _write_boundary(case, patches):
    pm = case / "constant" / "polyMesh"
    pm.mkdir(parents=True, exist_ok=True)
    lines = [_header("polyBoundaryMesh", "constant/polyMesh", "boundary"), f"{len(patches)}", "("]
    for i, (name, ptype) in enumerate(patches):
        lines += [f"    {name}", "    {", f"        type            {ptype};",
                  "        inGroups        List<word> 1(wall);" if ptype == "wall" else "",
                  f"        nFaces          {i};", f"        startFace       {i};", "    }"]
    lines += [")", _FOOTER]
    (pm / "boundary").write_text("\n".join(lines))


@pytest.fixture
def source_and_target(tmp_path):
    """A polyhedral CGNS source and a target case whose cells are the
    same, numbered in a different (random) order."""
    src = tmp_path / "poly.cgns"
    truth = poly_box_cgns(src, nx=4, ny=3, nz=2)
    centres, _ = geometry.cell_centres_and_volumes(pipeline.load(src).case.mesh)
    perm = np.random.default_rng(1).permutation(truth["n_cells"])
    target = tmp_path / "target"
    (target / "100").mkdir(parents=True)
    _write_vector_field(target / "100" / "C", centres[perm] + 1e-9)
    _write_boundary(target, [("walls", "wall"), ("inlet", "patch"),
                             ("seam", "nonConformalCyclic"), ("outlet", "patch")])
    return src, target, truth, perm, centres


def test_cell_centres_and_volumes_of_hex_box(tmp_path):
    p = tmp_path / "hex.cgns"
    truth = hex_box_cgns(p, nx=3, ny=2, nz=2, Lx=0.3, Ly=0.2, Lz=0.2)
    mesh = pipeline.load(p).case.mesh
    c, v = geometry.cell_centres_and_volumes(mesh)
    hexes = pipeline.cgns_reader.read_cgns(p)[1][0].connectivity
    np.testing.assert_allclose(c, truth["points"][hexes].mean(axis=1), atol=1e-15)
    np.testing.assert_allclose(v, 0.1 ** 3, rtol=1e-12)


def test_cell_centres_of_skewed_cell_are_volume_centroids():
    """A wedge-like cell where the vertex average is not the centroid."""
    from of_mesh_converter.mesh_builder import BoundaryFaceGroup, CellBlock, build_mesh
    from of_mesh_converter.elements import cell_faces

    pts = np.array([[0, 0, 0], [2, 0, 0], [0, 1, 0], [0, 0, 1], [2, 0, 1], [0, 1, 1]], float)
    row = np.arange(6)
    faces = cell_faces("PENTA_6", row)
    mesh = build_mesh(pts, [CellBlock("PENTA_6", row[None, :])],
                      [BoundaryFaceGroup("all", "wall", faces=faces)])
    c, v = geometry.cell_centres_and_volumes(mesh)
    np.testing.assert_allclose(v, [1.0])  # triangle area 1 x height 1
    np.testing.assert_allclose(c[0], [2 / 3, 1 / 3, 0.5], atol=1e-14)


def test_transfer_copies_by_cell_centre(source_and_target):
    src, target, truth, perm, _ = source_and_target
    match, report = transfer.transfer(src, target, fields=["G"])
    assert np.array_equal(match.source_of, perm)
    text = (target / "100" / "G").read_text()
    body = text[text.index("(", text.index("internalField")) + 1:text.index("\n)")]
    np.testing.assert_array_equal(np.array(body.split(), float), truth["G"][perm])
    assert "one to one" in report
    assert (target / "transfer_report.txt").exists()


def test_transfer_new_field_gets_zero_gradient_and_constraint_types(source_and_target):
    src, target, *_ = source_and_target
    transfer.transfer(src, target, fields=["G"])
    text = (target / "100" / "G").read_text()
    bf = text[text.index("boundaryField"):]
    assert bf.count("zeroGradient") == 3
    assert "type            nonConformalCyclic;" in bf
    assert "dimensions      [1 0 -3 0 0 0 0];" in text


def test_transfer_reads_binary_cell_centres(source_and_target):
    src, target, _truth, perm, centres = source_and_target
    _write_vector_field(target / "100" / "C", centres[perm], binary=True)
    match, _ = transfer.transfer(src, target, fields=["G"])
    assert np.array_equal(match.source_of, perm)
    assert match.distance.max() < 1e-15


def test_transfer_refuses_to_overwrite_without_flag(source_and_target):
    src, target, *_ = source_and_target
    (target / "100" / "G").write_text("existing")
    with pytest.raises(FileExistsError, match="--overwrite"):
        transfer.transfer(src, target, fields=["G"])
    assert (target / "100" / "G").read_text() == "existing"


def test_transfer_overwrite_keeps_existing_boundary_field(source_and_target):
    src, target, *_ = source_and_target
    old = _header("volScalarField", "100", "G") + (
        "\ndimensions [1 0 -3 0 0 0 0];\ninternalField uniform 0;\n"
        "boundaryField\n{\n    walls\n    {\n        type fixedValue;\n"
        "        value uniform 7;\n    }\n    \".*\"\n    {\n        type zeroGradient;\n    }\n}\n"
    )
    (target / "100" / "G").write_text(old)
    _, report = transfer.transfer(src, target, fields=["G"], overwrite=True)
    text = (target / "100" / "G").read_text()
    assert "value uniform 7;" in text
    assert "nonuniform List<scalar>" in text
    assert "kept from the file it replaced" in report


def test_transfer_rejects_different_mesh(source_and_target):
    src, target, _truth, perm, centres = source_and_target
    moved = centres[perm].copy()
    moved[5] += 0.02  # about a quarter of the cell size V**(1/3)
    _write_vector_field(target / "100" / "C", moved)
    with pytest.raises(ValueError, match="not the same"):
        transfer.transfer(src, target, fields=["G"])
    assert not (target / "100" / "G").exists()


def test_transfer_rejects_different_cell_count(source_and_target):
    src, target, _truth, perm, centres = source_and_target
    _write_vector_field(target / "100" / "C", centres[perm][:-1])
    with pytest.raises(ValueError, match="mapFields"):
        transfer.transfer(src, target, fields=["G"])


def test_transfer_rejects_many_to_one_match(source_and_target):
    src, target, _truth, perm, centres = source_and_target
    twice = centres[perm].copy()
    twice[1] = twice[0]
    _write_vector_field(target / "100" / "C", twice)
    with pytest.raises(ValueError):
        transfer.transfer(src, target, fields=["G"])


def test_transfer_needs_cell_centres(source_and_target):
    src, target, *_ = source_and_target
    (target / "100" / "C").unlink()
    with pytest.raises(FileNotFoundError, match="writeCellCentres -time 100"):
        transfer.transfer(src, target, fields=["G"])


def test_transfer_unknown_field(source_and_target):
    src, target, *_ = source_and_target
    with pytest.raises(ValueError, match="provides"):
        transfer.transfer(src, target, fields=["U"])


def test_transfer_cli(source_and_target, capsys):
    src, target, *_ = source_and_target
    assert main(["transfer", str(src), str(target), "--fields", "G", "--time", "100"]) == 0
    assert "cells matched" in capsys.readouterr().out
    assert main(["transfer", str(src), str(target), "--fields", "G"]) == 1
    assert "--overwrite" in capsys.readouterr().err


def test_match_cells_handles_crowded_hash_cells():
    """Many source centres in one hash cube (tolerance large against the
    spacing) must still give the exact nearest neighbours."""
    rng = np.random.default_rng(0)
    src = rng.random((500, 3))
    perm = rng.permutation(500)
    # Tolerance 0.5 makes the hash cubes half the domain: ~60 per cube.
    match = transfer.match_cells(src, np.full(500, 1e-2), src[perm] + 1e-6, rel_tol=50.0)
    assert np.array_equal(match.source_of, perm)


def test_match_cells_finds_neighbours_across_hash_cubes():
    """Targets displaced by most of the tolerance land in a neighbouring
    hash cube of their source far more often than not."""
    g = np.arange(6) * 0.3
    src = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
    rng = np.random.default_rng(2)
    shift = rng.normal(size=src.shape)
    shift *= 0.08 / np.linalg.norm(shift, axis=1, keepdims=True)
    perm = rng.permutation(len(src))
    # tolerance = 0.1 * 1.0 = 0.1 = hash cube side; shifts are 0.08.
    match = transfer.match_cells(src, np.ones(len(src)), (src + shift)[perm], rel_tol=0.1)
    assert np.array_equal(match.source_of, perm)

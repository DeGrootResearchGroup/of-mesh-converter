"""Integration tests: full pipeline from a synthetic CGNS file to a
written OpenFOAM case directory.

These exercise reader + sanitise + mesh builder + writer together.
They are the v1 stand-in for the "tier 1 self-roundtrip" validation
described in CLAUDE.md — the of-optical-radiation bit-for-bit roundtrip is a
separate, partner-dependent harness that lives outside CI.
"""

from __future__ import annotations

import numpy as np

from of_mesh_converter import pipeline

from .cgns_fixture import hex_box_cgns


def _read_lines(path):
    return path.read_text().splitlines()


def test_end_to_end_hex_box_writes_expected_files(tmp_path):
    cgns_path = tmp_path / "case.cgns"
    case_dir = tmp_path / "case_out"
    hex_box_cgns(cgns_path, nx=4, ny=2, nz=2)

    case, report = pipeline.convert(cgns_path, case_dir)

    # Required polyMesh files.
    pm = case_dir / "constant" / "polyMesh"
    for f in ("points", "faces", "owner", "neighbour", "boundary"):
        assert (pm / f).exists(), f"missing polyMesh/{f}"

    # Required 0/ field files.
    for f in ("U", "k", "epsilon"):
        assert (case_dir / "0" / f).exists(), f"missing 0/{f}"

    # System stubs.
    for f in ("controlDict", "fvSchemes", "fvSolution", "postProcess.dict"):
        assert (case_dir / "system" / f).exists(), f"missing system/{f}"

    # The report is non-empty and mentions all three patches.
    assert "inlet" in report
    assert "outlet" in report
    assert "walls" in report


def test_end_to_end_face_topology_is_consistent(tmp_path):
    """The face-based mesh produced by the pipeline must be a closed
    manifold: total face count = internal + sum(patch face counts),
    and every internal face has owner < neighbour."""
    cgns_path = tmp_path / "case.cgns"
    case_dir = tmp_path / "case_out"
    truth = hex_box_cgns(cgns_path, nx=3, ny=2, nz=2)

    case, _ = pipeline.convert(cgns_path, case_dir)

    mesh = case.mesh
    boundary_total = sum(p.n_faces for p in mesh.patches)
    assert mesh.n_faces == mesh.n_internal_faces + boundary_total
    # Owner < neighbour on every internal face.
    assert np.all(mesh.owner[: mesh.n_internal_faces] < mesh.neighbour)
    # Patches retain CGNS order.
    assert [p.name for p in mesh.patches] == truth["patches"]


def test_end_to_end_fields_reproduce_to_floating_point(tmp_path):
    """The fields we put in the CGNS file must come back out
    cell-by-cell to floating-point accuracy after the round-trip."""
    cgns_path = tmp_path / "case.cgns"
    case_dir = tmp_path / "case_out"
    truth = hex_box_cgns(
        cgns_path, nx=2, ny=2, nz=2,
        Ux=0.42, k_val=0.005, eps_val=0.0001,
    )
    case, _ = pipeline.convert(cgns_path, case_dir)
    n_cells = truth["n_cells"]
    np.testing.assert_allclose(case.vector_fields["U"][:, 0], 0.42)
    np.testing.assert_allclose(case.vector_fields["U"][:, 1], 0.0)
    np.testing.assert_allclose(case.vector_fields["U"][:, 2], 0.0)
    np.testing.assert_allclose(case.scalar_fields["k"], 0.005)
    np.testing.assert_allclose(case.scalar_fields["epsilon"], 0.0001)
    # The 0/U file contains the right number of cells.
    u_lines = _read_lines(case_dir / "0" / "U")
    assert any(line.strip() == str(n_cells) for line in u_lines)


def test_end_to_end_negative_k_is_clipped(tmp_path):
    """If a CGNS export has k <= 0 in some cells (common near walls
    in Fluent k-epsilon solutions), the converter floors those to
    the turbulence floor and reports the count in the sanity report.
    """
    cgns_path = tmp_path / "case.cgns"
    case_dir = tmp_path / "case_out"
    hex_box_cgns(cgns_path, nx=2, ny=1, nz=1, k_val=-1.0)
    case, report = pipeline.convert(cgns_path, case_dir)
    assert np.all(case.scalar_fields["k"] > 0)
    assert "k       clipped" in report


def test_pipeline_writes_sanity_report_file(tmp_path):
    cgns_path = tmp_path / "case.cgns"
    case_dir = tmp_path / "case_out"
    hex_box_cgns(cgns_path)
    pipeline.convert(cgns_path, case_dir)
    assert (case_dir / "conversion_report.txt").exists()


def test_pipeline_sanitises_patch_names(tmp_path):
    cgns_path = tmp_path / "case.cgns"
    case_dir = tmp_path / "case_out"
    hex_box_cgns(
        cgns_path,
        nx=2, ny=1, nz=1,
        inlet_name="interior-fluid",  # CGNS-legal, OF-illegal
        outlet_name="outflow:01",
        walls_name="wall surfaces",
    )
    case, report = pipeline.convert(cgns_path, case_dir)
    patch_names = [p.name for p in case.mesh.patches]
    assert "interior_fluid" in patch_names
    assert "outflow_01" in patch_names
    assert "wall_surfaces" in patch_names
    # Report mentions the sanitisation.
    assert "Patch-name sanitisation" in report


def test_cli_invocation(tmp_path):
    """Exercise the argparse entry point so the CLI doesn't bitrot."""
    from of_mesh_converter.__main__ import main

    cgns_path = tmp_path / "case.cgns"
    case_dir = tmp_path / "case_out"
    hex_box_cgns(cgns_path)
    rc = main([str(cgns_path), str(case_dir)])
    assert rc == 0
    assert (case_dir / "constant" / "polyMesh" / "boundary").exists()


def test_cli_reports_unsupported_polyhedral_clearly(tmp_path):
    """The CLI must exit non-zero with a clear message when handed a
    polyhedral CGNS file."""
    from of_mesh_converter.__main__ import main
    from of_mesh_converter._cgns_hdf5 import (
        CGNSNode,
        char_array,
        write_cgns_file,
    )

    p = tmp_path / "poly.cgns"
    zone = CGNSNode(
        name="Zone", label="Zone_t",
        data=np.array([[8, 0, 0]], dtype=np.int32),
        children=[
            CGNSNode(name="ZoneType", label="ZoneType_t",
                     data=char_array("Unstructured")),
            CGNSNode(
                name="GridCoordinates", label="GridCoordinates_t",
                children=[
                    CGNSNode(name="CoordinateX", label="DataArray_t",
                             data=np.zeros(8, dtype=np.float64)),
                    CGNSNode(name="CoordinateY", label="DataArray_t",
                             data=np.zeros(8, dtype=np.float64)),
                    CGNSNode(name="CoordinateZ", label="DataArray_t",
                             data=np.zeros(8, dtype=np.float64)),
                ],
            ),
            CGNSNode(
                name="Faces", label="Elements_t",
                data=np.array([22, 0], dtype=np.int32),  # NGON_n
                children=[
                    CGNSNode(name="ElementRange", label="IndexRange_t",
                             data=np.array([1, 1], dtype=np.int32)),
                    CGNSNode(name="ElementConnectivity", label="DataArray_t",
                             data=np.array([0], dtype=np.int32)),
                ],
            ),
        ],
    )
    base = CGNSNode(name="Base", label="CGNSBase_t",
                    data=np.array([3, 3], dtype=np.int32), children=[zone])
    root = CGNSNode(name="CGNS", label="Root Node of HDF5 File", children=[base])
    write_cgns_file(p, root)

    rc = main([str(p), str(tmp_path / "out")])
    assert rc == 1

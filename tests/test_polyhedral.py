"""Polyhedral (NGON_n / NFACE_n) input, in the layout Fluent writes.

The fixture is the hex box of ``hex_box_cgns`` stored as polyhedra, so
the standard-element path gives an independent answer to compare
against.
"""

from __future__ import annotations

import numpy as np
import pytest

from of_mesh_converter import cgns_reader, pipeline
from of_mesh_converter.mesh_builder import (
    BoundaryFaceGroup,
    PolyhedralCells,
    build_polyhedral_mesh,
)

from .cgns_fixture import hex_box_cgns, poly_box_cgns


def _face_area_vectors(mesh):
    pts = mesh.points
    out = np.zeros((mesh.n_faces, 3))
    centres = np.zeros((mesh.n_faces, 3))
    for i, f in enumerate(mesh.faces):
        p = pts[f]
        centres[i] = p.mean(axis=0)
        for j in range(1, len(f) - 1):
            out[i] += 0.5 * np.cross(p[j] - p[0], p[j + 1] - p[0])
    return out, centres


def _cell_centres(truth):
    return truth["points"][truth["hex_conn"]].mean(axis=1)


def test_poly_mesh_topology_and_orientation(tmp_path):
    p = tmp_path / "poly.cgns"
    truth = poly_box_cgns(p, nx=3, ny=2, nz=2)
    case, _ = pipeline.convert(p, tmp_path / "out")
    mesh = case.mesh

    assert mesh.n_cells == truth["n_cells"]
    assert mesh.n_faces == truth["n_faces"]
    assert mesh.n_internal_faces == truth["n_internal_faces"]

    owner = mesh.owner[: mesh.n_internal_faces]
    assert np.all(owner < mesh.neighbour)
    order = np.lexsort((mesh.neighbour, owner))
    assert np.array_equal(order, np.arange(order.size))

    # Every face normal points out of its owner (and into the neighbour).
    sf, cf = _face_area_vectors(mesh)
    cc = _cell_centres(truth)
    assert np.all(np.einsum("ij,ij->i", sf, cf - cc[mesh.owner]) > 0)
    n_int = mesh.n_internal_faces
    assert np.all(
        np.einsum("ij,ij->i", sf[:n_int], cc[mesh.neighbour] - cf[:n_int]) > 0
    )


def test_poly_mesh_matches_standard_element_path(tmp_path):
    """Same box via HEXA_8 and via NGON/NFACE: identical cells, so the
    same owner/neighbour pairs on the same (unordered) faces."""
    hp = tmp_path / "hex.cgns"
    pp = tmp_path / "poly.cgns"
    hex_box_cgns(hp, nx=3, ny=2, nz=2)
    poly_box_cgns(pp, nx=3, ny=2, nz=2)
    hmesh, _ = pipeline.convert(hp, tmp_path / "hex_out")
    pmesh, _ = pipeline.convert(pp, tmp_path / "poly_out")
    hmesh, pmesh = hmesh.mesh, pmesh.mesh

    def internal(m):
        return sorted(
            (tuple(sorted(f)), int(o), int(n))
            for f, o, n in zip(m.faces, m.owner, m.neighbour)
        )

    assert internal(hmesh) == internal(pmesh)
    assert hmesh.n_faces == pmesh.n_faces


def test_poly_fields_taken_at_cell_element_ids(tmp_path):
    p = tmp_path / "poly.cgns"
    truth = poly_box_cgns(p)
    case, report = pipeline.convert(p, tmp_path / "out")
    np.testing.assert_array_equal(case.scalar_fields["G"], truth["G"])
    assert "one value per element" in report
    assert (tmp_path / "out" / "0" / "G").exists()


def test_poly_patch_names_drop_fluent_type_suffix(tmp_path):
    p = tmp_path / "poly.cgns"
    truth = poly_box_cgns(p)
    case, _ = pipeline.convert(p, tmp_path / "out")
    sizes = {pt.name: pt.n_faces for pt in case.mesh.patches}
    assert sizes == truth["patch_sizes"]
    types = {pt.name: pt.type for pt in case.mesh.patches}
    assert types["walls"] == "wall"
    assert types["inlet"] == "patch"


def test_poly_legacy_cgns3_layout_gives_same_mesh(tmp_path):
    a = tmp_path / "v4.cgns"
    b = tmp_path / "v3.cgns"
    poly_box_cgns(a)
    poly_box_cgns(b, legacy=True)
    ma = pipeline.convert(a, tmp_path / "a")[0].mesh
    mb = pipeline.convert(b, tmp_path / "b")[0].mesh
    assert np.array_equal(ma.face_offsets, mb.face_offsets)
    assert np.array_equal(ma.face_vertices, mb.face_vertices)
    assert np.array_equal(ma.owner, mb.owner)
    assert np.array_equal(ma.neighbour, mb.neighbour)


def test_poly_interface_bcs_are_reported(tmp_path):
    p = tmp_path / "poly.cgns"
    poly_box_cgns(p, bc_suffixes={"inlet": "velocity-inlet",
                                  "outlet": "interface", "walls": "wall"})
    _, report = pipeline.convert(p, tmp_path / "out")
    assert "mesh interfaces" in report
    assert "outlet-interface" in report


def test_poly_cli_writes_case(tmp_path):
    from of_mesh_converter.__main__ import main

    p = tmp_path / "poly.cgns"
    poly_box_cgns(p)
    assert main([str(p), str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "constant" / "polyMesh" / "faces").exists()


# Builder-level error handling on a two-cell stack of unit cubes.

def _two_cube_input():
    pts = np.array(
        [[x, y, z] for z in (0, 1, 2) for y in (0, 1) for x in (0, 1)],
        dtype=np.float64,
    )

    def q(*v):
        return list(v)

    faces = [
        q(0, 2, 3, 1),      # 0: bottom of cell 0 (outward -z)
        q(0, 1, 5, 4),      # 1: y=0, cell 0
        q(1, 3, 7, 5),      # 2: x=1, cell 0
        q(3, 2, 6, 7),      # 3: y=1, cell 0
        q(2, 0, 4, 6),      # 4: x=0, cell 0
        q(4, 5, 7, 6),      # 5: z=1, shared, outward from cell 0
        q(4, 5, 9, 8),      # 6: y=0, cell 1
        q(5, 7, 11, 9),     # 7: x=1, cell 1
        q(7, 6, 10, 11),    # 8: y=1, cell 1
        q(6, 4, 8, 10),     # 9: x=0, cell 1
        q(8, 9, 11, 10),    # 10: top of cell 1
    ]
    face_offsets = np.zeros(len(faces) + 1, dtype=np.int64)
    np.cumsum([len(f) for f in faces], out=face_offsets[1:])
    cells = [[1, 2, 3, 4, 5, 6], [-6, 7, 8, 9, 10, 11]]
    cell_offsets = np.array([0, 6, 12], dtype=np.int64)
    poly = PolyhedralCells(
        face_offsets=face_offsets,
        face_vertices=np.array([v for f in faces for v in f], dtype=np.int64),
        cell_offsets=cell_offsets,
        cell_faces=np.array([x for c in cells for x in c], dtype=np.int64),
    )
    groups = [BoundaryFaceGroup(name="all", type="wall",
                                face_ids=np.array([0, 1, 2, 3, 4, 6, 7, 8, 9, 10]))]
    return pts, poly, groups


def test_poly_builder_two_cubes():
    pts, poly, groups = _two_cube_input()
    mesh = build_polyhedral_mesh(pts, poly, groups)
    assert mesh.n_internal_faces == 1
    assert (mesh.owner[0], mesh.neighbour[0]) == (0, 1)
    assert mesh.faces[0] == [4, 5, 7, 6]


def test_poly_builder_flips_face_pointing_into_owner():
    pts, poly, groups = _two_cube_input()
    # Store the shared face the other way round: outward from cell 1.
    poly.face_vertices[poly.face_offsets[5]:poly.face_offsets[6]] = [4, 6, 7, 5]
    poly.cell_faces[5] = -6
    poly.cell_faces[6] = 6
    mesh = build_polyhedral_mesh(pts, poly, groups)
    assert mesh.faces[0] == [4, 5, 7, 6]


def test_poly_builder_rejects_same_sign_internal_face():
    pts, poly, groups = _two_cube_input()
    poly.cell_faces[6] = 6
    with pytest.raises(ValueError, match="same sign"):
        build_polyhedral_mesh(pts, poly, groups)


def test_poly_builder_rejects_unassigned_boundary_face():
    pts, poly, groups = _two_cube_input()
    groups[0].face_ids = groups[0].face_ids[:-1]
    with pytest.raises(ValueError, match="not assigned"):
        build_polyhedral_mesh(pts, poly, groups)


def test_poly_builder_rejects_internal_face_in_patch():
    pts, poly, groups = _two_cube_input()
    groups[0].face_ids = np.append(groups[0].face_ids, 5)
    with pytest.raises(ValueError, match="not a boundary face"):
        build_polyhedral_mesh(pts, poly, groups)


def test_reader_rejects_field_of_unexpected_length(tmp_path):
    from of_mesh_converter._cgns_hdf5 import read_cgns_file, write_cgns_file

    p = tmp_path / "poly.cgns"
    poly_box_cgns(p)
    root = read_cgns_file(p)
    zone = root.children_of_label("CGNSBase_t")[0].children_of_label("Zone_t")[0]
    fs = zone.children_of_label("FlowSolution_t")[0]
    fs.child("Incident_Radiation").data = np.zeros(5)
    write_cgns_file(p, root)
    with pytest.raises(ValueError, match="does not match"):
        cgns_reader.read_cgns(p)

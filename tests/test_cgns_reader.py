"""Reader-side tests: read a synthetic CGNS file we just wrote, and
check that the structured-CGNS data we put in comes back out
correctly."""

from __future__ import annotations

import numpy as np

from of_mesh_converter import cgns_reader

from .cgns_fixture import hex_box_cgns


def test_reads_hex_box_points(tmp_path):
    p = tmp_path / "hex.cgns"
    truth = hex_box_cgns(p, nx=4, ny=2, nz=2)
    points, cell_blocks, _bg, _flow = cgns_reader.read_cgns(p)
    assert points.shape == (truth["n_points"], 3)
    np.testing.assert_allclose(points, truth["points"])


def test_reads_hex_box_cell_connectivity(tmp_path):
    p = tmp_path / "hex.cgns"
    truth = hex_box_cgns(p, nx=2, ny=1, nz=1)
    _pts, cell_blocks, _bg, _flow = cgns_reader.read_cgns(p)
    assert len(cell_blocks) == 1
    block = cell_blocks[0]
    assert block.element_type == "HEXA_8"
    assert block.connectivity.shape == (truth["n_cells"], 8)


def test_reads_hex_box_boundary_groups(tmp_path):
    p = tmp_path / "hex.cgns"
    truth = hex_box_cgns(p, nx=2, ny=1, nz=1)
    _pts, _cb, groups, _flow = cgns_reader.read_cgns(p)
    names = [g.name for g in groups]
    # The CGNS BCs name family inlet/outlet/walls; the reader
    # surfaces those as-is (sanitisation runs in the pipeline).
    assert "inlet" in names
    assert "outlet" in names
    assert "walls" in names
    # Walls have BCWall type → reader tags them as "wall".
    walls = next(g for g in groups if g.name == "walls")
    assert walls.type == "wall"


def test_reads_flow_solution(tmp_path):
    p = tmp_path / "hex.cgns"
    truth = hex_box_cgns(p, nx=2, ny=1, nz=1, Ux=0.7, k_val=0.02, eps_val=0.005)
    _pts, _cb, _bg, flow = cgns_reader.read_cgns(p)
    scalars = flow["scalars"]
    vectors = flow["vectors"]
    assert "U" in vectors
    assert vectors["U"].shape == (truth["n_cells"], 3)
    np.testing.assert_allclose(vectors["U"][:, 0], 0.7)
    np.testing.assert_allclose(vectors["U"][:, 1], 0.0)
    np.testing.assert_allclose(vectors["U"][:, 2], 0.0)
    np.testing.assert_allclose(scalars["k"], 0.02)
    np.testing.assert_allclose(scalars["epsilon"], 0.005)


def test_reader_rejects_polyhedral_cells(tmp_path):
    """NGON_n / NFACE_n is documented as a v1 non-feature; the reader
    must emit a clear error rather than silently producing a wrong
    mesh."""
    import pytest

    from of_mesh_converter._cgns_hdf5 import CGNSNode, char_array, write_cgns_file

    p = tmp_path / "poly.cgns"
    # Minimal NGON_n stub. The reader should refuse it before touching
    # the connectivity payload.
    ngon = CGNSNode(
        name="Faces",
        label="Elements_t",
        data=np.array([22, 0], dtype=np.int32),  # NGON_n code is 22
        children=[
            CGNSNode(
                name="ElementRange",
                label="IndexRange_t",
                data=np.array([1, 1], dtype=np.int32),
            ),
            CGNSNode(
                name="ElementConnectivity",
                label="DataArray_t",
                data=np.array([0], dtype=np.int32),
            ),
        ],
    )
    zone = CGNSNode(
        name="Zone1", label="Zone_t",
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
            ngon,
        ],
    )
    base = CGNSNode(name="Base", label="CGNSBase_t",
                    data=np.array([3, 3], dtype=np.int32), children=[zone])
    root = CGNSNode(name="CGNS", label="Root Node of HDF5 File", children=[base])
    write_cgns_file(p, root)

    with pytest.raises(NotImplementedError, match="polyhedral"):
        cgns_reader.read_cgns(p)

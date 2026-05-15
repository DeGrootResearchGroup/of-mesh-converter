"""Smoke tests for the OpenFOAM writer.

The writer is mostly textual layout — exhaustive content checks
belong in the integration test (where we run the full pipeline and
inspect the rendered files). Here we just verify it produces every
required file with non-empty content and that the dimension sets are
correct."""

from __future__ import annotations

import numpy as np
import pytest

from of_mesh_converter import foam_writer
from of_mesh_converter.mesh_builder import (
    BoundaryFaceGroup,
    CellBlock,
    build_mesh,
)
from of_mesh_converter.mesh_ir import CaseData


@pytest.fixture
def single_hex_case():
    pts = np.array(
        [
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ],
        dtype=np.float64,
    )
    cells = CellBlock(
        element_type="HEXA_8",
        connectivity=np.array([[0, 1, 2, 3, 4, 5, 6, 7]]),
    )
    bg = [BoundaryFaceGroup(name="walls", type="wall", faces=[
        (0, 3, 2, 1), (4, 5, 6, 7),
        (0, 1, 5, 4), (3, 2, 6, 7),
        (0, 4, 7, 3), (1, 5, 6, 2),
    ])]
    mesh = build_mesh(pts, [cells], bg)
    return CaseData(
        mesh=mesh,
        scalar_fields={
            "k":       np.array([1e-3]),
            "epsilon": np.array([1e-4]),
        },
        vector_fields={"U": np.array([[0.5, 0.0, 0.0]])},
    )


def test_writer_produces_polyMesh(tmp_path, single_hex_case):
    foam_writer.write_case(single_hex_case, tmp_path)
    pm = tmp_path / "constant" / "polyMesh"
    for fname in ("points", "faces", "owner", "neighbour", "boundary"):
        assert (pm / fname).exists(), f"missing {fname}"
        assert (pm / fname).stat().st_size > 0


def test_writer_produces_field_files(tmp_path, single_hex_case):
    foam_writer.write_case(single_hex_case, tmp_path)
    for fname in ("U", "k", "epsilon"):
        f = tmp_path / "0" / fname
        assert f.exists(), f"missing 0/{fname}"
        text = f.read_text()
        assert "internalField" in text
        assert "boundaryField" in text


def test_writer_dimensions_correct(tmp_path, single_hex_case):
    foam_writer.write_case(single_hex_case, tmp_path)
    u_text = (tmp_path / "0" / "U").read_text()
    k_text = (tmp_path / "0" / "k").read_text()
    eps_text = (tmp_path / "0" / "epsilon").read_text()
    assert "dimensions      [0 1 -1 0 0 0 0];" in u_text
    assert "dimensions      [0 2 -2 0 0 0 0];" in k_text
    assert "dimensions      [0 2 -3 0 0 0 0];" in eps_text


def test_writer_produces_system_stubs(tmp_path, single_hex_case):
    foam_writer.write_case(single_hex_case, tmp_path)
    for fname in ("controlDict", "fvSchemes", "fvSolution", "postProcess.dict"):
        f = tmp_path / "system" / fname
        assert f.exists(), f"missing system/{fname}"


def test_postProcess_dict_lists_patches(tmp_path, single_hex_case):
    foam_writer.write_case(single_hex_case, tmp_path)
    pp = (tmp_path / "system" / "postProcess.dict").read_text()
    assert "walls" in pp
    assert "radiationDose" in pp


def test_writer_rejects_field_with_wrong_shape(tmp_path, single_hex_case):
    single_hex_case.scalar_fields["k"] = np.array([1e-3, 2e-3])
    with pytest.raises(ValueError, match="mesh has 1 cells"):
        foam_writer.write_case(single_hex_case, tmp_path)


def test_boundary_file_marks_wall_with_inGroups(tmp_path, single_hex_case):
    foam_writer.write_case(single_hex_case, tmp_path)
    b = (tmp_path / "constant" / "polyMesh" / "boundary").read_text()
    assert "inGroups        List<word> 1(wall);" in b

"""Tutorial 01: synthesise a hex-box CGNS file and run the converter.

Run from this directory with ``python run.py``. Produces
``case_out.cgns`` (the synthetic input) and ``case_out/`` (the
OpenFOAM case the converter wrote). Both are gitignored.

Tutorials are not run by CI; the integration test suite under
``tests/test_integration.py`` covers the same code path with a
similar fixture.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Reach over into ``tests/cgns_fixture.py`` for the synthetic-CGNS
# builder. The tutorial deliberately reuses the test fixture so that
# users learning the tool see the same on-disk layout the test suite
# exercises.
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tests.cgns_fixture import hex_box_cgns  # noqa: E402

from of_mesh_converter import pipeline  # noqa: E402


def main() -> None:
    cgns_path = HERE / "case_out.cgns"
    case_dir = HERE / "case_out"

    print(f"Writing synthetic CGNS file:  {cgns_path}")
    truth = hex_box_cgns(
        cgns_path,
        nx=4, ny=2, nz=2,
        Lx=1.0, Ly=0.1, Lz=0.1,
        Ux=0.5, k_val=1.0e-3, eps_val=1.0e-4,
        inlet_name="inlet",
        outlet_name="outlet",
        walls_name="walls",
    )
    print(f"  {truth['n_cells']} cells, {truth['n_points']} points")

    print(f"\nRunning of-mesh-converter -> {case_dir}")
    _case, report = pipeline.convert(cgns_path, case_dir)

    print()
    print(report)
    print(f"OpenFOAM case written to: {case_dir}")
    print("Inspect constant/polyMesh/, 0/, and system/postProcess.dict.")


if __name__ == "__main__":
    main()

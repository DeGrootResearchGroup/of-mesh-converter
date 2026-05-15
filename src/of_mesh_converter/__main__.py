"""CLI entry point: ``python -m of_mesh_converter`` or
``of-mesh-converter`` (installed script).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .pipeline import convert


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="of-mesh-converter",
        description=(
            "Convert a finished Fluent flow solution exported to CGNS "
            "into the minimum OpenFOAM case skeleton needed by the "
            "of-optical-radiation radiationDose Lagrangian tracker."
        ),
    )
    p.add_argument(
        "cgns_file",
        type=Path,
        help="Input CGNS file (single base, single unstructured zone).",
    )
    p.add_argument(
        "case_dir",
        type=Path,
        help="Output OpenFOAM case directory (created if it does not exist).",
    )
    p.add_argument(
        "--no-report-file",
        action="store_true",
        help="Print the sanity report to stdout but do not save "
             "conversion_report.txt in the case directory.",
    )
    p.add_argument(
        "--version",
        action="version",
        version=f"of-mesh-converter {__version__}",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        _, report = convert(
            args.cgns_file,
            args.case_dir,
            write_report=not args.no_report_file,
        )
    except (NotImplementedError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())

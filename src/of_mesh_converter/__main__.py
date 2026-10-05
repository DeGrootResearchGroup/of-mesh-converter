"""CLI entry point: ``python -m of_mesh_converter`` or
``of-mesh-converter`` (installed script).

    of-mesh-converter CASE.cgns CASE_DIR            convert to a new case
    of-mesh-converter transfer CASE.cgns TARGET     copy fields into an
                                                    existing case, same mesh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .pipeline import convert
from .transfer import DEFAULT_REL_TOL, transfer


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="of-mesh-converter",
        description=(
            "Convert a finished Fluent flow solution exported to CGNS "
            "into the minimum OpenFOAM case skeleton needed by the "
            "of-optical-radiation radiationDose Lagrangian tracker."
        ),
        epilog=(
            "To copy the CGNS file's fields into an existing OpenFOAM case "
            "on the same mesh instead, run: of-mesh-converter transfer -h"
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


def _build_transfer_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="of-mesh-converter transfer",
        description=(
            "Copy fields from a CGNS file into an existing OpenFOAM case "
            "whose mesh has the same cells (in any order), matching cells "
            "by centre. Nothing is interpolated: the command refuses to "
            "run if the meshes differ. The target needs its cell centres: "
            "run `foamPostProcess -func writeCellCentres -time <time>` "
            "in it first."
        ),
    )
    p.add_argument("cgns_file", type=Path, help="Input CGNS file.")
    p.add_argument("target_case", type=Path, help="Existing OpenFOAM case.")
    p.add_argument(
        "--time",
        help="Time directory to write into (default: the latest).",
    )
    p.add_argument(
        "--fields",
        help="Comma-separated OpenFOAM field names to copy, e.g. G or U,k "
             "(default: every field in the CGNS file).",
    )
    p.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_REL_TOL,
        help="Largest accepted centre distance, as a fraction of the cell "
             f"size V^(1/3) (default {DEFAULT_REL_TOL:g}).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace field files that already exist, keeping their "
             "boundaryField.",
    )
    p.add_argument(
        "--no-report-file",
        action="store_true",
        help="Do not save transfer_report.txt in the target case.",
    )
    return p


def _transfer_main(argv: list[str]) -> int:
    args = _build_transfer_parser().parse_args(argv)
    fields = None
    if args.fields:
        fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    try:
        _, report = transfer(
            args.cgns_file,
            args.target_case,
            time=args.time,
            fields=fields,
            rel_tol=args.tolerance,
            overwrite=args.overwrite,
            write_report=not args.no_report_file,
        )
    except (NotImplementedError, ValueError, FileNotFoundError,
            FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(report)
    return 0


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "transfer":
        return _transfer_main(argv[1:])
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

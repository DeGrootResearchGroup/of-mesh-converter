"""Put fields from a CGNS file into an existing OpenFOAM case on the
same mesh.

The use case: the Fluent solve ran on a mesh exported from an
OpenFOAM case (or vice versa), and the user wants Fluent's fields —
typically the DO fluence rate — next to the OpenFOAM flow so the dose
tracker sees exactly one thing change. Fluent renumbers cells, so the
fields cannot be copied by index; they are copied by matching cell
centres.

This is a copy, not an interpolation, and it refuses to run unless
the two meshes have the same cells: every target cell must have a
source cell within ``rel_tol`` times that cell's size (``V**(1/3)``),
and the pairing must be one to one. Meshes that differ are a job for
OpenFOAM's ``mapFields``, which interpolates.

The target's cell centres come from its ``C`` field (``foamPostProcess
-func writeCellCentres``; ASCII or binary), so the converter needs no
polyMesh reader.
The source's centres are computed here with OpenFOAM's definitions
(``geometry.py``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np

from . import foam_writer, geometry, pipeline

# Patch types whose field patch type must equal the patch type.
CONSTRAINT_PATCH_TYPES = frozenset({
    "cyclic", "cyclicAMI", "cyclicRepeatAMI", "cyclicSlip",
    "nonConformalCyclic", "nonConformalError", "nonConformalProcessorCyclic",
    "processor", "processorCyclic",
    "empty", "wedge", "symmetry", "symmetryPlane",
})

DEFAULT_REL_TOL = 1.0e-3


@dataclass
class CellMatch:
    """``source_of[i]`` is the source cell matched to target cell ``i``;
    ``distance[i]`` its centre distance, ``rel_distance[i]`` that over
    the source cell's size."""

    source_of: np.ndarray
    distance: np.ndarray
    rel_distance: np.ndarray


def match_cells(
    source_centres: np.ndarray,
    source_sizes: np.ndarray,
    target_centres: np.ndarray,
    rel_tol: float = DEFAULT_REL_TOL,
) -> CellMatch:
    """Pair every target cell with the source cell at the same place.

    Raises ``ValueError`` unless the cell counts agree, every target
    cell has a source cell within ``rel_tol * size``, and no source
    cell is used twice.

    Nearest neighbours are found by hashing centres into cubes of side
    ``q`` no smaller than the largest tolerance, and searching each
    target's cube and its 26 neighbours: every source centre within
    ``q`` of a target is in one of them.
    """
    src = np.asarray(source_centres, dtype=np.float64)
    tgt = np.asarray(target_centres, dtype=np.float64)
    if src.shape[0] != tgt.shape[0]:
        raise ValueError(
            f"Source has {src.shape[0]} cells and target {tgt.shape[0]}: "
            "not the same mesh. Use OpenFOAM's mapFields to interpolate "
            "between different meshes."
        )
    tol = rel_tol * np.asarray(source_sizes, dtype=np.float64)

    lo = np.minimum(src.min(axis=0), tgt.min(axis=0))
    extent = float(np.max(np.maximum(src.max(axis=0), tgt.max(axis=0)) - lo))
    q = max(float(tol.max()), extent / 2**20, np.finfo(float).tiny)
    radix = int(np.floor(extent / q)) + 3
    if radix**3 >= 2**63:
        raise ValueError("Cell-centre hash overflow; tolerance too small for the extent")

    def keys(k):
        return (k[:, 0] * radix + k[:, 1]) * radix + k[:, 2]

    ks = np.floor((src - lo) / q).astype(np.int64) + 1
    kt = np.floor((tgt - lo) / q).astype(np.int64) + 1
    src_keys = keys(ks)
    order = np.argsort(src_keys, kind="stable")
    sorted_keys = src_keys[order]
    multiplicity = int(np.unique(sorted_keys, return_counts=True)[1].max())

    best = np.full(tgt.shape[0], -1, dtype=np.int64)
    best_d = np.full(tgt.shape[0], np.inf)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                k = keys(kt + np.array([dx, dy, dz]))
                left = np.searchsorted(sorted_keys, k, side="left")
                right = np.searchsorted(sorted_keys, k, side="right")
                for m in range(multiplicity):
                    pos = left + m
                    hit = np.flatnonzero(pos < right)
                    if hit.size == 0:
                        break
                    cand = order[pos[hit]]
                    d = np.linalg.norm(src[cand] - tgt[hit], axis=1)
                    better = d < best_d[hit]
                    best[hit[better]] = cand[better]
                    best_d[hit[better]] = d[better]

    found = best >= 0
    rel = np.full(tgt.shape[0], np.inf)
    rel[found] = best_d[found] / (tol[best[found]] / rel_tol)
    bad = ~found | (rel > rel_tol)
    if np.any(bad):
        worst = float(np.max(rel[found])) if np.any(found) else float("inf")
        raise ValueError(
            f"{int(bad.sum())} of {tgt.shape[0]} target cells have no source "
            f"cell within {rel_tol:g} of the cell size (worst match: "
            f"{worst:.3g}). The meshes are not the same; use OpenFOAM's "
            "mapFields to interpolate between different meshes."
        )
    if np.unique(best).size != best.size:
        raise ValueError(
            "Cell match is not one to one: some source cell is the nearest "
            "to two target cells. The meshes are not the same."
        )
    return CellMatch(source_of=best, distance=best_d, rel_distance=rel)


# ---------------------------------------------------------------- OF I/O ---

_FORMAT = re.compile(r"\bformat\s+(\w+)\s*;")


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def read_internal_vector_field(path: Path) -> np.ndarray:
    """Read the nonuniform internal field of a ``volVectorField``
    written in ASCII or binary format."""
    raw = Path(path).read_bytes()
    m = re.search(rb"internalField\s+nonuniform\s+List<vector>\s*(\d+)\s*\(", raw)
    if m is None:
        raise ValueError(f"{path}: no nonuniform List<vector> internalField")
    header = raw[:m.start()].decode("latin-1")
    n = int(m.group(1))
    start = m.end()
    fmt = _FORMAT.search(header)
    if fmt and fmt.group(1) == "binary":
        arch = re.search(r'\barch\s+"([^"]*)"', header)
        arch = arch.group(1) if arch else "LSB;label=32;scalar=64"
        bits = re.search(r"scalar=(\d+)", arch)
        size = int(bits.group(1)) // 8 if bits else 8
        dtype = np.dtype(f"{'>' if 'MSB' in arch else '<'}f{size}")
        if len(raw) < start + 3 * n * size:
            raise ValueError(f"{path}: truncated binary list")
        values = np.frombuffer(raw, dtype=dtype, count=3 * n, offset=start)
        return values.astype(np.float64).reshape(n, 3)
    text = raw[start:].decode("latin-1")
    end = text.find("\n)") if text.startswith("\n") else text.find("))") + 1
    body = text[:end].replace("(", " ").replace(")", " ")
    values = np.array(body.split(), dtype=np.float64)
    if values.size != 3 * n:
        raise ValueError(f"{path}: expected {n} vectors, read {values.size / 3:g}")
    return values.reshape(n, 3)


def read_boundary_patches(case: Path) -> list[tuple[str, str]]:
    """``(name, type)`` for each patch in ``constant/polyMesh/boundary``."""
    path = Path(case) / "constant" / "polyMesh" / "boundary"
    text = _strip_comments(path.read_bytes().decode("latin-1"))
    text = re.sub(r"FoamFile\s*\{.*?\}", "", text, count=1, flags=re.S)
    patches = []
    for name, body in re.findall(r"([^\s{}();]+)\s*\{([^{}]*)\}", text):
        t = re.search(r"\btype\s+(\w+)\s*;", body)
        if t is None:
            raise ValueError(f"{path}: patch {name!r} has no type")
        patches.append((name, t.group(1)))
    if not patches:
        raise ValueError(f"{path}: no patches found")
    return patches


def _existing_boundary_field(path: Path) -> list[str] | None:
    """The ``boundaryField { ... }`` block of an existing ASCII field
    file, as lines, or None if it cannot be reused."""
    raw = path.read_bytes()
    fmt = _FORMAT.search(raw[:2048].decode("latin-1"))
    if fmt and fmt.group(1) != "ascii":
        return None
    text = raw.decode("latin-1")
    start = text.find("boundaryField")
    if start < 0:
        return None
    i = text.index("{", start)
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[start:j + 1].splitlines()
    return None


def _default_boundary_field(patches: list[tuple[str, str]]) -> list[str]:
    out = ["boundaryField", "{"]
    for name, ptype in patches:
        bc = ptype if ptype in CONSTRAINT_PATCH_TYPES else "zeroGradient"
        out += [f"    {name}", "    {", f"        type            {bc};", "    }"]
    out.append("}")
    return out


def _time_dirs(case: Path) -> list[tuple[float, Path]]:
    out = []
    for d in Path(case).iterdir():
        if d.is_dir():
            try:
                out.append((float(d.name), d))
            except ValueError:
                pass
    return sorted(out)


def _find_cell_centres(case: Path, time_dir: Path) -> Path:
    candidates = [time_dir / "C"] + [d / "C" for _, d in _time_dirs(case)]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        f"No cell-centre field C in {case}. Write it with\n"
        f"    foamPostProcess -func writeCellCentres -time {time_dir.name}\n"
        "and run transfer again."
    )


# -------------------------------------------------------------- transfer ---

def transfer(
    cgns_path: Path | str,
    target_case: Path | str,
    *,
    time: str | None = None,
    fields: list[str] | None = None,
    rel_tol: float = DEFAULT_REL_TOL,
    overwrite: bool = False,
    write_report: bool = True,
) -> tuple[CellMatch, str]:
    """Copy fields from ``cgns_path`` into ``target_case/<time>/``.

    ``time`` defaults to the latest time directory; ``fields`` to every
    field the CGNS file provides. An existing field file is replaced
    only with ``overwrite``, and then keeps its ``boundaryField``; a new
    file gets ``zeroGradient`` on every patch except constraint patches
    (cyclic, non-conformal, processor, empty, ...), which get their own
    type. Returns the cell match and a report.
    """
    target_case = Path(target_case)
    times = _time_dirs(target_case)
    if time is None:
        if not times:
            raise FileNotFoundError(f"{target_case} has no time directories")
        time_dir = times[-1][1]
    else:
        time_dir = target_case / time
        if not time_dir.is_dir():
            raise FileNotFoundError(f"{time_dir} does not exist")

    loaded = pipeline.load(cgns_path)
    case = loaded.case
    available = {**case.scalar_fields, **case.vector_fields}
    names = list(available) if fields is None else list(fields)
    unknown = [n for n in names if n not in available]
    if unknown or not names:
        raise ValueError(
            f"Field(s) {unknown or names} not in {cgns_path}; it provides "
            f"{sorted(available) or 'none'}."
        )
    existing = [n for n in names if (time_dir / n).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"{time_dir} already has {', '.join(existing)}; pass "
            "--overwrite to replace (the existing boundaryField is kept)."
        )

    c_path = _find_cell_centres(target_case, time_dir)
    target_centres = read_internal_vector_field(c_path)
    source_centres, volumes = geometry.cell_centres_and_volumes(case.mesh)
    match = match_cells(source_centres, np.cbrt(volumes), target_centres, rel_tol)

    patches = read_boundary_patches(target_case)
    written = []
    for name in names:
        out = time_dir / name
        boundary = None
        how = "zeroGradient, constraint patches their own type"
        if out.exists():
            boundary = _existing_boundary_field(out)
            if boundary is not None:
                how = "kept from the file it replaced"
        if boundary is None:
            boundary = _default_boundary_field(patches)
        values = available[name][match.source_of]
        foam_writer.write_field_file(
            out, name, values, boundary, location=time_dir.name
        )
        written.append((name, values, how))

    report = _format_report(
        Path(cgns_path), target_case, time_dir, c_path, match, rel_tol,
        written, case.notes,
    )
    if write_report:
        (target_case / "transfer_report.txt").write_text(report)
    return match, report


def _format_report(cgns_path, target_case, time_dir, c_path, match, rel_tol,
                   written, notes) -> str:
    buf = StringIO()
    p = lambda s="": buf.write(s + "\n")  # noqa: E731
    p("=" * 72)
    p("of-mesh-converter transfer — report")
    p("=" * 72)
    p(f"  source        : {cgns_path}")
    p(f"  target        : {target_case}, time {time_dir.name}")
    p(f"  target centres: {c_path}")
    p(f"  cells matched : {match.source_of.size}, one to one")
    p(f"  max distance  : {match.distance.max():.3g} m "
      f"({match.rel_distance.max():.3g} of the cell size; tolerance {rel_tol:g})")
    p("")
    p("Fields written")
    for name, values, how in written:
        v = np.asarray(values)
        if v.ndim == 1:
            p(f"  {name:10s} min={v.min():+.6g} max={v.max():+.6g} mean={v.mean():+.6g}")
        else:
            mag = np.linalg.norm(v, axis=1)
            p(f"  {name:10s} |.| min={mag.min():.6g} max={mag.max():.6g} mean={mag.mean():.6g}")
        p(f"             boundaryField: {how}")
    if notes:
        p("")
        p("Notes from reading the CGNS file")
        for note in notes:
            p(f"  - {note}")
    p("=" * 72)
    return buf.getvalue()

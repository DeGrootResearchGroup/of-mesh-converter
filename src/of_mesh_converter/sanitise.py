"""Sanitise CGNS-side names and field values into shapes OpenFOAM
will accept.

Two responsibilities:

1. Patch names. Fluent zone names like ``interior-fluid`` or
   ``inlet:01`` contain characters OpenFOAM's `polyMesh/boundary`
   parser rejects. We replace anything outside ``[A-Za-z0-9_]`` with
   ``_``, prefix leading-digit names with ``p_``, and collapse runs
   of underscores. The mapping is returned so the sanity report can
   show users what got renamed (they need to reconcile against their
   dose-dict patch list).

2. Turbulence fields. Fluent occasionally emits ``k <= 0`` or
   ``epsilon <= 0`` in near-wall cells. Feeding either into the
   discrete random walk dispersion model in radiationDose produces
   NaN trajectories. We clip to 1e-12 and return a count so the
   sanity report can flag how many cells were touched.
"""

from __future__ import annotations

import re

import numpy as np

# Lowest positive value we'll accept for k or epsilon. Picked to be
# well below any physically reasonable wall-adjacent value but still
# strictly positive so the DRW model's sqrt(k) and 1/epsilon factors
# stay finite.
TURBULENCE_FLOOR: float = 1.0e-12

_INVALID_CHARS = re.compile(r"[^A-Za-z0-9_]")
_LEADING_DIGIT = re.compile(r"^\d")
_MULTI_UNDERSCORE = re.compile(r"_+")


def sanitise_patch_name(name: str) -> str:
    """Map an arbitrary CGNS zone/family name to a valid OF identifier."""
    if not name:
        return "unnamed"
    s = _INVALID_CHARS.sub("_", name)
    s = _MULTI_UNDERSCORE.sub("_", s)
    s = s.strip("_")
    if not s:
        return "unnamed"
    if _LEADING_DIGIT.match(s):
        s = "p_" + s
    return s


def sanitise_patch_names(names: list[str]) -> tuple[list[str], dict[str, str]]:
    """Sanitise a list of patch names; resolve collisions by appending
    ``_2``, ``_3``, ... to the second and later occurrences.

    Returns ``(sanitised_names, original_to_final_mapping)``. The
    mapping is shown in the sanity report so the user can match their
    dose-dict patch list to what actually got written.
    """
    out: list[str] = []
    seen: dict[str, int] = {}
    mapping: dict[str, str] = {}
    for original in names:
        base = sanitise_patch_name(original)
        if base not in seen:
            seen[base] = 1
            final = base
        else:
            seen[base] += 1
            final = f"{base}_{seen[base]}"
        out.append(final)
        mapping[original] = final
    return out, mapping


def clip_nonpositive(
    values: np.ndarray, floor: float = TURBULENCE_FLOOR
) -> tuple[np.ndarray, int]:
    """Clip values ``<= 0`` to ``floor``. Returns (clipped, n_clipped)."""
    arr = np.asarray(values, dtype=np.float64)
    mask = arr <= 0.0
    n_clipped = int(mask.sum())
    if n_clipped:
        arr = arr.copy()
        arr[mask] = floor
    return arr, n_clipped

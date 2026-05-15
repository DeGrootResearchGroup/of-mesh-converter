"""Dictionary-driven mapping from CGNS quantity names to OpenFOAM
field names.

The CGNS Standard Interface Data Structures (SIDS) document specifies
canonical names for common quantities. OpenFOAM uses its own
conventions. This module is the one place that knows the translation.

We support exactly the names the dose tracker consumes — velocity,
turbulence kinetic energy, dissipation rate, fluence rate. Adding a
new field is a one-line entry; nothing else in the converter needs
to change. That is the design pattern lifted from the dead
`cgnsToFromFoam` `CgnsToFoamDictionary` and reimplemented fresh.
"""

from __future__ import annotations

# Scalar fields: CGNS canonical name -> OpenFOAM field name.
SCALAR_FIELD_MAP: dict[str, str] = {
    "TurbulentEnergyKinetic": "k",
    "TurbulentDissipationRate": "epsilon",
    "FluenceRate": "G",  # not CGNS-canonical; allow direct match
    "G": "G",
}

# Vector fields: a vector field in CGNS is stored as three scalar
# arrays with these component-name patterns. The map carries
# (X-component name, Y-component name, Z-component name) for each
# OpenFOAM vector field. The reader assembles the three into an
# (n_cells, 3) array.
VECTOR_FIELD_MAP: dict[str, tuple[str, str, str]] = {
    "U": ("VelocityX", "VelocityY", "VelocityZ"),
}


# OpenFOAM dimension sets for the fields we emit. Used by the writer
# to put the right `dimensions [...];` line at the top of each field
# file. Order: [M L T Theta N I Cd]  (kg, m, s, K, mol, A, cd).
OF_DIMENSIONS: dict[str, tuple[int, int, int, int, int, int, int]] = {
    "U":       (0, 1, -1, 0, 0, 0, 0),   # m/s
    "k":       (0, 2, -2, 0, 0, 0, 0),   # m^2/s^2
    "epsilon": (0, 2, -3, 0, 0, 0, 0),   # m^2/s^3
    "G":       (1, 0, -3, 0, 0, 0, 0),   # W/m^2 = kg/s^3
}


def known_scalar_fields() -> tuple[str, ...]:
    return tuple(sorted(set(SCALAR_FIELD_MAP.values())))


def known_vector_fields() -> tuple[str, ...]:
    return tuple(sorted(VECTOR_FIELD_MAP.keys()))


def of_scalar_name(cgns_name: str) -> str | None:
    """Return the OpenFOAM scalar field name for a CGNS quantity, or
    None if the CGNS name is not one we map."""
    return SCALAR_FIELD_MAP.get(cgns_name)


def of_vector_components(of_name: str) -> tuple[str, str, str]:
    """Return the (X, Y, Z) CGNS component names that compose the
    given OpenFOAM vector field."""
    if of_name not in VECTOR_FIELD_MAP:
        raise KeyError(f"No vector mapping for OpenFOAM field {of_name!r}")
    return VECTOR_FIELD_MAP[of_name]

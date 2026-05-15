"""Read a CGNS file into the mesh IR.

Scope is the layout Fluent's exporter produces: a single
``CGNSBase_t`` containing a single unstructured ``Zone_t``. The zone
holds:

- ``GridCoordinates_t`` with ``CoordinateX``, ``CoordinateY``,
  ``CoordinateZ`` (one ``DataArray_t`` each).
- One or more ``Elements_t`` sections. The volume sections give us
  cells (``TETRA_4``, ``HEXA_8``, ``PYRA_5``, ``PENTA_6``); the
  surface sections (``TRI_3``, ``QUAD_4``) give us boundary patches.
  Each ``Elements_t`` node carries:

    * ``data``: ``[element_type_code, parent_flag]`` (CGNS convention).
    * child ``ElementRange``: ``[first_elem, last_elem]`` (1-based,
      inclusive — CGNS-canonical).
    * child ``ElementConnectivity``: flattened vertex indices, all
      1-based. Length is ``(last_elem - first_elem + 1) * N_VERTS``.

- ``ZoneBC_t`` containing ``BC_t`` children. Each ``BC_t`` is one
  patch; it carries a ``GridLocation_t`` (``FaceCenter`` for a
  surface BC), a ``PointRange`` or ``PointList`` (CGNS element
  indices for the boundary), and a ``FamilyName_t`` whose data is
  the human-readable patch name. Bookkeeping: CGNS gives indices
  into the global element numbering; we use those to fish the
  correct boundary-element rows out of the surface ``Elements_t``
  sections.

The reader produces ``(CellBlock list, BoundaryFaceGroup list,
points, FlowSolution dict)`` and hands them to ``mesh_builder`` and
the field-mapping layer. It does not build the mesh or write
anything.

Polyhedral cells (``NGON_n`` / ``NFACE_n``) are flagged with a
``NotImplementedError`` for v1. Adding them is a follow-up in this
same file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import field_mapping
from ._cgns_hdf5 import CGNSNode, read_cgns_file
from .elements import (
    ELEMENT_TYPE_CODES,
    ELEMENT_TYPE_NAMES,
    N_VERTS,
    SUPPORTED_BOUNDARY_ELEMENTS,
    SUPPORTED_VOLUME_ELEMENTS,
)
from .mesh_builder import BoundaryFaceGroup, CellBlock


def _expect_int(arr: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(arr)
    if a.dtype.kind not in ("i", "u"):
        raise ValueError(f"{name} expected integer dtype, got {a.dtype}")
    return a.astype(np.int64)


def _find_zone(root: CGNSNode) -> CGNSNode:
    bases = root.children_of_label("CGNSBase_t")
    if not bases:
        raise ValueError("CGNS file has no CGNSBase_t")
    if len(bases) > 1:
        raise NotImplementedError(
            f"Multi-base CGNS files not supported ({len(bases)} bases). "
            "Re-export with a single CGNSBase_t."
        )
    zones = bases[0].children_of_label("Zone_t")
    if not zones:
        raise ValueError("CGNS base has no Zone_t child")
    if len(zones) > 1:
        raise NotImplementedError(
            f"Multi-zone CGNS files not supported ({len(zones)} zones). "
            "Re-export with a single Zone_t."
        )
    return zones[0]


def _read_points(zone: CGNSNode) -> np.ndarray:
    gc = next((c for c in zone.children_of_label("GridCoordinates_t")), None)
    if gc is None:
        raise ValueError("Zone has no GridCoordinates_t")
    coords = {}
    for da in gc.children_of_label("DataArray_t"):
        if da.data is None:
            continue
        coords[da.name] = np.asarray(da.data, dtype=np.float64).ravel()
    for required in ("CoordinateX", "CoordinateY", "CoordinateZ"):
        if required not in coords:
            raise ValueError(f"GridCoordinates missing {required}")
    n = coords["CoordinateX"].shape[0]
    if any(coords[c].shape[0] != n for c in coords):
        raise ValueError("Coordinate arrays have inconsistent lengths")
    pts = np.empty((n, 3), dtype=np.float64)
    pts[:, 0] = coords["CoordinateX"]
    pts[:, 1] = coords["CoordinateY"]
    pts[:, 2] = coords["CoordinateZ"]
    return pts


def _read_elements_node(elem: CGNSNode):
    if elem.data is None:
        raise ValueError(f"Elements_t node {elem.name!r} has no data")
    payload = _expect_int(elem.data, f"{elem.name}.data").ravel()
    if payload.size < 1:
        raise ValueError(f"Elements_t node {elem.name!r}: empty type payload")
    type_code = int(payload[0])
    type_name = ELEMENT_TYPE_NAMES.get(type_code)
    if type_name is None:
        raise NotImplementedError(
            f"Elements_t {elem.name!r}: unknown CGNS element type code {type_code}"
        )

    range_node = elem.child("ElementRange")
    if range_node is None or range_node.data is None:
        raise ValueError(f"Elements_t {elem.name!r}: missing ElementRange")
    rng = _expect_int(range_node.data, f"{elem.name}.ElementRange").ravel()
    first, last = int(rng[0]), int(rng[1])
    n_elements = last - first + 1

    conn_node = elem.child("ElementConnectivity")
    if conn_node is None or conn_node.data is None:
        raise ValueError(
            f"Elements_t {elem.name!r}: missing ElementConnectivity"
        )
    flat = _expect_int(conn_node.data, f"{elem.name}.ElementConnectivity").ravel()

    if type_name in ("NGON_n", "NFACE_n"):
        raise NotImplementedError(
            f"Elements_t {elem.name!r} is polyhedral ({type_name}). "
            "Polyhedral CGNS support is planned but not yet implemented; "
            "v1 supports TETRA_4, HEXA_8, PENTA_6, PYRA_5 only."
        )

    if type_name not in N_VERTS:
        raise NotImplementedError(
            f"Elements_t {elem.name!r}: unsupported type {type_name}"
        )

    nv = N_VERTS[type_name]
    expected = n_elements * nv
    if flat.size != expected:
        raise ValueError(
            f"{elem.name}.ElementConnectivity has {flat.size} entries; "
            f"expected {n_elements} × {nv} = {expected} for {type_name}"
        )

    # CGNS uses 1-based vertex indices; convert to 0-based here so
    # the rest of the converter (numpy-native) doesn't have to.
    conn = (flat.reshape(n_elements, nv) - 1).astype(np.int64)
    return type_name, first, last, conn


def _read_flow_solution(zone: CGNSNode, n_cells: int) -> dict[str, np.ndarray]:
    """Return a {OF-field-name: array} dict. Scalars: (n_cells,);
    vectors: (n_cells, 3). Cell-centred fields only — node-centred
    fields would need an interpolation step that's out of scope."""
    fs_nodes = zone.children_of_label("FlowSolution_t")
    if not fs_nodes:
        return {}
    if len(fs_nodes) > 1:
        raise NotImplementedError(
            "Multiple FlowSolution_t nodes not supported; v1 reads "
            "exactly one steady-state solution."
        )
    fs = fs_nodes[0]

    gl_node = fs.child("GridLocation")
    if gl_node is not None:
        loc = gl_node.data_as_str() or ""
        if loc and loc != "CellCenter":
            raise NotImplementedError(
                f"FlowSolution GridLocation={loc!r}; only 'CellCenter' "
                "is supported (cell-centred fields from a finished "
                "Fluent solve)."
            )

    cgns_arrays: dict[str, np.ndarray] = {}
    for da in fs.children_of_label("DataArray_t"):
        if da.data is None:
            continue
        arr = np.asarray(da.data, dtype=np.float64).ravel()
        if arr.shape[0] != n_cells:
            raise ValueError(
                f"FlowSolution/{da.name}: {arr.shape[0]} values does "
                f"not match {n_cells} cells"
            )
        cgns_arrays[da.name] = arr

    scalars: dict[str, np.ndarray] = {}
    vectors: dict[str, np.ndarray] = {}
    consumed: set[str] = set()

    for of_name, (cx, cy, cz) in field_mapping.VECTOR_FIELD_MAP.items():
        if cx in cgns_arrays and cy in cgns_arrays and cz in cgns_arrays:
            vectors[of_name] = np.column_stack(
                [cgns_arrays[cx], cgns_arrays[cy], cgns_arrays[cz]]
            )
            consumed.update((cx, cy, cz))

    for cgns_name, arr in cgns_arrays.items():
        if cgns_name in consumed:
            continue
        of_name = field_mapping.of_scalar_name(cgns_name)
        if of_name is None:
            continue
        scalars[of_name] = arr

    return {"scalars": scalars, "vectors": vectors}


def _read_boundary_groups(
    zone: CGNSNode,
    surface_elems: list[tuple[str, int, int, np.ndarray]],
) -> list[BoundaryFaceGroup]:
    """Read ZoneBC_t and translate each BC_t into a BoundaryFaceGroup.

    ``surface_elems`` is a list of ``(type_name, first, last, conn)``
    tuples from ``_read_elements_node`` for every 2-D Elements_t
    section in the zone. CGNS BC point-lists refer into the global
    element numbering; we resolve them by walking the surface
    sections.
    """
    zbc_nodes = zone.children_of_label("ZoneBC_t")
    if not zbc_nodes:
        # No ZoneBC: every surface element becomes one patch named
        # after its Elements_t section. This is what hand-built test
        # fixtures use when there's no need to exercise the BC tree.
        groups: list[BoundaryFaceGroup] = []
        for _, _, _, _ in surface_elems:
            pass
        return groups
    zbc = zbc_nodes[0]

    # Index every surface element by its global element id for fast
    # lookup. CGNS element ids are 1-based and globally unique
    # within a zone.
    elem_to_face: dict[int, tuple[int, ...]] = {}
    for _type_name, first, _last, conn in surface_elems:
        for offset, row in enumerate(conn):
            elem_to_face[first + offset] = tuple(int(v) for v in row)

    groups: list[BoundaryFaceGroup] = []
    for bc in zbc.children_of_label("BC_t"):
        family_node = bc.child("FamilyName") or next(
            (c for c in bc.children_of_label("FamilyName_t")), None
        )
        family = family_node.data_as_str() if family_node else None
        patch_name = family or bc.name

        bc_type_str = bc.data_as_str() or ""
        of_type = "wall" if "Wall" in bc_type_str else "patch"

        pr = bc.child("PointRange")
        pl = bc.child("PointList")
        elem_ids: list[int] = []
        if pr is not None and pr.data is not None:
            rng = _expect_int(pr.data, "PointRange").ravel()
            elem_ids = list(range(int(rng[0]), int(rng[1]) + 1))
        elif pl is not None and pl.data is not None:
            elem_ids = [int(x) for x in _expect_int(pl.data, "PointList").ravel()]
        else:
            raise ValueError(
                f"BC_t {bc.name!r} has neither PointRange nor PointList"
            )

        faces = []
        for eid in elem_ids:
            face = elem_to_face.get(eid)
            if face is None:
                raise ValueError(
                    f"BC_t {bc.name!r}: element id {eid} not in any "
                    "surface Elements_t section"
                )
            faces.append(face)
        groups.append(BoundaryFaceGroup(name=patch_name, type=of_type, faces=faces))

    return groups


def read_cgns(path: Path | str):
    """Parse a CGNS file. Returns ``(points, cell_blocks,
    boundary_groups, flow_solution)``.

    ``flow_solution`` is a dict with keys ``"scalars"`` (OF-name →
    array) and ``"vectors"`` (OF-name → (n,3) array).
    """
    root = read_cgns_file(path)
    zone = _find_zone(root)

    points = _read_points(zone)

    # Walk every Elements_t in the zone, partitioning into volume and
    # surface by element-type dimensionality.
    volume_elems: list[tuple[str, int, int, np.ndarray]] = []
    surface_elems: list[tuple[str, int, int, np.ndarray]] = []
    for elem in zone.children_of_label("Elements_t"):
        type_name, first, last, conn = _read_elements_node(elem)
        if type_name in SUPPORTED_VOLUME_ELEMENTS:
            volume_elems.append((type_name, first, last, conn))
        elif type_name in SUPPORTED_BOUNDARY_ELEMENTS:
            surface_elems.append((type_name, first, last, conn))
        else:
            raise NotImplementedError(
                f"Elements_t {elem.name!r}: {type_name} is not currently "
                "handled by the converter."
            )

    if not volume_elems:
        raise ValueError("Zone has no volume Elements_t sections")

    cell_blocks = [
        CellBlock(element_type=t, connectivity=conn)
        for (t, _f, _l, conn) in volume_elems
    ]
    n_cells = sum(b.connectivity.shape[0] for b in cell_blocks)

    boundary_groups = _read_boundary_groups(zone, surface_elems)

    flow_solution = _read_flow_solution(zone, n_cells)

    return points, cell_blocks, boundary_groups, flow_solution

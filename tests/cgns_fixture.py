"""Helpers for synthesising CGNS files in tests and tutorials.

We do not depend on ``pycgns``: the converter reads CGNS via h5py
following the CGNS-HDF5 file mapping, so we can write the same
layout here from scratch. This keeps the test suite installable
with just ``numpy`` + ``h5py`` and dodges pycgns' platform quirks.

The functions below cover the cases the test suite needs:

- ``hex_box_cgns`` — a structured hex box meshed into HEXA_8 cells,
  with three boundary patches (inlet/outlet/walls) and uniform
  velocity / k / epsilon fields. The canonical fixture; everything
  else is built on top of it.
- ``tetra_box_cgns`` — same domain, but each hex split into 6 tets;
  exercises the TETRA_4 face decomposition.
- ``mixed_box_cgns`` — one hex layer + one wedge (PENTA_6) layer +
  one pyramid (PYRA_5) layer, to exercise the mixed-element path.

Each returns ``(points, cell_blocks, boundary_groups, scalars, vectors)``
as Python-side data alongside writing the CGNS file. Tests use the
side data to assert the converter round-tripped the inputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from of_mesh_converter._cgns_hdf5 import CGNSNode, char_array, write_cgns_file
from of_mesh_converter.elements import ELEMENT_TYPE_CODES, N_VERTS


def _data_array(name: str, arr: np.ndarray) -> CGNSNode:
    return CGNSNode(name=name, label="DataArray_t", data=arr)


def _elements_node(
    name: str,
    type_name: str,
    first: int,
    last: int,
    connectivity_1based: np.ndarray,
) -> CGNSNode:
    type_code = ELEMENT_TYPE_CODES[type_name]
    head = np.array([type_code, 0], dtype=np.int32)
    return CGNSNode(
        name=name,
        label="Elements_t",
        data=head,
        children=[
            CGNSNode(
                name="ElementRange",
                label="IndexRange_t",
                data=np.array([first, last], dtype=np.int32),
            ),
            CGNSNode(
                name="ElementConnectivity",
                label="DataArray_t",
                data=connectivity_1based.astype(np.int32).ravel(),
            ),
        ],
    )


def _bc_node(name: str, family_name: str, point_range: tuple[int, int],
             bc_type: str = "BCWall") -> CGNSNode:
    """One BC_t. ``point_range`` is the (first, last) global element
    id of the surface elements in this patch."""
    return CGNSNode(
        name=name,
        label="BC_t",
        data=char_array(bc_type),
        children=[
            CGNSNode(
                name="GridLocation",
                label="GridLocation_t",
                data=char_array("FaceCenter"),
            ),
            CGNSNode(
                name="PointRange",
                label="IndexRange_t",
                data=np.array(point_range, dtype=np.int32),
            ),
            CGNSNode(
                name="FamilyName",
                label="FamilyName_t",
                data=char_array(family_name),
            ),
        ],
    )


def _zone(
    name: str,
    points: np.ndarray,
    elements: list[CGNSNode],
    zone_bc: CGNSNode | None,
    flow_solution: CGNSNode | None,
) -> CGNSNode:
    n_pts = points.shape[0]
    n_cells = 0
    for e in elements:
        # Sum cells across volume sections by reading the ElementRange.
        # Surface sections also have a range; we don't try to
        # discriminate here — the test caller passes ``n_cells_total``
        # via the zone dims below.
        rng = e.child("ElementRange").data
        # Volume sections in the test fixtures always come first.
        # The zone.data dimensions are filled in by caller in cases
        # where the bookkeeping matters.
        pass

    # CGNS Zone_t data is shape (1, 3): [n_vertices, n_cells, n_boundary_vertices].
    # We leave n_boundary_vertices=0 (sorted-or-not is reader-side concern).
    children = [
        CGNSNode(name="ZoneType", label="ZoneType_t",
                 data=char_array("Unstructured")),
        CGNSNode(
            name="GridCoordinates",
            label="GridCoordinates_t",
            children=[
                _data_array("CoordinateX", points[:, 0].astype(np.float64)),
                _data_array("CoordinateY", points[:, 1].astype(np.float64)),
                _data_array("CoordinateZ", points[:, 2].astype(np.float64)),
            ],
        ),
    ]
    children.extend(elements)
    if zone_bc is not None:
        children.append(zone_bc)
    if flow_solution is not None:
        children.append(flow_solution)

    return CGNSNode(
        name=name,
        label="Zone_t",
        data=np.array([[n_pts, n_cells, 0]], dtype=np.int32),
        children=children,
    )


def _root(base_name: str, zone: CGNSNode) -> CGNSNode:
    base = CGNSNode(
        name=base_name,
        label="CGNSBase_t",
        data=np.array([3, 3], dtype=np.int32),  # cell_dim=3, phys_dim=3
        children=[zone],
    )
    return CGNSNode(name="CGNS", label="Root Node of HDF5 File",
                    children=[base])


# ---------------------------------------------------------------------------
# Concrete fixtures

def _hex_block_points(nx: int, ny: int, nz: int,
                      Lx: float, Ly: float, Lz: float) -> np.ndarray:
    xs = np.linspace(0.0, Lx, nx + 1)
    ys = np.linspace(0.0, Ly, ny + 1)
    zs = np.linspace(0.0, Lz, nz + 1)
    pts = np.empty(((nx + 1) * (ny + 1) * (nz + 1), 3), dtype=np.float64)
    idx = 0
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                pts[idx] = (xs[i], ys[j], zs[k])
                idx += 1
    return pts


def _point_index(i, j, k, nx, ny) -> int:
    return i + (nx + 1) * (j + (ny + 1) * k)


def _hex_cells(nx: int, ny: int, nz: int) -> np.ndarray:
    """CGNS HEXA_8 connectivity (1-based) for a structured nx*ny*nz box.

    Vertex numbering matches the CGNS SIDS HEXA_8 convention: nodes
    1-4 form the bottom face (z lowest) traversed CCW from below,
    nodes 5-8 the top face directly above 1-4.
    """
    cells = np.empty((nx * ny * nz, 8), dtype=np.int32)
    c = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                # 0-based corner indices.
                p = [
                    _point_index(i,     j,     k,     nx, ny),
                    _point_index(i + 1, j,     k,     nx, ny),
                    _point_index(i + 1, j + 1, k,     nx, ny),
                    _point_index(i,     j + 1, k,     nx, ny),
                    _point_index(i,     j,     k + 1, nx, ny),
                    _point_index(i + 1, j,     k + 1, nx, ny),
                    _point_index(i + 1, j + 1, k + 1, nx, ny),
                    _point_index(i,     j + 1, k + 1, nx, ny),
                ]
                cells[c] = [x + 1 for x in p]  # 1-based for CGNS
                c += 1
    return cells


def _box_boundary_quads(nx: int, ny: int, nz: int):
    """Return six lists of QUAD_4 face vertex tuples (1-based) for
    the six external faces of a structured hex box.

    Order within each face: matches the outward-normal CGNS QUAD_4
    convention. We use this only to record the boundary 2-D
    elements in the CGNS file; the converter's mesh builder will
    match each one against a cell face by sorted-vertex key.
    """
    xlow:  list[tuple[int, int, int, int]] = []
    xhigh: list[tuple[int, int, int, int]] = []
    ylow:  list[tuple[int, int, int, int]] = []
    yhigh: list[tuple[int, int, int, int]] = []
    zlow:  list[tuple[int, int, int, int]] = []
    zhigh: list[tuple[int, int, int, int]] = []

    # x = 0 face
    for k in range(nz):
        for j in range(ny):
            xlow.append(tuple(p + 1 for p in (
                _point_index(0, j,     k,     nx, ny),
                _point_index(0, j,     k + 1, nx, ny),
                _point_index(0, j + 1, k + 1, nx, ny),
                _point_index(0, j + 1, k,     nx, ny),
            )))
    # x = Lx face
    for k in range(nz):
        for j in range(ny):
            xhigh.append(tuple(p + 1 for p in (
                _point_index(nx, j,     k,     nx, ny),
                _point_index(nx, j + 1, k,     nx, ny),
                _point_index(nx, j + 1, k + 1, nx, ny),
                _point_index(nx, j,     k + 1, nx, ny),
            )))
    # y = 0 face
    for k in range(nz):
        for i in range(nx):
            ylow.append(tuple(p + 1 for p in (
                _point_index(i,     0, k,     nx, ny),
                _point_index(i + 1, 0, k,     nx, ny),
                _point_index(i + 1, 0, k + 1, nx, ny),
                _point_index(i,     0, k + 1, nx, ny),
            )))
    # y = Ly face
    for k in range(nz):
        for i in range(nx):
            yhigh.append(tuple(p + 1 for p in (
                _point_index(i,     ny, k,     nx, ny),
                _point_index(i,     ny, k + 1, nx, ny),
                _point_index(i + 1, ny, k + 1, nx, ny),
                _point_index(i + 1, ny, k,     nx, ny),
            )))
    # z = 0 face
    for j in range(ny):
        for i in range(nx):
            zlow.append(tuple(p + 1 for p in (
                _point_index(i,     j,     0, nx, ny),
                _point_index(i,     j + 1, 0, nx, ny),
                _point_index(i + 1, j + 1, 0, nx, ny),
                _point_index(i + 1, j,     0, nx, ny),
            )))
    # z = Lz face
    for j in range(ny):
        for i in range(nx):
            zhigh.append(tuple(p + 1 for p in (
                _point_index(i,     j,     nz, nx, ny),
                _point_index(i + 1, j,     nz, nx, ny),
                _point_index(i + 1, j + 1, nz, nx, ny),
                _point_index(i,     j + 1, nz, nx, ny),
            )))
    return xlow, xhigh, ylow, yhigh, zlow, zhigh


def hex_box_cgns(
    out_path: Path,
    *,
    nx: int = 4,
    ny: int = 2,
    nz: int = 2,
    Lx: float = 1.0,
    Ly: float = 0.1,
    Lz: float = 0.1,
    Ux: float = 0.5,
    k_val: float = 1.0e-3,
    eps_val: float = 1.0e-4,
    inlet_name: str = "inlet",
    outlet_name: str = "outlet",
    walls_name: str = "walls",
) -> dict:
    """Write a structured hex-box CGNS file. Returns a dict of the
    Python-side ground truth (points, cell connectivity, boundary
    face lists, field arrays) so the tests can assert exact match."""
    points = _hex_block_points(nx, ny, nz, Lx, Ly, Lz)
    hex_conn = _hex_cells(nx, ny, nz)
    n_cells = hex_conn.shape[0]

    # Volume Elements_t section: ids 1..n_cells
    vol = _elements_node("Cells", "HEXA_8", 1, n_cells, hex_conn)

    # Boundary Elements_t sections (QUAD_4), one per patch.
    xlow, xhigh, ylow, yhigh, zlow, zhigh = _box_boundary_quads(nx, ny, nz)

    # Group: inlet = x=0 face. outlet = x=Lx face. walls = the other 4.
    walls = ylow + yhigh + zlow + zhigh
    inlet = xlow
    outlet = xhigh

    next_id = n_cells + 1
    inlet_range = (next_id, next_id + len(inlet) - 1)
    next_id += len(inlet)
    outlet_range = (next_id, next_id + len(outlet) - 1)
    next_id += len(outlet)
    walls_range = (next_id, next_id + len(walls) - 1)
    next_id += len(walls)

    surf_inlet = _elements_node(
        "InletFaces", "QUAD_4",
        inlet_range[0], inlet_range[1],
        np.array(inlet, dtype=np.int32),
    )
    surf_outlet = _elements_node(
        "OutletFaces", "QUAD_4",
        outlet_range[0], outlet_range[1],
        np.array(outlet, dtype=np.int32),
    )
    surf_walls = _elements_node(
        "WallFaces", "QUAD_4",
        walls_range[0], walls_range[1],
        np.array(walls, dtype=np.int32),
    )

    zone_bc = CGNSNode(
        name="ZoneBC",
        label="ZoneBC_t",
        children=[
            _bc_node("BCInlet",  inlet_name,  inlet_range,  bc_type="BCInflow"),
            _bc_node("BCOutlet", outlet_name, outlet_range, bc_type="BCOutflow"),
            _bc_node("BCWalls",  walls_name,  walls_range,  bc_type="BCWall"),
        ],
    )

    flow = CGNSNode(
        name="FlowSolution",
        label="FlowSolution_t",
        children=[
            CGNSNode(name="GridLocation", label="GridLocation_t",
                     data=char_array("CellCenter")),
            _data_array("VelocityX", np.full(n_cells, Ux, dtype=np.float64)),
            _data_array("VelocityY", np.zeros(n_cells, dtype=np.float64)),
            _data_array("VelocityZ", np.zeros(n_cells, dtype=np.float64)),
            _data_array("TurbulentEnergyKinetic",
                        np.full(n_cells, k_val, dtype=np.float64)),
            _data_array("TurbulentDissipationRate",
                        np.full(n_cells, eps_val, dtype=np.float64)),
        ],
    )

    zone = _zone(
        "Zone1",
        points,
        elements=[vol, surf_inlet, surf_outlet, surf_walls],
        zone_bc=zone_bc,
        flow_solution=flow,
    )
    # Patch Zone_t.data with the right cell count.
    zone.data = np.array([[points.shape[0], n_cells, 0]], dtype=np.int32)

    root = _root("Base", zone)
    write_cgns_file(out_path, root)

    return {
        "n_cells": n_cells,
        "n_points": points.shape[0],
        "points": points,
        "inlet_faces": inlet,
        "outlet_faces": outlet,
        "walls_faces": walls,
        "patches": [inlet_name, outlet_name, walls_name],
        "Ux": Ux,
        "k_val": k_val,
        "eps_val": eps_val,
    }


def _poly_elements_node(
    name: str,
    type_name: str,
    first: int,
    entries: list[list[int]],
    *,
    legacy: bool,
) -> CGNSNode:
    """One NGON_n / NFACE_n section. ``entries`` are 1-based (signed for
    NFACE_n). ``legacy`` writes the CGNS 3.x layout (each entry
    prefixed by its length, no ElementStartOffset)."""
    last = first + len(entries) - 1
    children = [
        CGNSNode(
            name="ElementRange",
            label="IndexRange_t",
            data=np.array([first, last], dtype=np.int64),
        )
    ]
    if legacy:
        flat = [x for e in entries for x in [len(e), *e]]
    else:
        flat = [x for e in entries for x in e]
        offsets = np.zeros(len(entries) + 1, dtype=np.int64)
        np.cumsum([len(e) for e in entries], out=offsets[1:])
        children.append(
            CGNSNode(name="ElementStartOffset", label="DataArray_t", data=offsets)
        )
    children.append(
        CGNSNode(
            name="ElementConnectivity",
            label="DataArray_t",
            data=np.array(flat, dtype=np.int64),
        )
    )
    return CGNSNode(
        name=name,
        label="Elements_t",
        data=np.array([ELEMENT_TYPE_CODES[type_name], 0], dtype=np.int32),
        children=children,
    )


def poly_box_cgns(
    out_path: Path,
    *,
    nx: int = 3,
    ny: int = 2,
    nz: int = 2,
    Lx: float = 1.0,
    Ly: float = 0.1,
    Lz: float = 0.1,
    legacy: bool = False,
    bc_suffixes: dict[str, str] | None = None,
) -> dict:
    """Write the ``hex_box_cgns`` geometry as Fluent writes a polyhedral
    mesh: NGON_n faces in one ``<zone>-Pg`` section per boundary zone
    plus an interior section, NFACE_n cells in one ``fluid-Ph``
    section, BC_t nodes named ``<zone>-<zone type>`` with PointLists
    and no FamilyName, and FlowSolution arrays sized by the total
    element count and indexed by element id.

    Every third face is stored with reversed vertex order (and its
    NFACE_n signs flipped to match) so the builder's reorientation is
    exercised; boundary faces among them end up referenced with a
    negative sign. Returns the Python-side ground truth."""
    if bc_suffixes is None:
        bc_suffixes = {"inlet": "velocity-inlet", "outlet": "pressure-outlet",
                       "walls": "wall"}
    points = _hex_block_points(nx, ny, nz, Lx, Ly, Lz)
    hex_conn = _hex_cells(nx, ny, nz) - 1
    n_cells = hex_conn.shape[0]

    # Unique faces, oriented outward from the first cell that lists
    # them; ``refs[c]`` is the cell's list of (face, sign).
    from of_mesh_converter.elements import cell_faces

    face_index: dict[tuple[int, ...], int] = {}
    faces: list[tuple[int, ...]] = []
    users: list[list[int]] = []
    refs: list[list[tuple[int, int]]] = []
    for c, row in enumerate(hex_conn):
        cell_refs = []
        for f in cell_faces("HEXA_8", row):
            key = tuple(sorted(f))
            if key not in face_index:
                face_index[key] = len(faces)
                faces.append(f)
                users.append([])
                cell_refs.append((face_index[key], +1))
            else:
                cell_refs.append((face_index[key], -1))
            users[face_index[key]].append(c)
        refs.append(cell_refs)

    flipped = set(range(0, len(faces), 3))
    stored = [
        (f[0], *reversed(f[1:])) if i in flipped else f for i, f in enumerate(faces)
    ]

    def zone_of(i: int) -> str:
        if len(users[i]) == 2:
            return "interior"
        x = points[list(faces[i]), 0]
        if np.allclose(x, 0.0):
            return "inlet"
        if np.allclose(x, Lx):
            return "outlet"
        return "walls"

    zones = ["walls", "inlet", "outlet", "interior"]
    by_zone = {z: [i for i in range(len(faces)) if zone_of(i) == z] for z in zones}

    # Element ids: boundary zones first, then interior, then cells.
    elem_id: dict[int, int] = {}
    elements: list[CGNSNode] = []
    bcs: list[CGNSNode] = []
    next_id = 1
    for z in zones:
        ids = by_zone[z]
        for k, i in enumerate(ids):
            elem_id[i] = next_id + k
        sec_name = "fluid-IF" if z == "interior" else f"{z}-Pg"
        elements.append(_poly_elements_node(
            sec_name, "NGON_n", next_id,
            [[v + 1 for v in stored[i]] for i in ids], legacy=legacy,
        ))
        if z != "interior":
            bc_type = "BCWall" if z == "walls" else (
                "BCInflow" if z == "inlet" else "BCOutflow")
            bcs.append(CGNSNode(
                name=f"{z}-{bc_suffixes[z]}",
                label="BC_t",
                data=char_array(bc_type),
                children=[
                    CGNSNode(name="PointList", label="IndexArray_t",
                             data=np.array([[elem_id[i]] for i in ids],
                                           dtype=np.int64)),
                    CGNSNode(name="GridLocation", label="GridLocation_t",
                             data=char_array("FaceCenter")),
                ],
            ))
        next_id += len(ids)

    cell_first = next_id
    nface = []
    for cell_refs in refs:
        nface.append([
            s * (-1 if i in flipped else 1) * elem_id[i] for i, s in cell_refs
        ])
    elements.append(_poly_elements_node(
        "fluid-Ph", "NFACE_n", cell_first, nface, legacy=legacy,
    ))
    n_elements = cell_first + n_cells - 1

    G = np.arange(1, n_cells + 1, dtype=np.float64) * 10.0
    G_by_elem = np.zeros(n_elements)
    G_by_elem[:len(faces)] = -1.0  # face values Fluent also writes
    G_by_elem[cell_first - 1:] = G

    flow = CGNSNode(
        name="FlowSolution.E:1",
        label="FlowSolution_t",
        children=[
            CGNSNode(name="GridLocation", label="GridLocation_t",
                     data=char_array("CellCenter")),
            _data_array("Incident_Radiation", G_by_elem),
        ],
    )
    zone = _zone("Zone", points, elements=elements,
                 zone_bc=CGNSNode(name="ZoneBC", label="ZoneBC_t", children=bcs),
                 flow_solution=flow)
    zone.data = np.array([[points.shape[0], n_elements, 0]], dtype=np.int64)
    write_cgns_file(out_path, _root("Base", zone))

    return {
        "n_cells": n_cells,
        "n_faces": len(faces),
        "n_internal_faces": len(by_zone["interior"]),
        "hex_conn": hex_conn,
        "points": points,
        "patch_sizes": {z: len(by_zone[z]) for z in zones if z != "interior"},
        "G": G,
    }

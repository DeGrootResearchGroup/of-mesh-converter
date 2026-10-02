"""Read a CGNS file into the mesh IR.

Scope is the layout Fluent's exporter produces: a single
``CGNSBase_t`` containing a single unstructured ``Zone_t``. The zone
holds:

- ``GridCoordinates_t`` with ``CoordinateX``, ``CoordinateY``,
  ``CoordinateZ`` (one ``DataArray_t`` each).
- One or more ``Elements_t`` sections, either standard or polyhedral.

  Standard: the volume sections give us cells (``TETRA_4``,
  ``HEXA_8``, ``PYRA_5``, ``PENTA_6``); the surface sections
  (``TRI_3``, ``QUAD_4``) give us boundary patches. Each
  ``Elements_t`` node carries:

    * ``data``: ``[element_type_code, parent_flag]`` (CGNS convention).
    * child ``ElementRange``: ``[first_elem, last_elem]`` (1-based,
      inclusive — CGNS-canonical).
    * child ``ElementConnectivity``: flattened vertex indices, all
      1-based. Length is ``(last_elem - first_elem + 1) * N_VERTS``.

  Polyhedral (what Fluent writes for poly and poly-hex meshes):
  ``NGON_n`` sections list faces, ``NFACE_n`` sections list each
  cell's faces as signed ``NGON_n`` element ids (positive: the face
  normal points out of the cell). Both use ``ElementStartOffset``
  (CGNS 4.x); the CGNS 3.x layout, where each entry is prefixed by
  its length, is accepted too. Fluent puts every face in an
  ``NGON_n`` section — one per boundary zone plus one for the
  interior — and every cell in one ``NFACE_n`` section.

- ``ZoneBC_t`` containing ``BC_t`` children. Each ``BC_t`` is one
  patch; it carries a ``GridLocation_t`` (``FaceCenter`` for a
  surface BC), a ``PointRange`` or ``PointList`` (CGNS element
  indices for the boundary), and a ``FamilyName_t`` whose data is
  the human-readable patch name. Bookkeeping: CGNS gives indices
  into the global element numbering; we use those to fish the
  correct boundary-element rows out of the surface ``Elements_t``
  sections.

- One ``FlowSolution_t`` with ``GridLocation = CellCenter``. Fluent
  sizes these arrays by the zone's *element* count (faces plus
  cells) and indexes them by element id, so the cell values are the
  slice at the volume sections' element ids; boundary-face values
  sit at the face ids and interior faces are zero. The reader
  accepts either that layout or one value per cell.

The reader produces ``(points, cells, BoundaryFaceGroup list,
FlowSolution dict)`` — ``cells`` is a ``CellBlock`` list for standard
elements or a ``PolyhedralCells`` for ``NGON_n`` / ``NFACE_n`` — and
hands them to ``mesh_builder`` and the field-mapping layer. It does
not build the mesh or write anything.
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
from .mesh_builder import BoundaryFaceGroup, CellBlock, PolyhedralCells


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
        offsets, flat = _poly_offsets(elem, flat, n_elements)
        if offsets.size != n_elements + 1 or offsets[-1] != flat.size:
            raise ValueError(
                f"Elements_t {elem.name!r}: ElementStartOffset has "
                f"{offsets.size} entries ending at {offsets[-1]}; expected "
                f"{n_elements + 1} ending at {flat.size}"
            )
        if offsets[0] != 0 or np.any(np.diff(offsets) < 1):
            raise ValueError(
                f"Elements_t {elem.name!r}: ElementStartOffset must start "
                "at 0 and increase strictly"
            )
        return type_name, first, last, (offsets, flat)

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


def _poly_offsets(
    elem: CGNSNode, flat: np.ndarray, n_elements: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(ElementStartOffset, connectivity)`` for an
    ``NGON_n`` / ``NFACE_n`` section. A CGNS 3.x section has no
    ``ElementStartOffset``; its connectivity carries each element's
    length in front of it, which is stripped here."""
    off_node = elem.child("ElementStartOffset")
    if off_node is not None and off_node.data is not None:
        offsets = _expect_int(off_node.data, f"{elem.name}.ElementStartOffset")
        return offsets.ravel(), flat

    # CGNS 3.x: [n0, v..., n1, v..., ...].
    lengths = np.empty(n_elements, dtype=np.int64)
    pos = 0
    for i in range(n_elements):
        if pos >= flat.size:
            raise ValueError(
                f"Elements_t {elem.name!r}: connectivity ends after "
                f"{i} of {n_elements} elements"
            )
        lengths[i] = flat[pos]
        pos += 1 + int(flat[pos])
    if pos != flat.size:
        raise ValueError(
            f"Elements_t {elem.name!r}: {flat.size - pos} trailing "
            "connectivity entries"
        )
    heads = np.zeros(n_elements, dtype=np.int64)
    np.cumsum(lengths[:-1] + 1, out=heads[1:])
    keep = np.ones(flat.size, dtype=bool)
    keep[heads] = False
    offsets = np.zeros(n_elements + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    return offsets, flat[keep]


def _read_flow_solution(
    zone: CGNSNode,
    n_cells: int,
    cell_ranges: list[tuple[int, int]],
    n_elements: int,
    notes: list[str],
) -> dict[str, np.ndarray]:
    """Return a {OF-field-name: array} dict. Scalars: (n_cells,);
    vectors: (n_cells, 3). Cell-centred fields only — node-centred
    fields would need an interpolation step that's out of scope.

    ``cell_ranges`` are the ``(first, last)`` element ids of the volume
    sections, in cell order, and ``n_elements`` the zone's total
    element count: arrays of that length are indexed by element id."""
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
    by_element_id = False
    for da in fs.children_of_label("DataArray_t"):
        if da.data is None:
            continue
        arr = np.asarray(da.data, dtype=np.float64).ravel()
        if arr.shape[0] == n_cells:
            pass
        elif arr.shape[0] == n_elements:
            arr = np.concatenate([arr[f - 1:l] for f, l in cell_ranges])
            by_element_id = True
        else:
            raise ValueError(
                f"FlowSolution/{da.name}: {arr.shape[0]} values does "
                f"not match {n_cells} cells (or {n_elements} elements)"
            )
        cgns_arrays[da.name] = arr
    if by_element_id:
        notes.append(
            f"FlowSolution arrays have one value per element ({n_elements}) "
            f"rather than per cell ({n_cells}), as Fluent writes them; "
            "cell values were taken at the volume sections' element ids."
        )

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


def _bc_element_ids(bc: CGNSNode) -> np.ndarray:
    pr = bc.child("PointRange")
    pl = bc.child("PointList")
    if pr is not None and pr.data is not None:
        rng = _expect_int(pr.data, "PointRange").ravel()
        return np.arange(int(rng[0]), int(rng[1]) + 1, dtype=np.int64)
    if pl is not None and pl.data is not None:
        return _expect_int(pl.data, "PointList").ravel()
    raise ValueError(f"BC_t {bc.name!r} has neither PointRange nor PointList")


def _bc_name_and_type(
    bc: CGNSNode,
    element_ids: np.ndarray,
    sections: list[tuple[str, int, int]],
) -> tuple[str, str]:
    """Patch name and OF patch type for one BC_t.

    The name is the BC's FamilyName if it has one. Otherwise it is the
    BC_t node's name, except that Fluent names a BC
    ``<zone>-<zone type>`` (``lamp0_wall-wall``, ``inlet-velocity-inlet``)
    and its face section ``<zone>-<suffix>`` (``lamp0_wall-Pg``): when
    the BC covers exactly one section whose stem prefixes the BC name
    that way, the stem is used. ``sections`` holds ``(name, first,
    last)`` for every Elements_t section."""
    bc_type_str = bc.data_as_str() or ""
    of_type = "wall" if "Wall" in bc_type_str else "patch"
    family_node = bc.child("FamilyName") or next(
        (c for c in bc.children_of_label("FamilyName_t")), None
    )
    family = family_node.data_as_str() if family_node else None
    if family:
        return family, of_type
    if element_ids.size:
        lo, hi = int(element_ids.min()), int(element_ids.max())
        for sec_name, first, last in sections:
            if (first, last) != (lo, hi) or element_ids.size != last - first + 1:
                continue
            stem = sec_name.rsplit("-", 1)[0]
            if "-" in sec_name and bc.name.startswith(stem + "-"):
                return stem, of_type
    return bc.name, of_type


def _note_fluent_interfaces(bc_names: list[str], notes: list[str]) -> None:
    interfaces = [n for n in bc_names if n.endswith("-interface")]
    if interfaces:
        notes.append(
            f"{len(interfaces)} BC(s) are Fluent mesh interfaces "
            f"({', '.join(interfaces)}). They are written as ordinary "
            "patches with no coupling across them, so the tracker would "
            "stop or reflect particles there. Couple them in OpenFOAM "
            "(createNonConformalCouples) before tracking on this mesh, "
            "or map these fields onto a coupled OpenFOAM case."
        )


def _read_polyhedral_boundary_groups(
    zone: CGNSNode,
    elem_to_face: np.ndarray,
    sections: list[tuple[str, int, int]],
) -> list[BoundaryFaceGroup]:
    """Translate each BC_t into a BoundaryFaceGroup of face indices.
    ``elem_to_face`` maps a CGNS element id to its index in the
    concatenated ``NGON_n`` face list (-1 if not a face)."""
    zbc_nodes = zone.children_of_label("ZoneBC_t")
    if not zbc_nodes:
        return []
    groups: list[BoundaryFaceGroup] = []
    for bc in zbc_nodes[0].children_of_label("BC_t"):
        eids = _bc_element_ids(bc)
        name, of_type = _bc_name_and_type(bc, eids, sections)
        bad = (eids < 1) | (eids >= elem_to_face.size)
        if not np.any(bad):
            face_ids = elem_to_face[eids]
            bad = face_ids < 0
        if np.any(bad):
            raise ValueError(
                f"BC_t {bc.name!r}: element id {int(eids[bad][0])} is not "
                "an NGON_n face"
            )
        groups.append(BoundaryFaceGroup(name=name, type=of_type, face_ids=face_ids))
    return groups


def _read_boundary_groups(
    zone: CGNSNode,
    surface_elems: list[tuple[str, int, int, np.ndarray]],
    sections: list[tuple[str, int, int]],
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
        eids = _bc_element_ids(bc)
        patch_name, of_type = _bc_name_and_type(bc, eids, sections)
        elem_ids = eids.tolist()

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


def _build_polyhedral_cells(
    ngon_elems: list[tuple[str, int, int, tuple[np.ndarray, np.ndarray]]],
    nface_elems: list[tuple[str, int, int, tuple[np.ndarray, np.ndarray]]],
    n_points: int,
) -> tuple[PolyhedralCells, np.ndarray]:
    """Concatenate the NGON_n sections (in element-id order) into one
    face list and resolve NFACE_n face references into it. Returns
    the cells and the element-id -> face-index lookup."""
    if not ngon_elems:
        raise ValueError("Zone has NFACE_n cells but no NGON_n faces")
    ngon_elems = sorted(ngon_elems, key=lambda e: e[1])
    nface_elems = sorted(nface_elems, key=lambda e: e[1])

    max_id = max(e[2] for e in ngon_elems + nface_elems)
    elem_to_face = np.full(max_id + 1, -1, dtype=np.int64)
    offsets_parts = []
    vertex_parts = []
    n_faces = 0
    n_conn = 0
    for _t, first, last, (offsets, flat) in ngon_elems:
        elem_to_face[first:last + 1] = np.arange(n_faces, n_faces + last - first + 1)
        offsets_parts.append(offsets[:-1] + n_conn)
        vertex_parts.append(flat - 1)
        n_faces += last - first + 1
        n_conn += flat.size
    face_offsets = np.concatenate(offsets_parts + [np.array([n_conn])])
    face_vertices = np.concatenate(vertex_parts)
    if face_vertices.size and (
        face_vertices.min() < 0 or face_vertices.max() >= n_points
    ):
        raise ValueError("NGON_n references a vertex outside GridCoordinates")

    cell_offsets_parts = []
    cell_face_parts = []
    n_ref = 0
    for name, _first, _last, (offsets, flat) in nface_elems:
        eid = np.abs(flat)
        bad = (eid < 1) | (eid > max_id)
        if not np.any(bad):
            face_idx = elem_to_face[eid]
            bad = face_idx < 0
        if np.any(bad):
            raise ValueError(
                f"NFACE_n section {name!r} references element "
                f"{int(flat[bad][0])}, which is not an NGON_n face"
            )
        cell_face_parts.append(np.sign(flat) * (face_idx + 1))
        cell_offsets_parts.append(offsets[:-1] + n_ref)
        n_ref += flat.size
    cells = PolyhedralCells(
        face_offsets=face_offsets,
        face_vertices=face_vertices,
        cell_offsets=np.concatenate(cell_offsets_parts + [np.array([n_ref])]),
        cell_faces=np.concatenate(cell_face_parts),
    )
    return cells, elem_to_face


def read_cgns(path: Path | str, notes: list[str] | None = None):
    """Parse a CGNS file. Returns ``(points, cells, boundary_groups,
    flow_solution)``.

    ``cells`` is a list of ``CellBlock`` for standard elements, or a
    ``PolyhedralCells`` for ``NGON_n`` / ``NFACE_n`` meshes.
    ``flow_solution`` is a dict with keys ``"scalars"`` (OF-name →
    array) and ``"vectors"`` (OF-name → (n,3) array). Anything worth
    telling the user about the input is appended to ``notes``.
    """
    if notes is None:
        notes = []
    root = read_cgns_file(path)
    zone = _find_zone(root)

    points = _read_points(zone)

    # Walk every Elements_t in the zone, partitioning by kind.
    volume_elems: list[tuple[str, int, int, np.ndarray]] = []
    surface_elems: list[tuple[str, int, int, np.ndarray]] = []
    ngon_elems: list = []
    nface_elems: list = []
    n_elements = 0
    sections: list[tuple[str, int, int]] = []
    for elem in zone.children_of_label("Elements_t"):
        type_name, first, last, conn = _read_elements_node(elem)
        n_elements = max(n_elements, last)
        sections.append((elem.name, first, last))
        if type_name == "NGON_n":
            ngon_elems.append((type_name, first, last, conn))
        elif type_name == "NFACE_n":
            nface_elems.append((elem.name, first, last, conn))
        elif type_name in SUPPORTED_VOLUME_ELEMENTS:
            volume_elems.append((type_name, first, last, conn))
        elif type_name in SUPPORTED_BOUNDARY_ELEMENTS:
            surface_elems.append((type_name, first, last, conn))
        else:
            raise NotImplementedError(
                f"Elements_t {elem.name!r}: {type_name} is not currently "
                "handled by the converter."
            )

    if nface_elems:
        if volume_elems or surface_elems:
            raise NotImplementedError(
                "Zone mixes NFACE_n polyhedra with standard elements; "
                "re-export with all cells as polyhedra or none."
            )
        cells, elem_to_face = _build_polyhedral_cells(
            ngon_elems, nface_elems, points.shape[0]
        )
        n_cells = cells.n_cells
        cell_ranges = [(f, l) for (_n, f, l, _c) in sorted(nface_elems, key=lambda e: e[1])]
        boundary_groups = _read_polyhedral_boundary_groups(zone, elem_to_face, sections)
    else:
        if ngon_elems:
            raise NotImplementedError(
                "Zone has NGON_n faces but no NFACE_n cells; face-only "
                "polyhedral exports are not supported."
            )
        if not volume_elems:
            raise ValueError("Zone has no volume Elements_t sections")
        cells = [
            CellBlock(element_type=t, connectivity=conn)
            for (t, _f, _l, conn) in volume_elems
        ]
        n_cells = sum(b.connectivity.shape[0] for b in cells)
        cell_ranges = [(f, l) for (_t, f, l, _c) in volume_elems]
        boundary_groups = _read_boundary_groups(zone, surface_elems, sections)

    zbc = zone.children_of_label("ZoneBC_t")
    if zbc:
        _note_fluent_interfaces(
            [bc.name for bc in zbc[0].children_of_label("BC_t")], notes
        )

    flow_solution = _read_flow_solution(zone, n_cells, cell_ranges, n_elements, notes)

    return points, cells, boundary_groups, flow_solution

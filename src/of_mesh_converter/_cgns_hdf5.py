"""Minimal CGNS-HDF5 file-mapping helpers.

A CGNS file is an HDF5 file with a specific layout (the CGNS-HDF5
file mapping documented at
https://cgns.github.io/CGNS_docs_current/filemap/index.html).

Each CGNS node is one HDF5 group with:

- attribute ``label``   — the CGNS_t type name, e.g. ``Zone_t``
- attribute ``name``    — the node's own name
- attribute ``type``    — data-type code: ``I4``, ``I8``, ``R4``,
                          ``R8``, ``C1``, ``MT``
- attribute ``flags``   — bitmask, typically 1
- dataset   `` data``   — the node's value (numerical or character;
                          omitted for ``MT`` / structural nodes)

Children are HDF5 groups with the same convention. This module is
the only place that knows the on-disk attribute and dataset names;
the rest of the converter sees a flat ``CGNSNode`` API.

The helpers here are deliberately permissive on read (accept both
``name`` and `` name`` attribute spellings — older specs use the
leading-space form — and treat ``MT`` nodes as data-less) and strict
on write (we always emit the modern, no-leading-space form).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import h5py
import numpy as np

# Some CGNS HDF5 files use a leading-space attribute name as an
# ADF-mapping legacy. Accept both on read.
_NAME_ATTRS = ("name", " name")
_LABEL_ATTRS = ("label", " label")
_TYPE_ATTRS = ("type", " type")
_DATA_DATASETS = (" data", "data")


def _get_attr(group: h5py.Group, candidates) -> str | None:
    for key in candidates:
        if key in group.attrs:
            raw = group.attrs[key]
            if isinstance(raw, bytes):
                return raw.decode("ascii", errors="replace")
            if isinstance(raw, np.ndarray):
                # h5py sometimes returns a (n,) array of bytes for
                # fixed-length strings; flatten and decode.
                return b"".join(raw.flatten().tolist()).decode(
                    "ascii", errors="replace"
                )
            return str(raw)
    return None


def _get_data(group: h5py.Group) -> np.ndarray | None:
    for key in _DATA_DATASETS:
        if key in group:
            return group[key][...]
    return None


@dataclass
class CGNSNode:
    """In-memory representation of one CGNS node, plus its children.

    Children are stored as a list (CGNS order) and additionally
    indexed by ``label`` for the common "give me every Zone_t child"
    query.
    """

    name: str
    label: str
    data: np.ndarray | None = None
    children: list["CGNSNode"] = field(default_factory=list)

    def child(self, name: str) -> "CGNSNode | None":
        for c in self.children:
            if c.name == name:
                return c
        return None

    def children_of_label(self, label: str) -> list["CGNSNode"]:
        return [c for c in self.children if c.label == label]

    def data_as_str(self) -> str | None:
        if self.data is None:
            return None
        arr = np.asarray(self.data).flatten()
        if arr.dtype.kind in ("i", "u"):
            return bytes(arr.astype(np.uint8).tolist()).decode(
                "ascii", errors="replace"
            ).rstrip("\x00")
        if arr.dtype.kind == "S":
            return b"".join(arr.tolist()).decode("ascii", errors="replace").rstrip("\x00")
        return None


def read_cgns_file(path) -> CGNSNode:
    """Parse a CGNS file into a tree of ``CGNSNode``."""
    with h5py.File(path, "r") as f:
        return _read_group(f)


def _read_group(group) -> CGNSNode:
    name = _get_attr(group, _NAME_ATTRS) or (
        group.name.rsplit("/", 1)[-1] if group.name else ""
    )
    label = _get_attr(group, _LABEL_ATTRS) or ""
    data = _get_data(group)
    children: list[CGNSNode] = []
    for child_key in group:
        if child_key in _DATA_DATASETS:
            continue
        child = group[child_key]
        if isinstance(child, h5py.Group):
            children.append(_read_group(child))
    return CGNSNode(name=name, label=label, data=data, children=children)


def write_cgns_file(path, root: CGNSNode) -> None:
    """Write a ``CGNSNode`` tree to an HDF5 file in CGNS layout.

    Used only by tests and tutorials. Real Fluent exports go through
    ``read_cgns_file`` on the input side.
    """
    with h5py.File(path, "w") as f:
        for child in root.children:
            _write_group(f, child)
        # Root's own attributes mirror the CGNS root, label-wise.
        f.attrs["name"] = np.bytes_(root.name)
        f.attrs["label"] = np.bytes_(root.label)
        f.attrs["type"] = np.bytes_("MT")
        f.attrs["flags"] = np.int32(1)


def _write_group(parent, node: CGNSNode) -> None:
    g = parent.create_group(node.name)
    g.attrs["name"] = np.bytes_(node.name)
    g.attrs["label"] = np.bytes_(node.label)
    g.attrs["flags"] = np.int32(1)
    if node.data is None:
        g.attrs["type"] = np.bytes_("MT")
    else:
        arr = np.asarray(node.data)
        type_code = {
            np.dtype("int32"):   "I4",
            np.dtype("int64"):   "I8",
            np.dtype("float32"): "R4",
            np.dtype("float64"): "R8",
            np.dtype("uint8"):   "C1",
        }.get(arr.dtype, None)
        if type_code is None:
            # Fall back to int8 character payload for strings; the
            # caller is expected to have packed text into uint8.
            arr = arr.astype(np.uint8)
            type_code = "C1"
        g.attrs["type"] = np.bytes_(type_code)
        g.create_dataset(" data", data=arr)
    for c in node.children:
        _write_group(g, c)


def char_array(s: str) -> np.ndarray:
    """Pack a Python string into a CGNS C1 (uint8) byte array."""
    return np.frombuffer(s.encode("ascii"), dtype=np.uint8).copy()

"""Unit tests for ``sanitise.py``."""

from __future__ import annotations

import numpy as np
import pytest

from of_mesh_converter import sanitise


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("inlet", "inlet"),
        ("interior-fluid", "interior_fluid"),
        ("inlet:01", "inlet_01"),
        ("2_walls", "p_2_walls"),
        ("walls  with   spaces", "walls_with_spaces"),
        ("__leading_underscore", "leading_underscore"),
        ("", "unnamed"),
        ("!!!", "unnamed"),
    ],
)
def test_sanitise_patch_name(raw, expected):
    assert sanitise.sanitise_patch_name(raw) == expected


def test_sanitise_patch_names_resolves_collisions():
    names = ["wall-1", "wall:1", "wall 1"]
    out, mapping = sanitise.sanitise_patch_names(names)
    # All three collide on "wall_1"; the second and third should be
    # decorated with _2, _3.
    assert out == ["wall_1", "wall_1_2", "wall_1_3"]
    assert mapping["wall-1"] == "wall_1"
    assert mapping["wall:1"] == "wall_1_2"
    assert mapping["wall 1"] == "wall_1_3"


def test_clip_nonpositive_clips_and_counts():
    arr = np.array([0.0, -1e-10, 1e-3, 1.0, -5.0])
    clipped, n = sanitise.clip_nonpositive(arr)
    assert n == 3
    # Positives untouched.
    assert clipped[2] == 1e-3
    assert clipped[3] == 1.0
    # Non-positives all floored to the same value.
    assert clipped[0] == sanitise.TURBULENCE_FLOOR
    assert clipped[1] == sanitise.TURBULENCE_FLOOR
    assert clipped[4] == sanitise.TURBULENCE_FLOOR


def test_clip_nonpositive_returns_original_when_all_positive():
    arr = np.array([1.0, 2.0, 0.5])
    clipped, n = sanitise.clip_nonpositive(arr)
    assert n == 0
    np.testing.assert_array_equal(clipped, arr)

"""Unit tests for ``field_mapping.py``."""

from __future__ import annotations

import pytest

from of_mesh_converter import field_mapping


def test_scalar_mappings_cover_dose_tracker_inputs():
    """Anything radiationDose reads must have a mapping."""
    assert "k" in field_mapping.known_scalar_fields()
    assert "epsilon" in field_mapping.known_scalar_fields()
    assert "G" in field_mapping.known_scalar_fields()


def test_vector_mapping_for_velocity():
    cx, cy, cz = field_mapping.of_vector_components("U")
    assert (cx, cy, cz) == ("VelocityX", "VelocityY", "VelocityZ")


def test_of_scalar_name_returns_none_for_unknown():
    assert field_mapping.of_scalar_name("RandomPropertyFluentEmits") is None


def test_of_scalar_name_handles_known_cgns_names():
    assert field_mapping.of_scalar_name("TurbulentEnergyKinetic") == "k"
    assert field_mapping.of_scalar_name("TurbulentDissipationRate") == "epsilon"


def test_dimension_sets_present_for_all_known_fields():
    """Writer needs a dimension set for every field name it might emit."""
    for name in field_mapping.known_scalar_fields():
        assert name in field_mapping.OF_DIMENSIONS, name
    for name in field_mapping.known_vector_fields():
        assert name in field_mapping.OF_DIMENSIONS, name


def test_of_vector_components_rejects_unknown():
    with pytest.raises(KeyError):
        field_mapping.of_vector_components("NotAField")

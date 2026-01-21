"""
Unit tests for ControlPoint class and bidirectional linking.
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

from watfIGA.discretization.control_point import (
    ControlPoint, create_control_points_from_array
)


class TestControlPoint:
    """Tests for ControlPoint class."""

    def test_basic_creation(self):
        """Test basic control point creation."""
        cp = ControlPoint(
            id=0,
            coordinates=np.array([1.0, 2.0]),
            weight=1.0,
            level=0,
            active=True
        )

        assert cp.id == 0
        assert_array_almost_equal(cp.coordinates, [1.0, 2.0])
        assert cp.weight == 1.0
        assert cp.level == 0
        assert cp.active is True
        assert len(cp.supported_elements) == 0

    def test_coordinate_properties(self):
        """Test x, y, z coordinate properties."""
        cp_2d = ControlPoint(id=0, coordinates=np.array([1.0, 2.0]))
        assert cp_2d.x == 1.0
        assert cp_2d.y == 2.0
        assert cp_2d.z is None
        assert cp_2d.n_dim == 2

        cp_3d = ControlPoint(id=1, coordinates=np.array([1.0, 2.0, 3.0]))
        assert cp_3d.x == 1.0
        assert cp_3d.y == 2.0
        assert cp_3d.z == 3.0
        assert cp_3d.n_dim == 3

    def test_default_values(self):
        """Test default values for optional parameters."""
        cp = ControlPoint(id=0, coordinates=np.array([0.0, 0.0]))

        assert cp.weight == 1.0
        assert cp.level == 0
        assert cp.active is True
        assert cp.supported_elements == set()

    def test_add_remove_element(self):
        """Test adding and removing element links."""
        cp = ControlPoint(id=0, coordinates=np.array([0.0, 0.0]))

        cp.add_element(5)
        cp.add_element(10)
        assert cp.supported_elements == {5, 10}

        cp.add_element(5)  # Duplicate, should not add
        assert cp.supported_elements == {5, 10}

        cp.remove_element(5)
        assert cp.supported_elements == {10}

        cp.remove_element(999)  # Non-existent, should not error
        assert cp.supported_elements == {10}

    def test_activate_deactivate(self):
        """Test activation and deactivation."""
        cp = ControlPoint(id=0, coordinates=np.array([0.0, 0.0]))
        assert cp.active is True

        cp.deactivate()
        assert cp.active is False

        cp.activate()
        assert cp.active is True

    def test_hash_and_equality(self):
        """Test hash and equality based on ID."""
        cp1 = ControlPoint(id=5, coordinates=np.array([0.0, 0.0]))
        cp2 = ControlPoint(id=5, coordinates=np.array([1.0, 1.0]))  # Same ID
        cp3 = ControlPoint(id=6, coordinates=np.array([0.0, 0.0]))

        assert cp1 == cp2  # Same ID
        assert cp1 != cp3  # Different ID
        assert hash(cp1) == hash(cp2)

        # Can use in sets
        cp_set = {cp1, cp2, cp3}
        assert len(cp_set) == 2  # cp1 and cp2 are equal

    def test_repr(self):
        """Test string representation."""
        cp = ControlPoint(
            id=0,
            coordinates=np.array([1.0, 2.0]),
            weight=0.5,
            level=1,
            active=False,
            supported_elements={1, 2, 3}
        )
        repr_str = repr(cp)
        assert "id=0" in repr_str
        assert "level=1" in repr_str
        assert "active=False" in repr_str
        assert "n_elements=3" in repr_str


class TestCreateControlPointsFromArray:
    """Tests for create_control_points_from_array function."""

    def test_basic_creation(self):
        """Test creating control points from coordinate array."""
        coords = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ])

        cp_dict = create_control_points_from_array(coords)

        assert len(cp_dict) == 4
        assert all(isinstance(cp, ControlPoint) for cp in cp_dict.values())
        assert_array_almost_equal(cp_dict[0].coordinates, [0.0, 0.0])
        assert_array_almost_equal(cp_dict[3].coordinates, [1.0, 1.0])

    def test_with_weights(self):
        """Test creating control points with custom weights."""
        coords = np.array([[0.0, 0.0], [1.0, 1.0]])
        weights = np.array([0.5, 2.0])

        cp_dict = create_control_points_from_array(coords, weights)

        assert cp_dict[0].weight == 0.5
        assert cp_dict[1].weight == 2.0

    def test_with_level(self):
        """Test creating control points with specified level."""
        coords = np.array([[0.0, 0.0], [1.0, 1.0]])

        cp_dict = create_control_points_from_array(coords, level=2)

        assert cp_dict[0].level == 2
        assert cp_dict[1].level == 2

    def test_all_active(self):
        """Test that all created control points are active."""
        coords = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        cp_dict = create_control_points_from_array(coords)

        assert all(cp.active for cp in cp_dict.values())

    def test_empty_supported_elements(self):
        """Test that supported_elements starts empty."""
        coords = np.array([[0.0, 0.0]])

        cp_dict = create_control_points_from_array(coords)

        assert cp_dict[0].supported_elements == set()

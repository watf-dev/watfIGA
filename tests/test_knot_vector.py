"""
Unit tests for knot vector utilities.
"""

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from watfIGA.discretization.knot_vector import (
    KnotVector, make_open_knot_vector, insert_knot, compute_multiplicity
)


class TestKnotVector:
    """Tests for KnotVector class."""

    def test_open_knot_vector_creation(self):
        """Test creating an open (clamped) uniform knot vector."""
        kv = make_open_knot_vector(n_basis=5, degree=2, domain=(0.0, 1.0))

        # Check basic properties
        assert kv.degree == 2
        assert kv.n_basis == 5
        assert len(kv.knots) == 5 + 2 + 1  # n + p + 1

        # Check open knot vector structure: p+1 repeated at ends
        assert_array_equal(kv.knots[:3], [0.0, 0.0, 0.0])
        assert_array_equal(kv.knots[-3:], [1.0, 1.0, 1.0])

    def test_knot_vector_domain(self):
        """Test that domain is correctly computed."""
        kv = make_open_knot_vector(n_basis=4, degree=2, domain=(0.0, 1.0))
        assert kv.domain == (0.0, 1.0)

        kv2 = make_open_knot_vector(n_basis=4, degree=2, domain=(-1.0, 2.0))
        assert kv2.domain == (-1.0, 2.0)

    def test_n_elements(self):
        """Test counting number of elements (non-zero knot spans)."""
        # Degree 2, 4 basis functions: knots = [0,0,0,0.5,1,1,1]
        kv = make_open_knot_vector(n_basis=4, degree=2, domain=(0.0, 1.0))
        assert kv.n_elements == 2

        # Degree 2, 6 basis functions
        kv = make_open_knot_vector(n_basis=6, degree=2, domain=(0.0, 1.0))
        assert kv.n_elements == 4

    def test_elements_list(self):
        """Test that element intervals are correct."""
        kv = make_open_knot_vector(n_basis=4, degree=2, domain=(0.0, 1.0))
        elements = kv.elements

        assert len(elements) == 2
        assert elements[0] == (0.0, 0.5)
        assert elements[1] == (0.5, 1.0)

    def test_find_span_interior(self):
        """Test finding knot span for interior points."""
        kv = make_open_knot_vector(n_basis=4, degree=2, domain=(0.0, 1.0))

        # Point in first span
        assert kv.find_span(0.25) == 2

        # Point in second span
        assert kv.find_span(0.75) == 3

    def test_find_span_boundaries(self):
        """Test finding knot span at domain boundaries."""
        kv = make_open_knot_vector(n_basis=4, degree=2, domain=(0.0, 1.0))

        # At left boundary
        assert kv.find_span(0.0) == 2

        # At right boundary (should return last span)
        assert kv.find_span(1.0) == 3

    def test_find_element(self):
        """Test finding element index for a parameter value."""
        kv = make_open_knot_vector(n_basis=4, degree=2, domain=(0.0, 1.0))

        assert kv.find_element(0.25) == 0
        assert kv.find_element(0.5) == 1  # On boundary, belongs to second element
        assert kv.find_element(0.75) == 1

    def test_active_basis_indices(self):
        """Test getting active basis function indices for an element."""
        kv = make_open_knot_vector(n_basis=4, degree=2, domain=(0.0, 1.0))

        # First element: active functions are 0, 1, 2
        indices = kv.active_basis_indices(0)
        assert_array_equal(indices, [0, 1, 2])

        # Second element: active functions are 1, 2, 3
        indices = kv.active_basis_indices(1)
        assert_array_equal(indices, [1, 2, 3])

    def test_greville_abscissae(self):
        """Test Greville abscissae computation."""
        # For open uniform knot vector [0,0,0,0.5,1,1,1] with p=2:
        # Greville_0 = (0 + 0) / 2 = 0
        # Greville_1 = (0 + 0.5) / 2 = 0.25
        # Greville_2 = (0.5 + 1) / 2 = 0.75
        # Greville_3 = (1 + 1) / 2 = 1
        kv = make_open_knot_vector(n_basis=4, degree=2, domain=(0.0, 1.0))
        greville = kv.greville_abscissae()

        assert_array_almost_equal(greville, [0.0, 0.25, 0.75, 1.0])

    def test_unique_knots(self):
        """Test unique knots (breakpoints)."""
        kv = make_open_knot_vector(n_basis=4, degree=2, domain=(0.0, 1.0))
        unique = kv.unique_knots

        assert_array_almost_equal(unique, [0.0, 0.5, 1.0])

    def test_knot_insertion(self):
        """Test knot insertion utility."""
        kv = make_open_knot_vector(n_basis=4, degree=2, domain=(0.0, 1.0))
        kv_new = insert_knot(kv, 0.25)

        assert kv_new.n_basis == kv.n_basis + 1
        assert 0.25 in kv_new.knots

    def test_multiplicity(self):
        """Test knot multiplicity computation."""
        kv = make_open_knot_vector(n_basis=4, degree=2, domain=(0.0, 1.0))

        # Boundary knots have multiplicity p+1 = 3
        assert compute_multiplicity(kv, 0.0) == 3
        assert compute_multiplicity(kv, 1.0) == 3

        # Internal knot has multiplicity 1
        assert compute_multiplicity(kv, 0.5) == 1

        # Non-existent knot has multiplicity 0
        assert compute_multiplicity(kv, 0.25) == 0

    def test_invalid_knot_vector(self):
        """Test that invalid knot vectors raise errors."""
        # Too few knots
        with pytest.raises(ValueError):
            KnotVector(np.array([0.0, 1.0]), degree=2)

        # Non-increasing knots
        with pytest.raises(ValueError):
            KnotVector(np.array([0.0, 0.0, 0.5, 0.3, 1.0, 1.0]), degree=1)

    def test_degree_3(self):
        """Test with cubic (degree 3) basis."""
        kv = make_open_knot_vector(n_basis=6, degree=3, domain=(0.0, 1.0))

        assert kv.degree == 3
        assert kv.n_basis == 6
        assert kv.n_elements == 3

        # Check open structure
        assert_array_equal(kv.knots[:4], [0.0, 0.0, 0.0, 0.0])
        assert_array_equal(kv.knots[-4:], [1.0, 1.0, 1.0, 1.0])

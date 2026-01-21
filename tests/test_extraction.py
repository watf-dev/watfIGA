"""
Unit tests for Bézier extraction operators.
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal

from watfIGA.discretization.knot_vector import make_open_knot_vector
from watfIGA.discretization.extraction import (
    compute_extraction_operators_1d,
    compute_extraction_operators_2d,
    BernsteinBasis,
    _bernstein_basis,
    bernstein_basis_ders
)
from watfIGA.geometry.bspline import eval_basis_1d


class TestBernsteinBasis:
    """Tests for Bernstein polynomial basis."""

    def test_partition_of_unity(self):
        """Test that Bernstein polynomials sum to 1."""
        for p in [1, 2, 3, 4]:
            for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
                B = _bernstein_basis(p, t)
                assert_almost_equal(np.sum(B), 1.0, decimal=14)

    def test_boundary_values(self):
        """Test Bernstein values at boundaries."""
        for p in [1, 2, 3]:
            # At t=0: B_0 = 1, others = 0
            B = _bernstein_basis(p, 0.0)
            assert_almost_equal(B[0], 1.0)
            assert_almost_equal(B[1:], 0.0)

            # At t=1: B_p = 1, others = 0
            B = _bernstein_basis(p, 1.0)
            assert_almost_equal(B[-1], 1.0)
            assert_almost_equal(B[:-1], 0.0)

    def test_symmetry(self):
        """Test Bernstein symmetry: B_{i,p}(t) = B_{p-i,p}(1-t)."""
        for p in [2, 3]:
            for t in [0.25, 0.5, 0.75]:
                B_t = _bernstein_basis(p, t)
                B_1mt = _bernstein_basis(p, 1 - t)
                assert_array_almost_equal(B_t, B_1mt[::-1])

    def test_known_values_quadratic(self):
        """Test known quadratic Bernstein values."""
        # B_0 = (1-t)^2, B_1 = 2t(1-t), B_2 = t^2
        t = 0.5
        B = _bernstein_basis(2, t)
        expected = [0.25, 0.5, 0.25]  # (1-0.5)^2, 2*0.5*0.5, 0.5^2
        assert_array_almost_equal(B, expected)

    def test_derivatives(self):
        """Test Bernstein derivatives."""
        # d/dt B_{i,p} = p * (B_{i-1,p-1} - B_{i,p-1})
        Bders = bernstein_basis_ders(2, 0.5, n_ders=1)

        # Values at t=0.5
        assert_array_almost_equal(Bders[0, :], [0.25, 0.5, 0.25])

        # First derivatives at t=0.5:
        # dB_0/dt = -2(1-t) = -1
        # dB_1/dt = 2(1-2t) = 0
        # dB_2/dt = 2t = 1
        assert_array_almost_equal(Bders[1, :], [-1.0, 0.0, 1.0])


class TestBernsteinBasisClass:
    """Tests for BernsteinBasis class."""

    def test_1d_basis(self):
        """Test 1D Bernstein basis class."""
        basis = BernsteinBasis((2,))
        assert basis.n_basis == 3
        assert basis.n_dim == 1

        B = basis.eval((0.5,))
        assert_array_almost_equal(B, [0.25, 0.5, 0.25])

    def test_2d_basis(self):
        """Test 2D tensor-product Bernstein basis."""
        basis = BernsteinBasis((2, 2))
        assert basis.n_basis == 9
        assert basis.n_dim == 2

        B = basis.eval((0.5, 0.5))
        assert_almost_equal(np.sum(B), 1.0)

    def test_2d_derivatives(self):
        """Test 2D Bernstein derivatives."""
        basis = BernsteinBasis((2, 2))
        B, dB_dxi, dB_deta = basis.eval_ders((0.5, 0.5), n_ders=1)

        # Derivatives should sum to zero
        assert_almost_equal(np.sum(dB_dxi), 0.0, decimal=10)
        assert_almost_equal(np.sum(dB_deta), 0.0, decimal=10)


class TestExtractionOperators1D:
    """Tests for 1D Bézier extraction operators."""

    def test_single_element_identity(self):
        """Test that single element (Bezier) has identity extraction."""
        # Single element = Bezier, so C should be identity
        kv = make_open_knot_vector(n_basis=3, degree=2, domain=(0.0, 1.0))
        C_list = compute_extraction_operators_1d(kv)

        assert len(C_list) == 1
        assert_array_almost_equal(C_list[0], np.eye(3))

    def test_extraction_preserves_basis(self):
        """Test that N = C @ B reproduces spline basis."""
        kv = make_open_knot_vector(n_basis=5, degree=2, domain=(0.0, 1.0))
        C_list = compute_extraction_operators_1d(kv)
        elements = kv.elements

        for e, (xi_min, xi_max) in enumerate(elements):
            C_e = C_list[e]
            h = xi_max - xi_min

            # Test at several points in this element
            for t in [0.25, 0.5, 0.75]:
                xi = xi_min + t * h

                # Bernstein at reference point
                B = _bernstein_basis(2, t)

                # Spline basis from extraction
                N_from_C = C_e @ B

                # Spline basis directly
                N_direct = eval_basis_1d(kv, xi)

                assert_array_almost_equal(N_from_C, N_direct, decimal=10)

    def test_column_sums_one(self):
        """Test that extraction operator columns sum to 1 (partition of unity)."""
        kv = make_open_knot_vector(n_basis=5, degree=2, domain=(0.0, 1.0))
        C_list = compute_extraction_operators_1d(kv)

        for C_e in C_list:
            col_sums = np.sum(C_e, axis=0)
            assert_array_almost_equal(col_sums, np.ones(C_e.shape[1]))

    def test_extraction_shape(self):
        """Test extraction operator dimensions."""
        kv = make_open_knot_vector(n_basis=5, degree=2, domain=(0.0, 1.0))
        C_list = compute_extraction_operators_1d(kv)

        # Each element has p+1 = 3 active basis functions
        for C_e in C_list:
            assert C_e.shape == (3, 3)


class TestExtractionOperators2D:
    """Tests for 2D Bézier extraction operators."""

    def test_2d_extraction_count(self):
        """Test number of 2D extraction operators."""
        kv_xi = make_open_knot_vector(n_basis=4, degree=2, domain=(0.0, 1.0))
        kv_eta = make_open_knot_vector(n_basis=5, degree=2, domain=(0.0, 1.0))

        C_list = compute_extraction_operators_2d(kv_xi, kv_eta)

        # Should have n_elem_xi * n_elem_eta operators
        n_elem_xi = kv_xi.n_elements
        n_elem_eta = kv_eta.n_elements
        assert len(C_list) == n_elem_xi * n_elem_eta

    def test_2d_extraction_shape(self):
        """Test 2D extraction operator dimensions."""
        kv_xi = make_open_knot_vector(n_basis=4, degree=2, domain=(0.0, 1.0))
        kv_eta = make_open_knot_vector(n_basis=4, degree=2, domain=(0.0, 1.0))

        C_list = compute_extraction_operators_2d(kv_xi, kv_eta)

        # Each should be (p_xi+1)*(p_eta+1) x (p_xi+1)*(p_eta+1) = 9x9
        for C_e in C_list:
            assert C_e.shape == (9, 9)

    def test_2d_extraction_kronecker(self):
        """Test that 2D extraction is Kronecker product of 1D."""
        kv_xi = make_open_knot_vector(n_basis=4, degree=2, domain=(0.0, 1.0))
        kv_eta = make_open_knot_vector(n_basis=4, degree=2, domain=(0.0, 1.0))

        C_xi_list = compute_extraction_operators_1d(kv_xi)
        C_eta_list = compute_extraction_operators_1d(kv_eta)
        C_2d_list = compute_extraction_operators_2d(kv_xi, kv_eta)

        # Check first element
        C_2d_expected = np.kron(C_xi_list[0], C_eta_list[0])
        assert_array_almost_equal(C_2d_list[0], C_2d_expected)

    def test_2d_column_sums(self):
        """Test 2D extraction column sums."""
        kv_xi = make_open_knot_vector(n_basis=4, degree=2, domain=(0.0, 1.0))
        kv_eta = make_open_knot_vector(n_basis=4, degree=2, domain=(0.0, 1.0))

        C_list = compute_extraction_operators_2d(kv_xi, kv_eta)

        for C_e in C_list:
            col_sums = np.sum(C_e, axis=0)
            assert_array_almost_equal(col_sums, np.ones(C_e.shape[1]))

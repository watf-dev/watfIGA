"""
Unit tests for B-spline basis function evaluation.
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal

from watfIGA.discretization.knot_vector import make_open_knot_vector
from watfIGA.geometry.bspline import (
    eval_basis_1d, eval_basis_ders_1d, BSplineBasis, TensorProductBasis
)


class TestBasisEvaluation1D:
    """Tests for 1D B-spline basis evaluation."""

    def test_partition_of_unity(self):
        """Test that basis functions sum to 1 (partition of unity)."""
        kv = make_open_knot_vector(n_basis=5, degree=2, domain=(0.0, 1.0))

        # Test at multiple points
        for xi in [0.0, 0.25, 0.5, 0.75, 1.0]:
            N = eval_basis_1d(kv, xi)
            assert_almost_equal(np.sum(N), 1.0, decimal=14)

    def test_non_negativity(self):
        """Test that basis functions are non-negative."""
        kv = make_open_knot_vector(n_basis=5, degree=2, domain=(0.0, 1.0))

        for xi in np.linspace(0, 1, 20):
            N = eval_basis_1d(kv, xi)
            assert np.all(N >= -1e-14), f"Negative basis value at xi={xi}"

    def test_boundary_interpolation(self):
        """Test that open knot vector basis interpolates at boundaries."""
        kv = make_open_knot_vector(n_basis=5, degree=2, domain=(0.0, 1.0))

        # At left boundary, only first basis function is 1
        N = eval_basis_1d(kv, 0.0)
        assert_almost_equal(N[0], 1.0)
        assert_almost_equal(N[1:], 0.0)

        # At right boundary, only last basis function is 1
        N = eval_basis_1d(kv, 1.0)
        assert_almost_equal(N[-1], 1.0)
        assert_almost_equal(N[:-1], 0.0)

    def test_local_support(self):
        """Test that exactly p+1 basis functions are non-zero."""
        for p in [1, 2, 3]:
            kv = make_open_knot_vector(n_basis=p + 3, degree=p, domain=(0.0, 1.0))

            for xi in [0.25, 0.5, 0.75]:
                N = eval_basis_1d(kv, xi)
                n_nonzero = np.sum(np.abs(N) > 1e-14)
                assert n_nonzero == p + 1, f"Expected {p+1} non-zero, got {n_nonzero}"

    def test_known_values_degree_1(self):
        """Test known values for linear (p=1) basis."""
        # Linear basis on [0,0,0.5,1,1]: hat functions
        kv = make_open_knot_vector(n_basis=3, degree=1, domain=(0.0, 1.0))

        # At midpoint of first span
        N = eval_basis_1d(kv, 0.25)
        assert_almost_equal(N[0], 0.5)
        assert_almost_equal(N[1], 0.5)

    def test_known_values_degree_2(self):
        """Test known values for quadratic (p=2) basis."""
        # Quadratic basis on [0,0,0,1,1,1]: single Bezier element
        kv = make_open_knot_vector(n_basis=3, degree=2, domain=(0.0, 1.0))

        # At midpoint: Bernstein polynomials
        N = eval_basis_1d(kv, 0.5)
        # B_0 = (1-t)^2 = 0.25, B_1 = 2t(1-t) = 0.5, B_2 = t^2 = 0.25
        assert_almost_equal(N, [0.25, 0.5, 0.25])


class TestBasisDerivatives1D:
    """Tests for 1D B-spline basis derivatives."""

    def test_derivative_sum_zero(self):
        """Test that derivative of partition of unity is zero."""
        kv = make_open_knot_vector(n_basis=5, degree=2, domain=(0.0, 1.0))

        for xi in [0.1, 0.5, 0.9]:
            Nders = eval_basis_ders_1d(kv, xi, n_ders=1)
            dN = Nders[1, :]
            assert_almost_equal(np.sum(dN), 0.0, decimal=12)

    def test_derivative_values(self):
        """Test derivative values against known results."""
        # Single Bezier element (Bernstein)
        kv = make_open_knot_vector(n_basis=3, degree=2, domain=(0.0, 1.0))

        Nders = eval_basis_ders_1d(kv, 0.5, n_ders=1)

        # Values
        assert_almost_equal(Nders[0, :], [0.25, 0.5, 0.25])

        # First derivatives of Bernstein:
        # dB_0/dt = -2(1-t) = -1 at t=0.5
        # dB_1/dt = 2(1-2t) = 0 at t=0.5
        # dB_2/dt = 2t = 1 at t=0.5
        assert_almost_equal(Nders[1, :], [-1.0, 0.0, 1.0])

    def test_second_derivative(self):
        """Test second derivative values."""
        kv = make_open_knot_vector(n_basis=3, degree=2, domain=(0.0, 1.0))

        Nders = eval_basis_ders_1d(kv, 0.5, n_ders=2)

        # Second derivatives of Bernstein:
        # d²B_0/dt² = 2
        # d²B_1/dt² = -4
        # d²B_2/dt² = 2
        assert_almost_equal(Nders[2, :], [2.0, -4.0, 2.0])


class TestBSplineBasisClass:
    """Tests for BSplineBasis class."""

    def test_basic_properties(self):
        """Test basic properties of BSplineBasis."""
        kv = make_open_knot_vector(n_basis=5, degree=2, domain=(0.0, 1.0))
        basis = BSplineBasis(kv)

        assert basis.degree == 2
        assert basis.n_basis == 5
        assert basis.n_elements == 3

    def test_eval_method(self):
        """Test eval method matches function."""
        kv = make_open_knot_vector(n_basis=5, degree=2, domain=(0.0, 1.0))
        basis = BSplineBasis(kv)

        for xi in [0.25, 0.5, 0.75]:
            N1 = basis.eval(xi)
            N2 = eval_basis_1d(kv, xi)
            assert_array_almost_equal(N1, N2)


class TestTensorProductBasis:
    """Tests for tensor-product B-spline basis."""

    def test_2d_partition_of_unity(self):
        """Test partition of unity in 2D."""
        kv_xi = make_open_knot_vector(n_basis=4, degree=2, domain=(0.0, 1.0))
        kv_eta = make_open_knot_vector(n_basis=4, degree=2, domain=(0.0, 1.0))

        basis_xi = BSplineBasis(kv_xi)
        basis_eta = BSplineBasis(kv_eta)
        tp_basis = TensorProductBasis((basis_xi, basis_eta))

        for xi in [0.25, 0.5, 0.75]:
            for eta in [0.25, 0.5, 0.75]:
                N = tp_basis.eval((xi, eta))
                assert_almost_equal(np.sum(N), 1.0, decimal=12)

    def test_2d_dimensions(self):
        """Test dimensions of 2D basis."""
        kv_xi = make_open_knot_vector(n_basis=4, degree=2, domain=(0.0, 1.0))
        kv_eta = make_open_knot_vector(n_basis=5, degree=2, domain=(0.0, 1.0))

        basis_xi = BSplineBasis(kv_xi)
        basis_eta = BSplineBasis(kv_eta)
        tp_basis = TensorProductBasis((basis_xi, basis_eta))

        assert tp_basis.n_dim == 2
        assert tp_basis.n_basis_per_dir == (4, 5)
        assert tp_basis.n_basis_total == 20
        assert tp_basis.degrees == (2, 2)

    def test_2d_local_support(self):
        """Test local support in 2D."""
        kv_xi = make_open_knot_vector(n_basis=5, degree=2, domain=(0.0, 1.0))
        kv_eta = make_open_knot_vector(n_basis=5, degree=2, domain=(0.0, 1.0))

        basis_xi = BSplineBasis(kv_xi)
        basis_eta = BSplineBasis(kv_eta)
        tp_basis = TensorProductBasis((basis_xi, basis_eta))

        # At interior point, (p+1)^2 = 9 basis functions should be non-zero
        N = tp_basis.eval((0.5, 0.5))
        n_nonzero = np.sum(np.abs(N) > 1e-14)
        assert n_nonzero == 9

    def test_index_conversion(self):
        """Test global to tensor index conversion."""
        kv_xi = make_open_knot_vector(n_basis=3, degree=1, domain=(0.0, 1.0))
        kv_eta = make_open_knot_vector(n_basis=4, degree=1, domain=(0.0, 1.0))

        basis_xi = BSplineBasis(kv_xi)
        basis_eta = BSplineBasis(kv_eta)
        tp_basis = TensorProductBasis((basis_xi, basis_eta))

        # Test round-trip conversion
        for global_idx in range(tp_basis.n_basis_total):
            tensor_idx = tp_basis.global_to_tensor_index(global_idx)
            back = tp_basis.tensor_to_global_index(tensor_idx)
            assert back == global_idx

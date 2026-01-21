"""
Unit tests for Gauss-Legendre quadrature.
"""

import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from watfIGA.quadrature.gauss import (
    gauss_legendre_1d, gauss_legendre_2d, gauss_legendre_3d,
    GaussQuadrature
)


class TestGaussLegendre1D:
    """Tests for 1D Gauss-Legendre quadrature."""

    def test_weights_sum_to_one(self):
        """Test that weights sum to 1 (domain is [0,1])."""
        for n in [1, 2, 3, 4, 5]:
            pts, wts = gauss_legendre_1d(n)
            assert_almost_equal(np.sum(wts), 1.0, decimal=14)

    def test_points_in_domain(self):
        """Test that all points are in [0, 1]."""
        for n in [1, 2, 3, 4, 5]:
            pts, wts = gauss_legendre_1d(n)
            assert np.all(pts >= 0.0)
            assert np.all(pts <= 1.0)

    def test_integrate_constant(self):
        """Test integration of constant function."""
        # ∫_0^1 c dx = c
        pts, wts = gauss_legendre_1d(1)
        c = 3.5
        result = np.sum(c * wts)
        assert_almost_equal(result, c)

    def test_integrate_polynomial(self):
        """Test exact integration of polynomials up to degree 2n-1."""
        # n=2 should integrate exactly up to degree 3
        pts, wts = gauss_legendre_1d(2)

        # ∫_0^1 x^3 dx = 1/4
        result = np.sum(pts**3 * wts)
        assert_almost_equal(result, 0.25, decimal=14)

        # n=3 should integrate exactly up to degree 5
        pts, wts = gauss_legendre_1d(3)

        # ∫_0^1 x^5 dx = 1/6
        result = np.sum(pts**5 * wts)
        assert_almost_equal(result, 1/6, decimal=14)

    def test_integrate_sin(self):
        """Test integration of sin function with high enough order."""
        # ∫_0^1 sin(x) dx = 1 - cos(1) ≈ 0.4597
        pts, wts = gauss_legendre_1d(10)
        result = np.sum(np.sin(pts) * wts)
        exact = 1 - np.cos(1)
        assert_almost_equal(result, exact, decimal=10)


class TestGaussLegendre2D:
    """Tests for 2D Gauss-Legendre quadrature."""

    def test_weights_sum_to_one(self):
        """Test that 2D weights sum to 1 (domain is [0,1]^2)."""
        for n in [1, 2, 3, 4]:
            pts, wts = gauss_legendre_2d(n, n)
            assert_almost_equal(np.sum(wts), 1.0, decimal=14)

    def test_points_shape(self):
        """Test shape of 2D quadrature points."""
        pts, wts = gauss_legendre_2d(3, 4)
        assert pts.shape == (12, 2)
        assert wts.shape == (12,)

    def test_integrate_constant(self):
        """Test 2D integration of constant."""
        pts, wts = gauss_legendre_2d(2, 2)
        c = 2.5
        result = np.sum(c * wts)
        assert_almost_equal(result, c)

    def test_integrate_polynomial_2d(self):
        """Test 2D integration of polynomial."""
        # ∫∫ x*y dA = 1/4 on [0,1]^2
        pts, wts = gauss_legendre_2d(2, 2)
        result = np.sum(pts[:, 0] * pts[:, 1] * wts)
        assert_almost_equal(result, 0.25, decimal=14)

        # ∫∫ x^2 + y^2 dA = 2/3 on [0,1]^2
        result = np.sum((pts[:, 0]**2 + pts[:, 1]**2) * wts)
        assert_almost_equal(result, 2/3, decimal=14)

    def test_points_in_domain(self):
        """Test that all 2D points are in [0,1]^2."""
        pts, wts = gauss_legendre_2d(5, 5)
        assert np.all(pts >= 0.0)
        assert np.all(pts <= 1.0)


class TestGaussLegendre3D:
    """Tests for 3D Gauss-Legendre quadrature."""

    def test_weights_sum_to_one(self):
        """Test that 3D weights sum to 1."""
        pts, wts = gauss_legendre_3d(2, 2, 2)
        assert_almost_equal(np.sum(wts), 1.0, decimal=14)

    def test_points_shape(self):
        """Test shape of 3D quadrature points."""
        pts, wts = gauss_legendre_3d(2, 3, 4)
        assert pts.shape == (24, 3)
        assert wts.shape == (24,)

    def test_integrate_constant(self):
        """Test 3D integration of constant."""
        pts, wts = gauss_legendre_3d(2, 2, 2)
        c = 1.5
        result = np.sum(c * wts)
        assert_almost_equal(result, c)


class TestGaussQuadratureClass:
    """Tests for GaussQuadrature class."""

    def test_1d_quadrature(self):
        """Test 1D quadrature via class."""
        quad = GaussQuadrature((3,))
        assert quad.n_dim == 1
        assert quad.n_points == 3
        assert quad.points.shape == (3, 1)

    def test_2d_quadrature(self):
        """Test 2D quadrature via class."""
        quad = GaussQuadrature((3, 4))
        assert quad.n_dim == 2
        assert quad.n_points == 12
        assert quad.points.shape == (12, 2)

    def test_for_degree(self):
        """Test automatic quadrature from degree."""
        # Full integration rule: p+1 points per direction
        quad = GaussQuadrature.for_degree((2, 3), rule="full")
        assert quad.n_points_per_dir == (3, 4)

        # Reduced integration rule: p points per direction
        quad = GaussQuadrature.for_degree((2, 3), rule="reduced")
        assert quad.n_points_per_dir == (2, 3)

    def test_properties(self):
        """Test quadrature properties."""
        quad = GaussQuadrature((3, 3))

        assert quad.n_dim == 2
        assert quad.n_points == 9
        assert quad.points.shape == (9, 2)
        assert quad.weights.shape == (9,)
        assert_almost_equal(np.sum(quad.weights), 1.0)

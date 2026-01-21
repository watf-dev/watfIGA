"""
Unit tests for NURBS geometry.
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal

from watfIGA.discretization.knot_vector import make_open_knot_vector
from watfIGA.geometry.nurbs import (
    NURBSCurve, NURBSSurface,
    make_nurbs_unit_square, make_nurbs_rectangle
)


class TestNURBSCurve:
    """Tests for NURBS curves."""

    def test_bspline_curve_endpoints(self):
        """Test that B-spline curve interpolates endpoints."""
        kv = make_open_knot_vector(n_basis=4, degree=2, domain=(0.0, 1.0))
        control_points = np.array([
            [0.0, 0.0],
            [0.5, 1.0],
            [1.0, 1.0],
            [1.5, 0.0]
        ])

        curve = NURBSCurve(kv, control_points)

        # Should interpolate first and last control points
        p0 = curve.eval_point(0.0)
        p1 = curve.eval_point(1.0)

        assert_array_almost_equal(p0, [0.0, 0.0])
        assert_array_almost_equal(p1, [1.5, 0.0])

    def test_bspline_line(self):
        """Test that linear control points give a line."""
        kv = make_open_knot_vector(n_basis=3, degree=2, domain=(0.0, 1.0))
        control_points = np.array([
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 1.0]
        ])

        curve = NURBSCurve(kv, control_points)

        # All points should lie on line y = x
        for xi in [0.0, 0.25, 0.5, 0.75, 1.0]:
            p = curve.eval_point(xi)
            assert_almost_equal(p[0], p[1], decimal=10)

    def test_curve_derivatives(self):
        """Test curve derivative computation."""
        kv = make_open_knot_vector(n_basis=3, degree=2, domain=(0.0, 1.0))
        # Straight line from (0,0) to (2,4)
        control_points = np.array([
            [0.0, 0.0],
            [1.0, 2.0],
            [2.0, 4.0]
        ])

        curve = NURBSCurve(kv, control_points)
        C, dC = curve.eval_derivatives(0.5, n_ders=1)

        # For a line, derivative should be constant
        # Direction is (2, 4), but scaled by curve parameterization
        assert dC[0] / dC[1] == pytest.approx(0.5, rel=1e-10)

    def test_nurbs_curve_with_weights(self):
        """Test NURBS curve with non-unit weights."""
        kv = make_open_knot_vector(n_basis=3, degree=2, domain=(0.0, 1.0))
        control_points = np.array([
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 0.0]
        ])
        # Higher weight at middle pushes curve toward middle control point
        weights = np.array([1.0, 2.0, 1.0])

        curve = NURBSCurve(kv, control_points, weights)

        # At midpoint, should be closer to middle CP than B-spline would be
        p = curve.eval_point(0.5)
        assert p[1] > 0.25  # B-spline would give 0.25 here


class TestNURBSSurface:
    """Tests for NURBS surfaces."""

    def test_unit_square_identity_mapping(self):
        """Test that unit square maps identity: (xi, eta) -> (xi, eta)."""
        surface = make_nurbs_unit_square(p=2, n_elem_xi=2, n_elem_eta=2)

        # Test at various points
        for xi in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for eta in [0.0, 0.25, 0.5, 0.75, 1.0]:
                point = surface.eval_point((xi, eta))
                assert_almost_equal(point[0], xi, decimal=10)
                assert_almost_equal(point[1], eta, decimal=10)

    def test_unit_square_corners(self):
        """Test unit square corner mapping."""
        surface = make_nurbs_unit_square(p=2, n_elem_xi=2, n_elem_eta=2)

        corners = [
            ((0.0, 0.0), [0.0, 0.0]),
            ((1.0, 0.0), [1.0, 0.0]),
            ((0.0, 1.0), [0.0, 1.0]),
            ((1.0, 1.0), [1.0, 1.0]),
        ]

        for (xi, eta), expected in corners:
            point = surface.eval_point((xi, eta))
            assert_array_almost_equal(point, expected)

    def test_unit_square_jacobian(self):
        """Test that unit square has identity Jacobian."""
        surface = make_nurbs_unit_square(p=2, n_elem_xi=2, n_elem_eta=2)

        for xi in [0.25, 0.5, 0.75]:
            for eta in [0.25, 0.5, 0.75]:
                S, dS_dxi, dS_deta = surface.eval_derivatives((xi, eta))

                # For identity mapping, Jacobian should be identity
                assert_array_almost_equal(dS_dxi, [1.0, 0.0])
                assert_array_almost_equal(dS_deta, [0.0, 1.0])

    def test_rectangle(self):
        """Test rectangle geometry."""
        surface = make_nurbs_rectangle(
            x_range=(0.0, 2.0),
            y_range=(0.0, 3.0),
            p=2, n_elem_xi=2, n_elem_eta=2
        )

        # Check corners
        assert_array_almost_equal(surface.eval_point((0.0, 0.0)), [0.0, 0.0])
        assert_array_almost_equal(surface.eval_point((1.0, 0.0)), [2.0, 0.0])
        assert_array_almost_equal(surface.eval_point((0.0, 1.0)), [0.0, 3.0])
        assert_array_almost_equal(surface.eval_point((1.0, 1.0)), [2.0, 3.0])

        # Check center
        assert_array_almost_equal(surface.eval_point((0.5, 0.5)), [1.0, 1.5])

    def test_surface_properties(self):
        """Test basic surface properties."""
        surface = make_nurbs_unit_square(p=2, n_elem_xi=4, n_elem_eta=3)

        assert surface.n_dim_parametric == 2
        assert surface.n_dim_physical == 2
        assert surface.degrees == (2, 2)
        assert surface.n_control_points_per_dir == (6, 5)
        assert surface.n_control_points == 30
        assert surface.domain == ((0.0, 1.0), (0.0, 1.0))

    def test_control_points_grid(self):
        """Test control points grid access."""
        surface = make_nurbs_unit_square(p=2, n_elem_xi=2, n_elem_eta=2)

        grid = surface.control_points_grid
        assert grid.shape == (4, 4, 2)

        # Corners of control point grid should match geometry corners
        assert_array_almost_equal(grid[0, 0], [0.0, 0.0])
        assert_array_almost_equal(grid[-1, -1], [1.0, 1.0])

    def test_rational_basis_partition_of_unity(self):
        """Test that rational basis sums to 1."""
        surface = make_nurbs_unit_square(p=2, n_elem_xi=2, n_elem_eta=2)

        for xi in [0.25, 0.5, 0.75]:
            for eta in [0.25, 0.5, 0.75]:
                R, indices = surface.eval_rational_basis((xi, eta))
                assert_almost_equal(np.sum(R), 1.0, decimal=12)

    def test_weights_default_to_one(self):
        """Test that default weights are all 1 (B-spline)."""
        surface = make_nurbs_unit_square(p=2, n_elem_xi=2, n_elem_eta=2)

        assert_array_almost_equal(surface.weights, np.ones(surface.n_control_points))


class TestNURBSSurfaceDerivatives:
    """Tests for NURBS surface derivatives."""

    def test_derivatives_at_center(self):
        """Test derivative computation at center."""
        surface = make_nurbs_unit_square(p=2, n_elem_xi=2, n_elem_eta=2)

        S, dS_dxi, dS_deta = surface.eval_derivatives((0.5, 0.5))

        assert_array_almost_equal(S, [0.5, 0.5])
        assert_array_almost_equal(dS_dxi, [1.0, 0.0])
        assert_array_almost_equal(dS_deta, [0.0, 1.0])

    def test_rational_basis_derivatives(self):
        """Test rational basis derivative computation."""
        surface = make_nurbs_unit_square(p=2, n_elem_xi=2, n_elem_eta=2)

        R, dR_dxi, dR_deta, indices = surface.eval_rational_basis(
            (0.5, 0.5), return_derivatives=True
        )

        # Derivatives should sum to zero (derivative of partition of unity)
        assert_almost_equal(np.sum(dR_dxi), 0.0, decimal=10)
        assert_almost_equal(np.sum(dR_deta), 0.0, decimal=10)

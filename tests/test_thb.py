#!/usr/bin/env python3
"""
Tests for THB-splines (Truncated Hierarchical B-splines).

Created: 2025-01-19
Author: Wataru Fukuda
"""

import numpy as np
import pytest

from watfIGA.discretization.knot_vector import (
    KnotVector, make_open_knot_vector,
    compute_knot_insertion_matrix, refine_knot_vector_dyadic,
    compute_refinement_matrix, compute_multiplicity
)
from watfIGA.geometry.thb import THBHierarchy1D, THBHierarchy2D, THBSurface
from watfIGA.geometry.nurbs import make_nurbs_unit_square
from watfIGA.discretization.mesh import Mesh, build_thb_mesh


class TestKnotInsertion:
    """Tests for knot insertion utilities."""

    def test_insertion_matrix_shape(self):
        """Test knot insertion matrix has correct shape."""
        kv = make_open_knot_vector(5, degree=2)
        xi = 0.5
        new_kv, A = compute_knot_insertion_matrix(kv, xi)

        assert A.shape == (kv.n_basis + 1, kv.n_basis)
        assert new_kv.n_basis == kv.n_basis + 1

    def test_insertion_preserves_basis_sum(self):
        """Test that knot insertion matrix has row sums = 1 (affine combination)."""
        kv = make_open_knot_vector(6, degree=2)
        xi = 0.25

        new_kv, A = compute_knot_insertion_matrix(kv, xi)

        # Row sums should be 1 (each new control point is affine combination of old)
        row_sums = np.sum(A, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)

    def test_multiplicity_computation(self):
        """Test knot multiplicity computation."""
        kv = make_open_knot_vector(5, degree=2)

        # Boundary knots have multiplicity p+1 = 3
        assert compute_multiplicity(kv, 0.0) == 3
        assert compute_multiplicity(kv, 1.0) == 3

        # Interior knots have multiplicity 1
        for knot in kv.unique_knots[1:-1]:
            assert compute_multiplicity(kv, knot) == 1

    def test_dyadic_refinement(self):
        """Test dyadic refinement doubles elements."""
        kv = make_open_knot_vector(5, degree=2)
        n_elem_before = kv.n_elements

        refined_kv, T = refine_knot_vector_dyadic(kv)
        n_elem_after = refined_kv.n_elements

        # Dyadic refinement doubles the number of elements
        assert n_elem_after == 2 * n_elem_before

    def test_refinement_matrix_nesting(self):
        """Test refinement matrix respects knot nesting."""
        kv_coarse = make_open_knot_vector(4, degree=2)
        kv_fine, T = refine_knot_vector_dyadic(kv_coarse)

        # All coarse knots must be in fine knot vector
        for knot in kv_coarse.knots:
            assert np.any(np.abs(kv_fine.knots - knot) < 1e-14)


class TestTHBHierarchy1D:
    """Tests for 1D hierarchical structure."""

    def test_creation_from_knot_vector(self):
        """Test hierarchy creation from initial knot vector."""
        kv = make_open_knot_vector(5, degree=2)
        hierarchy = THBHierarchy1D.from_knot_vector(kv)

        assert hierarchy.n_levels == 1
        assert hierarchy.degree == 2
        assert len(hierarchy.active_basis[0]) == kv.n_basis

    def test_add_level(self):
        """Test adding refinement level."""
        kv = make_open_knot_vector(5, degree=2)
        hierarchy = THBHierarchy1D.from_knot_vector(kv)

        hierarchy.add_level()

        assert hierarchy.n_levels == 2
        assert hierarchy.get_n_basis(1) > hierarchy.get_n_basis(0)

    def test_ensure_level(self):
        """Test ensure_level creates required levels."""
        kv = make_open_knot_vector(4, degree=2)
        hierarchy = THBHierarchy1D.from_knot_vector(kv)

        hierarchy.ensure_level(3)

        assert hierarchy.n_levels == 4

    def test_basis_support(self):
        """Test basis function support computation."""
        kv = make_open_knot_vector(5, degree=2)
        hierarchy = THBHierarchy1D.from_knot_vector(kv)

        # First basis function support
        xi_min, xi_max = hierarchy.get_basis_support(0, 0)
        assert xi_min == 0.0
        assert xi_max > 0.0

        # Last basis function support
        xi_min, xi_max = hierarchy.get_basis_support(0, kv.n_basis - 1)
        assert xi_max == 1.0

    def test_activate_deactivate_basis(self):
        """Test basis activation/deactivation."""
        kv = make_open_knot_vector(5, degree=2)
        hierarchy = THBHierarchy1D.from_knot_vector(kv)

        # Deactivate some basis functions
        to_deactivate = {0, 1}
        hierarchy.deactivate_basis(0, to_deactivate)

        assert 0 not in hierarchy.active_basis[0]
        assert 1 not in hierarchy.active_basis[0]
        assert 2 in hierarchy.active_basis[0]

    def test_overlapping_fine_basis(self):
        """Test finding overlapping fine-level basis functions."""
        kv = make_open_knot_vector(4, degree=2)
        hierarchy = THBHierarchy1D.from_knot_vector(kv)
        hierarchy.add_level()

        # Find fine basis functions overlapping with first coarse basis
        overlapping = hierarchy.find_overlapping_fine_basis(0, 0)

        # Should find some overlapping basis
        assert len(overlapping) > 0


class TestTHBHierarchy2D:
    """Tests for 2D hierarchical structure."""

    def test_creation_from_knot_vectors(self):
        """Test 2D hierarchy creation."""
        kv_xi = make_open_knot_vector(4, degree=2)
        kv_eta = make_open_knot_vector(3, degree=2)

        hierarchy = THBHierarchy2D.from_knot_vectors(kv_xi, kv_eta)

        assert hierarchy.n_levels == 1
        assert hierarchy.degrees == (2, 2)

    def test_tensor_index_conversion(self):
        """Test tensor product index conversion."""
        kv_xi = make_open_knot_vector(4, degree=2)
        kv_eta = make_open_knot_vector(3, degree=2)
        hierarchy = THBHierarchy2D.from_knot_vectors(kv_xi, kv_eta)

        n_xi = kv_xi.n_basis

        # Test round-trip conversion
        for j in range(kv_eta.n_basis):
            for i in range(kv_xi.n_basis):
                global_idx = hierarchy.tensor_to_global(0, i, j)
                i_back, j_back = hierarchy.global_to_tensor(0, global_idx)
                assert i == i_back
                assert j == j_back

    def test_element_bounds(self):
        """Test element bounds computation."""
        kv_xi = make_open_knot_vector(4, degree=2)
        kv_eta = make_open_knot_vector(4, degree=2)
        hierarchy = THBHierarchy2D.from_knot_vectors(kv_xi, kv_eta)

        xi_bounds, eta_bounds = hierarchy.get_element_bounds(0, 0, 0)

        # First element should start at origin
        assert xi_bounds[0] == 0.0
        assert eta_bounds[0] == 0.0


class TestTHBSurface:
    """Tests for THB surface."""

    def test_creation_from_nurbs(self):
        """Test THB surface creation from NURBS."""
        surface = make_nurbs_unit_square(p=2, n_elem_xi=2, n_elem_eta=2)
        thb = THBSurface.from_nurbs_surface(surface)

        assert thb.n_levels == 1
        assert thb.degrees == (2, 2)
        assert thb.n_dim_parametric == 2
        assert thb.n_dim_physical == 2

    def test_control_point_access(self):
        """Test control point access."""
        surface = make_nurbs_unit_square(p=2, n_elem_xi=2, n_elem_eta=2)
        thb = THBSurface.from_nurbs_surface(surface)

        cp = thb.get_control_points(0)
        w = thb.get_weights(0)

        assert cp.shape[0] == surface.n_control_points
        assert w.shape[0] == surface.n_control_points

    def test_refine_element(self):
        """Test element refinement."""
        surface = make_nurbs_unit_square(p=2, n_elem_xi=2, n_elem_eta=2)
        thb = THBSurface.from_nurbs_surface(surface)

        # Refine element (0, 0)
        thb.refine_element(0, 0, 0)

        assert thb.n_levels == 2
        assert thb.is_element_refined(0, 0, 0)
        assert not thb.is_element_refined(0, 1, 0)

    def test_refined_control_points(self):
        """Test refined control points are computed."""
        surface = make_nurbs_unit_square(p=2, n_elem_xi=2, n_elem_eta=2)
        thb = THBSurface.from_nurbs_surface(surface)

        thb.refine_element(0, 0, 0)

        # Level 1 control points should exist
        cp_1 = thb.get_control_points(1)
        assert cp_1.shape[0] > surface.n_control_points

    def test_active_control_points_after_refinement(self):
        """Test active control points update after refinement."""
        surface = make_nurbs_unit_square(p=2, n_elem_xi=2, n_elem_eta=2)
        thb = THBSurface.from_nurbs_surface(surface)

        n_active_before = thb.get_total_active_dofs()

        thb.refine_element(0, 0, 0)
        thb.finalize_refinement()

        # After refinement, some fine-level control points should be active
        active_level_1 = thb.get_active_control_points(1)
        assert len(active_level_1) > 0

    def test_finalize_refinement_truncation(self):
        """Test that finalize_refinement applies truncation."""
        surface = make_nurbs_unit_square(p=2, n_elem_xi=2, n_elem_eta=2)
        thb = THBSurface.from_nurbs_surface(surface)

        n_active_0_before = len(thb.get_active_control_points(0))

        # Refine multiple adjacent elements to fully cover some coarse basis support
        thb.refine_element(0, 0, 0)
        thb.refine_element(0, 1, 0)
        thb.refine_element(0, 0, 1)
        thb.refine_element(0, 1, 1)  # All 4 elements refined

        thb.finalize_refinement()

        # Some coarse basis functions should be truncated (deactivated)
        n_active_0_after = len(thb.get_active_control_points(0))
        assert n_active_0_after < n_active_0_before


class TestBuildTHBMesh:
    """Tests for THB mesh building."""

    def test_build_unrefined_thb_mesh(self):
        """Test building mesh from unrefined THB surface."""
        surface = make_nurbs_unit_square(p=2, n_elem_xi=2, n_elem_eta=2)
        thb = THBSurface.from_nurbs_surface(surface)

        mesh = Mesh.build(thb)

        # Should have same number of elements as original
        assert mesh.n_elements == 4  # 2x2 elements
        assert mesh.n_control_points == surface.n_control_points

    def test_build_refined_thb_mesh(self):
        """Test building mesh with one refined element."""
        surface = make_nurbs_unit_square(p=2, n_elem_xi=2, n_elem_eta=2)
        thb = THBSurface.from_nurbs_surface(surface)

        # Refine one element
        thb.refine_element(0, 0, 0)

        mesh = Mesh.build(thb)

        # One coarse element replaced by 4 fine elements
        # Total: 3 coarse + 4 fine = 7 elements
        assert mesh.n_elements == 7

    def test_mesh_elements_have_geometry(self):
        """Test mesh elements have geometry cached."""
        surface = make_nurbs_unit_square(p=2, n_elem_xi=2, n_elem_eta=2)
        thb = THBSurface.from_nurbs_surface(surface)

        mesh = Mesh.build(thb, include_geometry=True)

        for elem in mesh.get_active_elements():
            assert elem.control_points is not None
            assert elem.weights is not None

    def test_mesh_bidirectional_linking(self):
        """Test mesh has valid bidirectional linking."""
        surface = make_nurbs_unit_square(p=2, n_elem_xi=2, n_elem_eta=2)
        thb = THBSurface.from_nurbs_surface(surface)
        thb.refine_element(0, 0, 0)

        # Should not raise
        mesh = Mesh.build(thb)

        # Verify linking
        for elem in mesh.get_active_elements():
            for cp_id in elem.control_point_ids:
                cp = mesh.get_control_point(cp_id)
                assert elem.id in cp.supported_elements

    def test_refined_elements_at_correct_level(self):
        """Test refined elements have correct level."""
        surface = make_nurbs_unit_square(p=2, n_elem_xi=2, n_elem_eta=2)
        thb = THBSurface.from_nurbs_surface(surface)
        thb.refine_element(0, 0, 0)

        mesh = Mesh.build(thb)

        # Count elements at each level
        level_0_elems = [e for e in mesh.get_active_elements() if e.level == 0]
        level_1_elems = [e for e in mesh.get_active_elements() if e.level == 1]

        assert len(level_0_elems) == 3  # 3 unrefined
        assert len(level_1_elems) == 4  # 4 child elements from refinement

    def test_max_level_tracking(self):
        """Test mesh tracks max refinement level."""
        surface = make_nurbs_unit_square(p=2, n_elem_xi=2, n_elem_eta=2)
        thb = THBSurface.from_nurbs_surface(surface)

        mesh_unrefined = Mesh.build(thb)
        assert mesh_unrefined.max_level == 0

        thb.refine_element(0, 0, 0)
        mesh_refined = Mesh.build(thb)
        assert mesh_refined.max_level == 1


class TestTHBMeshIntegration:
    """Integration tests for THB mesh with solver components."""

    def test_extraction_operators_shape(self):
        """Test extraction operators have correct shape."""
        surface = make_nurbs_unit_square(p=2, n_elem_xi=2, n_elem_eta=2)
        thb = THBSurface.from_nurbs_surface(surface)
        thb.refine_element(0, 0, 0)

        mesh = Mesh.build(thb)

        for elem in mesh.get_active_elements():
            p_xi, p_eta = elem.degrees
            expected_rows = (p_xi + 1) * (p_eta + 1)
            assert elem.extraction_operator.shape[1] == expected_rows

    def test_domain_coverage(self):
        """Test refined mesh still covers entire domain."""
        surface = make_nurbs_unit_square(p=2, n_elem_xi=2, n_elem_eta=2)
        thb = THBSurface.from_nurbs_surface(surface)
        thb.refine_element(0, 0, 0)
        thb.refine_element(0, 1, 1)

        mesh = Mesh.build(thb)

        # Check all elements have valid parametric bounds
        for elem in mesh.get_active_elements():
            for bounds in elem.parametric_bounds:
                xi_min, xi_max = bounds
                assert xi_min >= 0.0
                assert xi_max <= 1.0
                assert xi_min < xi_max

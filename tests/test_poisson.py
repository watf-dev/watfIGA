"""
Integration tests for Poisson equation solver.
"""

import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from watfIGA.geometry.nurbs import make_nurbs_unit_square, make_nurbs_rectangle
from watfIGA.discretization.mesh import (
    build_mesh_2d, get_all_boundary_dofs_2d, get_boundary_dofs_2d, DirichletBC
)
from watfIGA.solver.poisson import PoissonSolver
from watfIGA.postprocess.sampling import sample_solution_2d, compute_l2_error


class TestPoissonSolverBasic:
    """Basic tests for Poisson solver."""

    def test_solver_creates(self):
        """Test that solver can be created."""
        surface = make_nurbs_unit_square(p=2, n_elem_xi=2, n_elem_eta=2)
        mesh = build_mesh_2d(surface)
        solver = PoissonSolver(mesh, source=lambda x, y: 1.0)

        assert solver.n_dof == mesh.n_dof
        assert solver.mesh is mesh

    def test_assembly(self):
        """Test system assembly."""
        surface = make_nurbs_unit_square(p=2, n_elem_xi=2, n_elem_eta=2)
        mesh = build_mesh_2d(surface)
        solver = PoissonSolver(mesh, source=lambda x, y: 1.0)
        solver.assemble()

        assert solver.K is not None
        assert solver.f is not None
        assert solver.K.shape == (mesh.n_dof, mesh.n_dof)
        assert solver.f.shape == (mesh.n_dof,)

    def test_stiffness_symmetric(self):
        """Test that stiffness matrix is symmetric."""
        surface = make_nurbs_unit_square(p=2, n_elem_xi=2, n_elem_eta=2)
        mesh = build_mesh_2d(surface)
        solver = PoissonSolver(mesh, source=lambda x, y: 1.0)
        solver.assemble()

        K_dense = solver.K.toarray()
        assert_array_almost_equal(K_dense, K_dense.T, decimal=10)

    def test_stiffness_positive_definite(self):
        """Test stiffness matrix properties (semi-positive definite before BCs)."""
        surface = make_nurbs_unit_square(p=2, n_elem_xi=2, n_elem_eta=2)
        mesh = build_mesh_2d(surface)
        solver = PoissonSolver(mesh, source=lambda x, y: 1.0)
        solver.assemble()

        # Eigenvalues should be non-negative
        eigenvalues = np.linalg.eigvalsh(solver.K.toarray())
        assert np.all(eigenvalues >= -1e-10)


class TestPoissonManufacturedSolution:
    """Tests using manufactured solutions."""

    def test_sin_sin_solution(self):
        """Test with u = sin(πx)sin(πy)."""
        def u_exact(x, y):
            return np.sin(np.pi * x) * np.sin(np.pi * y)

        def source(x, y):
            return 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

        surface = make_nurbs_unit_square(p=2, n_elem_xi=4, n_elem_eta=4)
        mesh = build_mesh_2d(surface)

        solver = PoissonSolver(mesh, source=source)
        boundary_dofs = get_all_boundary_dofs_2d(surface)
        bc = DirichletBC.homogeneous(boundary_dofs)
        solver.add_dirichlet_bc(bc)
        u = solver.run()

        # Check L2 error is reasonably small
        l2_error = compute_l2_error(surface, u, u_exact, n_sample=50)
        assert l2_error < 0.01  # Should be ~0.002 for this mesh

    def test_polynomial_solution(self):
        """Test with polynomial u = x(1-x)y(1-y)."""
        def u_exact(x, y):
            return x * (1 - x) * y * (1 - y)

        def source(x, y):
            # -Δu = -∂²u/∂x² - ∂²u/∂y²
            # u = x(1-x)y(1-y)
            # ∂u/∂x = (1-2x)y(1-y)
            # ∂²u/∂x² = -2y(1-y)
            # Similarly ∂²u/∂y² = -2x(1-x)
            # -Δu = 2y(1-y) + 2x(1-x)
            return 2 * y * (1 - y) + 2 * x * (1 - x)

        surface = make_nurbs_unit_square(p=3, n_elem_xi=4, n_elem_eta=4)
        mesh = build_mesh_2d(surface)

        solver = PoissonSolver(mesh, source=source)
        boundary_dofs = get_all_boundary_dofs_2d(surface)
        bc = DirichletBC.homogeneous(boundary_dofs)
        solver.add_dirichlet_bc(bc)
        u = solver.run()

        l2_error = compute_l2_error(surface, u, u_exact, n_sample=50)
        # Polynomial should be captured very accurately with p=3
        assert l2_error < 1e-6


class TestPoissonConvergence:
    """Convergence tests for Poisson solver."""

    def test_h_convergence_p2(self):
        """Test h-convergence for degree 2."""
        def u_exact(x, y):
            return np.sin(np.pi * x) * np.sin(np.pi * y)

        def source(x, y):
            return 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

        errors = []
        for n_elem in [2, 4, 8]:
            surface = make_nurbs_unit_square(p=2, n_elem_xi=n_elem, n_elem_eta=n_elem)
            mesh = build_mesh_2d(surface)

            solver = PoissonSolver(mesh, source=source)
            bc = DirichletBC.homogeneous(get_all_boundary_dofs_2d(surface))
            solver.add_dirichlet_bc(bc)
            u = solver.run()

            l2_error = compute_l2_error(surface, u, u_exact, n_sample=50)
            errors.append(l2_error)

        # Check convergence rate ≈ p+1 = 3
        rate1 = np.log(errors[0] / errors[1]) / np.log(2)
        rate2 = np.log(errors[1] / errors[2]) / np.log(2)

        assert rate1 > 2.5  # Should be close to 3
        assert rate2 > 2.5

    def test_h_convergence_p3(self):
        """Test h-convergence for degree 3."""
        def u_exact(x, y):
            return np.sin(np.pi * x) * np.sin(np.pi * y)

        def source(x, y):
            return 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

        errors = []
        for n_elem in [2, 4, 8]:
            surface = make_nurbs_unit_square(p=3, n_elem_xi=n_elem, n_elem_eta=n_elem)
            mesh = build_mesh_2d(surface)

            solver = PoissonSolver(mesh, source=source)
            bc = DirichletBC.homogeneous(get_all_boundary_dofs_2d(surface))
            solver.add_dirichlet_bc(bc)
            u = solver.run()

            l2_error = compute_l2_error(surface, u, u_exact, n_sample=50)
            errors.append(l2_error)

        # Check convergence rate ≈ p+1 = 4
        # Note: first rate may be lower due to preasymptotic regime on coarse mesh
        rate1 = np.log(errors[0] / errors[1]) / np.log(2)
        rate2 = np.log(errors[1] / errors[2]) / np.log(2)

        assert rate1 > 2.8  # Preasymptotic, may be lower
        assert rate2 > 3.5  # Should be close to 4 in asymptotic regime


class TestBoundaryConditions:
    """Tests for boundary condition handling."""

    def test_homogeneous_bc(self):
        """Test homogeneous Dirichlet BC enforcement."""
        surface = make_nurbs_unit_square(p=2, n_elem_xi=2, n_elem_eta=2)
        mesh = build_mesh_2d(surface)

        solver = PoissonSolver(mesh, source=lambda x, y: 1.0)
        boundary_dofs = get_all_boundary_dofs_2d(surface)
        bc = DirichletBC.homogeneous(boundary_dofs)
        solver.add_dirichlet_bc(bc)
        u = solver.run()

        # Solution should be zero on boundary
        assert_array_almost_equal(u[boundary_dofs], 0.0)

    def test_nonhomogeneous_bc(self):
        """Test non-homogeneous Dirichlet BC."""
        surface = make_nurbs_unit_square(p=2, n_elem_xi=2, n_elem_eta=2)
        mesh = build_mesh_2d(surface)

        solver = PoissonSolver(mesh, source=lambda x, y: 0.0)

        # Set u = 1 on left boundary (x=0)
        left_dofs = get_boundary_dofs_2d(surface, "left")
        bc_left = DirichletBC(left_dofs, np.ones(len(left_dofs)))
        solver.add_dirichlet_bc(bc_left)

        # Set u = 0 on other boundaries
        for side in ["right", "bottom", "top"]:
            dofs = get_boundary_dofs_2d(surface, side)
            # Remove overlap with left
            dofs = np.setdiff1d(dofs, left_dofs)
            if len(dofs) > 0:
                bc = DirichletBC.homogeneous(dofs)
                solver.add_dirichlet_bc(bc)

        u = solver.run()

        # Check left boundary is 1
        assert_array_almost_equal(u[left_dofs], 1.0)


class TestMeshWithGeometry:
    """Tests for mesh building and geometry."""

    def test_mesh_properties(self):
        """Test mesh properties are correct."""
        surface = make_nurbs_unit_square(p=2, n_elem_xi=4, n_elem_eta=3)
        mesh = build_mesh_2d(surface)

        assert mesh.n_dof == 6 * 5  # (4+2) x (3+2)
        assert mesh.n_elements == 12  # 4 x 3
        assert mesh.n_dim_parametric == 2
        assert mesh.n_dim_physical == 2

    def test_element_properties(self):
        """Test element properties."""
        surface = make_nurbs_unit_square(p=2, n_elem_xi=2, n_elem_eta=2)
        mesh = build_mesh_2d(surface)

        elem = mesh.elements[0]
        assert elem.degrees == (2, 2)
        assert elem.n_local_basis == 9  # (p+1)^2
        assert len(elem.global_dof_indices) == 9
        assert elem.control_points.shape == (9, 2)


class TestSolutionSampling:
    """Tests for solution post-processing."""

    def test_sampling_shape(self):
        """Test solution sampling output shape."""
        surface = make_nurbs_unit_square(p=2, n_elem_xi=2, n_elem_eta=2)
        mesh = build_mesh_2d(surface)
        u = np.ones(mesh.n_dof)

        X, Y, U = sample_solution_2d(surface, u, n_xi=10, n_eta=15)

        assert X.shape == (10, 15)
        assert Y.shape == (10, 15)
        assert U.shape == (10, 15)

    def test_sampling_constant(self):
        """Test sampling of constant solution."""
        surface = make_nurbs_unit_square(p=2, n_elem_xi=2, n_elem_eta=2)
        mesh = build_mesh_2d(surface)
        u = 2.5 * np.ones(mesh.n_dof)

        X, Y, U = sample_solution_2d(surface, u, n_xi=10, n_eta=10)

        # All values should be 2.5
        assert_array_almost_equal(U, 2.5 * np.ones_like(U))

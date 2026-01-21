"""
Poisson equation solver for IGA.

Solves the scalar Poisson equation:
    -∇²u = f    in Ω
        u = g    on Γ_D (Dirichlet boundary)

Weak form:
    ∫_Ω ∇u · ∇v dΩ = ∫_Ω f v dΩ    for all v in V_0

where V_0 is the space of test functions vanishing on Γ_D.

Element stiffness matrix:
    K_{ij} = ∫_e ∇N_i · ∇N_j dΩ

Element load vector:
    f_i = ∫_e f N_i dΩ

Assembly follows the IGA paradigm:
1. Evaluate Bernstein basis and derivatives at quadrature points
2. Map derivatives to physical coordinates via Jacobian
3. Integrate in physical space (with |J| weight)
4. Apply extraction operator to get spline basis contribution

This solver is completely independent of the spline type used.
"""

import numpy as np
from typing import Tuple, Callable, Optional

from .base import (Solver, eval_geometry_at_quadrature_point,
                   compute_shape_function_derivatives)
from ..discretization.mesh import Mesh, DirichletBC
from ..discretization.element import Element
from ..discretization.extraction import BernsteinBasis
from ..quadrature.gauss import GaussQuadrature


class PoissonSolver(Solver):
    """
    Solver for the Poisson equation.

    Example usage:
        # Create geometry and mesh
        surface = make_nurbs_unit_square(p=2, n_elem_xi=4, n_elem_eta=4)
        mesh = build_mesh_2d(surface)

        # Create solver with source term
        solver = PoissonSolver(mesh, source=lambda x, y: 1.0)

        # Add boundary conditions
        bc = DirichletBC.homogeneous(get_all_boundary_dofs_2d(surface))
        solver.add_dirichlet_bc(bc)

        # Solve
        u = solver.run()
    """

    def __init__(self, mesh: Mesh,
                 source: Optional[Callable[[float, float], float]] = None,
                 diffusivity: float = 1.0):
        """
        Initialize Poisson solver.

        Parameters:
            mesh: Mesh from build_mesh_2d
            source: Source function f(x, y), defaults to 0
            diffusivity: Diffusion coefficient (scalar), defaults to 1
        """
        super().__init__(mesh)
        self.source = source if source is not None else lambda x, y: 0.0
        self.diffusivity = diffusivity

    def compute_element_matrices(self, element: Element,
                                  quadrature: GaussQuadrature) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute element stiffness matrix and load vector for Poisson equation.

        This method follows the Bézier extraction paradigm:
        1. Evaluate Bernstein basis and derivatives at quadrature points
        2. Compute geometry (Jacobian) using spline basis via extraction
        3. Transform Bernstein derivatives to physical coordinates
        4. Assemble in Bernstein basis: K_bern, f_bern
        5. The extraction operator will be applied in assemble(): K_e = C^T K_bern C

        Parameters:
            element: Element with geometry
            quadrature: Quadrature rule

        Returns:
            (K_bern, f_bern) in Bernstein basis
        """
        degrees = element.degrees
        bernstein = BernsteinBasis(degrees)
        n_local = bernstein.n_basis

        # Initialize element matrices (in Bernstein basis)
        K_bern = np.zeros((n_local, n_local))
        f_bern = np.zeros(n_local)

        # Jacobian for reference to parametric mapping
        det_ref_to_param = element.det_jacobian_ref_to_param()

        # Quadrature loop
        for q in range(quadrature.n_points):
            # Reference coordinates
            t_ref = tuple(quadrature.points[q])
            w_q = quadrature.weights[q]

            # Evaluate Bernstein basis and derivatives on reference element
            B_vals, dB_dt_xi, dB_dt_eta = bernstein.eval_ders(t_ref, n_ders=1)

            # Scale derivatives from reference to parametric domain
            # dB/dxi_param = dB/dt * dt/dxi_param = dB/dt / h
            h_xi = element.parametric_bounds[0][1] - element.parametric_bounds[0][0]
            h_eta = element.parametric_bounds[1][1] - element.parametric_bounds[1][0]
            dB_dxi = dB_dt_xi / h_xi
            dB_deta = dB_dt_eta / h_eta

            # Evaluate geometry at this quadrature point (uses extraction internally)
            point, jacobian, det_jac_phys = eval_geometry_at_quadrature_point(
                element, B_vals, dB_dxi, dB_deta)

            # Transform Bernstein derivatives to physical coordinates
            # [dB/dx]   [dxi/dx  deta/dx] [dB/dxi ]
            # [dB/dy] = [dxi/dy  deta/dy] [dB/deta]
            det_jac = jacobian[0, 0] * jacobian[1, 1] - jacobian[0, 1] * jacobian[1, 0]
            inv_jac = np.array([
                [ jacobian[1, 1], -jacobian[0, 1]],
                [-jacobian[1, 0],  jacobian[0, 0]]
            ]) / det_jac

            # Physical derivatives of Bernstein basis
            dB_dx = dB_dxi * inv_jac[0, 0] + dB_deta * inv_jac[1, 0]
            dB_dy = dB_dxi * inv_jac[0, 1] + dB_deta * inv_jac[1, 1]

            # Integration weight: w_q * |J_ref_to_param| * |J_param_to_phys|
            dV = w_q * det_ref_to_param * abs(det_jac_phys)

            # Stiffness contribution in Bernstein basis:
            # K_bern_ij = ∫ (dB_i/dx * dB_j/dx + dB_i/dy * dB_j/dy) dV
            K_bern += self.diffusivity * np.outer(dB_dx, dB_dx) * dV
            K_bern += self.diffusivity * np.outer(dB_dy, dB_dy) * dV

            # Load contribution in Bernstein basis:
            # f_bern_i = ∫ f(x,y) * B_i dV
            f_val = self.source(point[0], point[1])
            f_bern += f_val * B_vals * dV

        return K_bern, f_bern


def solve_poisson_2d(mesh: Mesh,
                     source: Callable[[float, float], float],
                     dirichlet_dofs: np.ndarray,
                     dirichlet_values: Optional[np.ndarray] = None,
                     diffusivity: float = 1.0) -> np.ndarray:
    """
    Convenience function to solve 2D Poisson problem.

    Parameters:
        mesh: Mesh
        source: Source function f(x, y)
        dirichlet_dofs: DOF indices for Dirichlet BC
        dirichlet_values: Prescribed values (defaults to 0)
        diffusivity: Diffusion coefficient

    Returns:
        Solution vector u
    """
    solver = PoissonSolver(mesh, source=source, diffusivity=diffusivity)

    if dirichlet_values is None:
        bc = DirichletBC.homogeneous(dirichlet_dofs)
    else:
        bc = DirichletBC(dirichlet_dofs, dirichlet_values)

    solver.add_dirichlet_bc(bc)
    return solver.run()


def manufactured_solution_poisson(mesh: Mesh,
                                   u_exact: Callable[[float, float], float],
                                   grad_u_exact: Callable[[float, float], Tuple[float, float]]) -> Tuple[np.ndarray, float, float]:
    """
    Solve Poisson with manufactured solution for verification.

    Given an exact solution u_exact, computes:
    - Source term f = -∇²u_exact
    - Boundary values from u_exact
    - L2 and H1 errors

    Parameters:
        mesh: Mesh
        u_exact: Exact solution u(x, y)
        grad_u_exact: Gradient (du/dx, du/dy)

    Returns:
        (u_numerical, L2_error, H1_error)
    """
    # Compute Laplacian of exact solution numerically
    h = 1e-6
    def laplacian_u(x, y):
        u_xx = (u_exact(x+h, y) - 2*u_exact(x, y) + u_exact(x-h, y)) / (h*h)
        u_yy = (u_exact(x, y+h) - 2*u_exact(x, y) + u_exact(x, y-h)) / (h*h)
        return u_xx + u_yy

    def source(x, y):
        return -laplacian_u(x, y)

    # For now, use all boundary DOFs based on control point positions
    control_pts = mesh.control_points_array  # Use array for backward compat
    n_dof = mesh.n_dof
    tol = 1e-10

    # Find DOFs on boundary (assuming unit square domain)
    x_min, x_max = control_pts[:, 0].min(), control_pts[:, 0].max()
    y_min, y_max = control_pts[:, 1].min(), control_pts[:, 1].max()

    boundary_dofs = []
    for i in range(n_dof):
        x, y = control_pts[i, 0], control_pts[i, 1]
        if (abs(x - x_min) < tol or abs(x - x_max) < tol or
            abs(y - y_min) < tol or abs(y - y_max) < tol):
            boundary_dofs.append(i)

    boundary_dofs = np.array(boundary_dofs)
    boundary_values = np.array([u_exact(control_pts[i, 0], control_pts[i, 1])
                                for i in boundary_dofs])

    # Solve
    solver = PoissonSolver(mesh, source=source)
    bc = DirichletBC(boundary_dofs, boundary_values)
    solver.add_dirichlet_bc(bc)
    u_numerical = solver.run()

    # Compute errors (simplified - integrate over active elements)
    L2_error_sq = 0.0
    H1_error_sq = 0.0

    first_elem = next(mesh.get_active_elements())
    degrees = first_elem.degrees
    quadrature = GaussQuadrature(tuple(p + 2 for p in degrees))  # Extra points for accuracy
    bernstein = BernsteinBasis(degrees)

    for element in mesh.get_active_elements():
        det_ref_to_param = element.det_jacobian_ref_to_param()
        C_e = element.extraction_operator
        u_local = u_numerical[element.global_dof_indices]

        for q in range(quadrature.n_points):
            t_ref = tuple(quadrature.points[q])
            w_q = quadrature.weights[q]

            B_vals, dB_dxi, dB_deta = bernstein.eval_ders(t_ref, n_ders=1)

            h_xi = element.parametric_bounds[0][1] - element.parametric_bounds[0][0]
            h_eta = element.parametric_bounds[1][1] - element.parametric_bounds[1][0]
            dB_dxi_param = dB_dxi / h_xi
            dB_deta_param = dB_deta / h_eta

            point, jacobian, det_jac_phys = eval_geometry_at_quadrature_point(
                element, B_vals, dB_dxi_param, dB_deta_param)

            R, dR_dx, dR_dy = compute_shape_function_derivatives(
                element, B_vals, dB_dxi_param, dB_deta_param, jacobian)

            # Apply extraction to get spline basis
            R_spline = C_e.T @ R
            dR_dx_spline = C_e.T @ dR_dx
            dR_dy_spline = C_e.T @ dR_dy

            # Numerical solution and derivatives
            u_h = np.dot(R_spline, u_local)
            du_h_dx = np.dot(dR_dx_spline, u_local)
            du_h_dy = np.dot(dR_dy_spline, u_local)

            # Exact solution and derivatives
            x, y = point[0], point[1]
            u_ex = u_exact(x, y)
            du_ex_dx, du_ex_dy = grad_u_exact(x, y)

            dV = w_q * det_ref_to_param * abs(det_jac_phys)

            L2_error_sq += (u_h - u_ex)**2 * dV
            H1_error_sq += ((du_h_dx - du_ex_dx)**2 + (du_h_dy - du_ex_dy)**2) * dV

    L2_error = np.sqrt(L2_error_sq)
    H1_error = np.sqrt(L2_error_sq + H1_error_sq)

    return u_numerical, L2_error, H1_error

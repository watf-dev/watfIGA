"""
Solution sampling for visualization and post-processing.

This module provides utilities for evaluating IGA solutions on
regular grids for visualization purposes.

Key functions:
- sample_solution_2d: Evaluate solution on a uniform grid
- evaluate_at_point: Evaluate solution at a single point

The sampling uses the NURBS geometry directly (not Bézier extraction)
since we're evaluating the solution, not assembling matrices.

TODO: Add gradient/derivative sampling
TODO: Add adaptive sampling based on solution variation
"""

import numpy as np
from typing import Tuple, Optional
from ..geometry.nurbs import NURBSSurface
from ..geometry.bspline import eval_basis_ders_1d


def sample_solution_2d(surface: NURBSSurface,
                        u: np.ndarray,
                        n_xi: int = 50,
                        n_eta: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample solution on a uniform grid in parametric space.

    Parameters:
        surface: NURBS surface geometry
        u: Solution vector (control point values)
        n_xi: Number of sample points in xi direction
        n_eta: Number of sample points in eta direction

    Returns:
        (X, Y, U) where:
        - X: Physical x-coordinates, shape (n_xi, n_eta)
        - Y: Physical y-coordinates, shape (n_xi, n_eta)
        - U: Solution values, shape (n_xi, n_eta)
    """
    kv_xi, kv_eta = surface.knot_vectors
    domain_xi = kv_xi.domain
    domain_eta = kv_eta.domain

    # Create uniform grid in parametric space
    xi_vals = np.linspace(domain_xi[0], domain_xi[1], n_xi)
    eta_vals = np.linspace(domain_eta[0], domain_eta[1], n_eta)

    # Output arrays
    X = np.zeros((n_xi, n_eta))
    Y = np.zeros((n_xi, n_eta))
    U = np.zeros((n_xi, n_eta))

    p_xi = kv_xi.degree
    p_eta = kv_eta.degree
    n_basis_xi = kv_xi.n_basis
    n_basis_eta = kv_eta.n_basis

    weights = surface.weights
    control_points = surface.control_points

    for i, xi in enumerate(xi_vals):
        span_xi = kv_xi.find_span(xi)
        N_xi = eval_basis_ders_1d(kv_xi, xi, 0, span_xi)[0, :]

        for j, eta in enumerate(eta_vals):
            span_eta = kv_eta.find_span(eta)
            N_eta = eval_basis_ders_1d(kv_eta, eta, 0, span_eta)[0, :]

            # Compute NURBS point and solution value
            x_pt = np.zeros(surface.n_dim_physical)
            u_val = 0.0
            W = 0.0

            for ii in range(p_xi + 1):
                for jj in range(p_eta + 1):
                    glob_i = span_xi - p_xi + ii
                    glob_j = span_eta - p_eta + jj
                    global_idx = glob_i * n_basis_eta + glob_j

                    Nij = N_xi[ii] * N_eta[jj]
                    w = weights[global_idx]
                    Nij_w = Nij * w

                    x_pt += Nij_w * control_points[global_idx]
                    u_val += Nij_w * u[global_idx]
                    W += Nij_w

            x_pt /= W
            u_val /= W

            X[i, j] = x_pt[0]
            Y[i, j] = x_pt[1]
            U[i, j] = u_val

    return X, Y, U


def evaluate_solution_at_point(surface: NURBSSurface,
                                u: np.ndarray,
                                xi: Tuple[float, float]) -> Tuple[np.ndarray, float]:
    """
    Evaluate solution at a single parametric point.

    Parameters:
        surface: NURBS surface
        u: Solution vector
        xi: Parametric coordinates (xi, eta)

    Returns:
        (physical_point, solution_value)
    """
    xi_val, eta_val = xi
    kv_xi, kv_eta = surface.knot_vectors
    p_xi = kv_xi.degree
    p_eta = kv_eta.degree
    n_basis_eta = kv_eta.n_basis

    span_xi = kv_xi.find_span(xi_val)
    span_eta = kv_eta.find_span(eta_val)

    N_xi = eval_basis_ders_1d(kv_xi, xi_val, 0, span_xi)[0, :]
    N_eta = eval_basis_ders_1d(kv_eta, eta_val, 0, span_eta)[0, :]

    weights = surface.weights
    control_points = surface.control_points

    x_pt = np.zeros(surface.n_dim_physical)
    u_val = 0.0
    W = 0.0

    for ii in range(p_xi + 1):
        for jj in range(p_eta + 1):
            glob_i = span_xi - p_xi + ii
            glob_j = span_eta - p_eta + jj
            global_idx = glob_i * n_basis_eta + glob_j

            Nij = N_xi[ii] * N_eta[jj]
            w = weights[global_idx]
            Nij_w = Nij * w

            x_pt += Nij_w * control_points[global_idx]
            u_val += Nij_w * u[global_idx]
            W += Nij_w

    x_pt /= W
    u_val /= W

    return x_pt, u_val


def sample_solution_gradient_2d(surface: NURBSSurface,
                                 u: np.ndarray,
                                 n_xi: int = 50,
                                 n_eta: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample solution and its gradient on a uniform grid.

    Parameters:
        surface: NURBS surface geometry
        u: Solution vector
        n_xi, n_eta: Number of sample points

    Returns:
        (X, Y, U, dU_dx, dU_dy) arrays of shape (n_xi, n_eta)
    """
    kv_xi, kv_eta = surface.knot_vectors
    domain_xi = kv_xi.domain
    domain_eta = kv_eta.domain

    xi_vals = np.linspace(domain_xi[0], domain_xi[1], n_xi)
    eta_vals = np.linspace(domain_eta[0], domain_eta[1], n_eta)

    X = np.zeros((n_xi, n_eta))
    Y = np.zeros((n_xi, n_eta))
    U = np.zeros((n_xi, n_eta))
    dU_dx = np.zeros((n_xi, n_eta))
    dU_dy = np.zeros((n_xi, n_eta))

    p_xi = kv_xi.degree
    p_eta = kv_eta.degree
    n_basis_eta = kv_eta.n_basis

    weights = surface.weights
    control_points = surface.control_points

    for i, xi in enumerate(xi_vals):
        span_xi = kv_xi.find_span(xi)
        Nders_xi = eval_basis_ders_1d(kv_xi, xi, 1, span_xi)
        N_xi = Nders_xi[0, :]
        dN_xi = Nders_xi[1, :]

        for j, eta in enumerate(eta_vals):
            span_eta = kv_eta.find_span(eta)
            Nders_eta = eval_basis_ders_1d(kv_eta, eta, 1, span_eta)
            N_eta = Nders_eta[0, :]
            dN_eta = Nders_eta[1, :]

            # Compute weighted sums
            A_geo = np.zeros(surface.n_dim_physical)  # Weighted geometry sum
            dA_geo_dxi = np.zeros(surface.n_dim_physical)
            dA_geo_deta = np.zeros(surface.n_dim_physical)
            A_u = 0.0  # Weighted solution sum
            dA_u_dxi = 0.0
            dA_u_deta = 0.0
            W = 0.0
            dW_dxi = 0.0
            dW_deta = 0.0

            for ii in range(p_xi + 1):
                for jj in range(p_eta + 1):
                    glob_i = span_xi - p_xi + ii
                    glob_j = span_eta - p_eta + jj
                    global_idx = glob_i * n_basis_eta + glob_j

                    P = control_points[global_idx]
                    w = weights[global_idx]
                    u_cp = u[global_idx]
                    wP = w * P
                    wu = w * u_cp

                    Nij = N_xi[ii] * N_eta[jj]
                    dNij_dxi = dN_xi[ii] * N_eta[jj]
                    dNij_deta = N_xi[ii] * dN_eta[jj]

                    A_geo += Nij * wP
                    dA_geo_dxi += dNij_dxi * wP
                    dA_geo_deta += dNij_deta * wP

                    A_u += Nij * wu
                    dA_u_dxi += dNij_dxi * wu
                    dA_u_deta += dNij_deta * wu

                    W += Nij * w
                    dW_dxi += dNij_dxi * w
                    dW_deta += dNij_deta * w

            # Rational values (quotient rule)
            x_pt = A_geo / W
            u_val = A_u / W

            dx_dxi = (dA_geo_dxi * W - A_geo * dW_dxi) / (W * W)
            dx_deta = (dA_geo_deta * W - A_geo * dW_deta) / (W * W)

            du_dxi = (dA_u_dxi * W - A_u * dW_dxi) / (W * W)
            du_deta = (dA_u_deta * W - A_u * dW_deta) / (W * W)

            # Jacobian
            jacobian = np.zeros((2, 2))
            jacobian[0, :] = dx_dxi[:2]  # dx/dxi, dy/dxi
            jacobian[1, :] = dx_deta[:2]  # dx/deta, dy/deta
            jacobian = jacobian.T  # Now [dx/dxi, dx/deta; dy/dxi, dy/deta]

            # Inverse Jacobian
            det_jac = jacobian[0, 0] * jacobian[1, 1] - jacobian[0, 1] * jacobian[1, 0]
            inv_jac = np.array([
                [ jacobian[1, 1], -jacobian[0, 1]],
                [-jacobian[1, 0],  jacobian[0, 0]]
            ]) / det_jac

            # Physical gradient
            du_dx = du_dxi * inv_jac[0, 0] + du_deta * inv_jac[1, 0]
            du_dy = du_dxi * inv_jac[0, 1] + du_deta * inv_jac[1, 1]

            X[i, j] = x_pt[0]
            Y[i, j] = x_pt[1]
            U[i, j] = u_val
            dU_dx[i, j] = du_dx
            dU_dy[i, j] = du_dy

    return X, Y, U, dU_dx, dU_dy


def compute_l2_error(surface: NURBSSurface,
                     u_numerical: np.ndarray,
                     u_exact: callable,
                     n_sample: int = 100) -> float:
    """
    Compute approximate L2 error using grid sampling.

    This is a quick approximation - for accurate error, use
    Gauss quadrature integration.

    Parameters:
        surface: NURBS surface
        u_numerical: Numerical solution
        u_exact: Exact solution function f(x, y)
        n_sample: Number of sample points per direction

    Returns:
        Approximate L2 error
    """
    X, Y, U = sample_solution_2d(surface, u_numerical, n_sample, n_sample)

    U_exact = np.zeros_like(U)
    for i in range(n_sample):
        for j in range(n_sample):
            U_exact[i, j] = u_exact(X[i, j], Y[i, j])

    # Approximate integral using trapezoidal rule
    domain_xi = surface.knot_vectors[0].domain
    domain_eta = surface.knot_vectors[1].domain
    dx = (domain_xi[1] - domain_xi[0]) / (n_sample - 1)
    dy = (domain_eta[1] - domain_eta[0]) / (n_sample - 1)

    error_sq = np.sum((U - U_exact)**2) * dx * dy

    return np.sqrt(error_sq)

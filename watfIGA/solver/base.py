"""
Base solver class for IGA.

This module defines the abstract interface for IGA solvers and provides
common utilities for element-by-element assembly.

Design principles:
1. Solver operates element-by-element, receiving Element objects
2. Solver evaluates Bernstein basis (not spline basis directly)
3. Solver applies extraction operators to map to global DOFs
4. Solver knows NOTHING about knot vectors or spline types

The assembly loop is:
    for element in mesh.elements:
        # 1. Get quadrature points on reference element [0,1]^d
        # 2. Evaluate Bernstein basis at quadrature points
        # 3. Apply extraction: N_spline = C_e @ B_bernstein
        # 4. Compute element matrix/vector (with Jacobians)
        # 5. Apply extraction to element matrix: K_e = C_e.T @ K_bern @ C_e
        # 6. Scatter to global matrix using element.global_dof_indices

Key point: The solver never needs to change when switching
           NURBS -> THB-splines -> T-splines

TODO: Add iterative solver options
TODO: Add multi-field (block) assembly support
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from typing import Tuple, List, Optional, Callable
from abc import ABC, abstractmethod

from ..discretization.mesh import Mesh, MeshWithGeometry, DirichletBC
from ..discretization.element import Element, ElementWithGeometry
from ..discretization.extraction import BernsteinBasis
from ..quadrature.gauss import GaussQuadrature


class Solver(ABC):
    """
    Abstract base class for IGA solvers.

    Subclasses implement specific PDEs by overriding:
    - compute_element_matrices: Builds element stiffness and load
    - (optionally) additional problem-specific methods
    """

    def __init__(self, mesh: Mesh):
        """
        Initialize solver with mesh.

        Parameters:
            mesh: Mesh containing discretization (supports active element/CP tracking)
        """
        self.mesh = mesh
        self.n_dof = mesh.n_dof

        # Storage for assembled system
        self.K = None  # Global stiffness matrix
        self.f = None  # Global load vector
        self.u = None  # Solution vector

        # Boundary conditions
        self._dirichlet_bcs: List[DirichletBC] = []

    def add_dirichlet_bc(self, bc: DirichletBC):
        """Add a Dirichlet boundary condition."""
        self._dirichlet_bcs.append(bc)

    @abstractmethod
    def compute_element_matrices(self, element: Element,
                                  quadrature: GaussQuadrature) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute element stiffness matrix and load vector.

        This is the main method that subclasses must implement.
        It should return matrices in the Bernstein basis (before extraction).

        Parameters:
            element: Element with geometry information
            quadrature: Quadrature rule for integration

        Returns:
            (K_e, f_e) where:
            - K_e: Element stiffness matrix in Bernstein basis, shape (n_local, n_local)
            - f_e: Element load vector in Bernstein basis, shape (n_local,)

        Note:
            The extraction operator will be applied after this method returns.
        """
        pass

    def assemble(self, n_gauss_per_dir: Optional[Tuple[int, ...]] = None):
        """
        Assemble the global stiffness matrix and load vector.

        Parameters:
            n_gauss_per_dir: Number of Gauss points per direction.
                            Defaults to (p+1) in each direction.
        """
        # Determine quadrature
        if n_gauss_per_dir is None:
            # Use p+1 points per direction by default
            # Get degrees from first active element
            first_elem = next(self.mesh.get_active_elements())
            degrees = first_elem.degrees
            n_gauss_per_dir = tuple(p + 1 for p in degrees)

        quadrature = GaussQuadrature(n_gauss_per_dir)

        # Initialize global system using sparse triplet format
        row_indices = []
        col_indices = []
        values = []
        self.f = np.zeros(self.n_dof)

        # Loop over active elements only
        for element in self.mesh.get_active_elements():
            # Compute element matrices in Bernstein basis
            K_bern, f_bern = self.compute_element_matrices(element, quadrature)

            # Apply extraction operator to convert Bernstein to spline basis
            # Since N = C_e @ B (N_i = sum_j C[i,j] B_j), we have:
            #   K_spline[i,k] = ∫ ∇N_i · ∇N_k = C[i,j] K_bern[j,l] C[k,l] = (C @ K_bern @ C^T)[i,k]
            #   f_spline[i] = ∫ f N_i = C[i,j] f_bern[j] = (C @ f_bern)[i]
            C_e = element.extraction_operator
            K_e = C_e @ K_bern @ C_e.T
            f_e = C_e @ f_bern

            # Scatter to global system
            global_dofs = element.global_dof_indices

            for i_local, i_global in enumerate(global_dofs):
                self.f[i_global] += f_e[i_local]
                for j_local, j_global in enumerate(global_dofs):
                    row_indices.append(i_global)
                    col_indices.append(j_global)
                    values.append(K_e[i_local, j_local])

        # Build sparse matrix
        self.K = sparse.csr_matrix(
            (values, (row_indices, col_indices)),
            shape=(self.n_dof, self.n_dof)
        )

    def apply_boundary_conditions(self, method: str = "elimination"):
        """
        Apply Dirichlet boundary conditions.

        Parameters:
            method: "elimination" for row/column zeroing (recommended)
                   "penalty" for large penalty method
        """
        if method == "elimination":
            self._apply_bc_elimination()
        elif method == "penalty":
            self._apply_bc_penalty()
        else:
            raise ValueError(f"Unknown BC method: {method}")

    def _apply_bc_elimination(self):
        """Apply Dirichlet BCs by modifying rows and columns."""
        # Convert to lil_matrix for efficient modification
        K_lil = self.K.tolil()

        for bc in self._dirichlet_bcs:
            for idx, (dof, value) in enumerate(zip(bc.dof_indices, bc.values)):
                # Zero out row and column
                K_lil[dof, :] = 0
                K_lil[:, dof] = 0
                # Set diagonal to 1
                K_lil[dof, dof] = 1.0
                # Set RHS to prescribed value
                self.f[dof] = value

        self.K = K_lil.tocsr()

    def _apply_bc_penalty(self, penalty: float = 1e10):
        """Apply Dirichlet BCs using penalty method."""
        K_lil = self.K.tolil()

        for bc in self._dirichlet_bcs:
            for dof, value in zip(bc.dof_indices, bc.values):
                K_lil[dof, dof] += penalty
                self.f[dof] += penalty * value

        self.K = K_lil.tocsr()

    def solve(self) -> np.ndarray:
        """
        Solve the linear system.

        Returns:
            Solution vector u
        """
        if self.K is None or self.f is None:
            raise RuntimeError("System not assembled. Call assemble() first.")

        self.u = spsolve(self.K, self.f)
        return self.u

    def run(self, n_gauss_per_dir: Optional[Tuple[int, ...]] = None,
            bc_method: str = "elimination") -> np.ndarray:
        """
        Convenience method to assemble, apply BCs, and solve.

        Parameters:
            n_gauss_per_dir: Quadrature points per direction
            bc_method: Method for applying Dirichlet BCs

        Returns:
            Solution vector
        """
        self.assemble(n_gauss_per_dir)
        self.apply_boundary_conditions(bc_method)
        return self.solve()


def eval_geometry_at_quadrature_point(element: Element,
                                       B_vals: np.ndarray,
                                       dB_dxi: np.ndarray,
                                       dB_deta: np.ndarray) -> Tuple[
                                           np.ndarray, np.ndarray, float]:
    """
    Evaluate geometry and Jacobian at a quadrature point.

    Given Bernstein basis values and derivatives, compute:
    1. Physical coordinates (x, y)
    2. Jacobian matrix J = [dx/dxi, dx/deta; dy/dxi, dy/deta]
    3. Determinant |J|

    For NURBS (rational), applies the quotient rule for derivatives.

    IMPORTANT: This function first applies the Bézier extraction operator
    to convert Bernstein basis to spline basis, then computes the rational
    (NURBS) basis from the spline basis.

    Parameters:
        element: Element with control points and weights
        B_vals: Bernstein basis values at the quadrature point
        dB_dxi: Bernstein basis xi-derivatives (w.r.t. parametric coords)
        dB_deta: Bernstein basis eta-derivatives (w.r.t. parametric coords)

    Returns:
        (point, jacobian, det_jac) where:
        - point: Physical coordinates (n_dim_physical,)
        - jacobian: Jacobian matrix (n_dim_physical, 2)
        - det_jac: Determinant of Jacobian (for integration weight)
    """
    P = element.control_points  # (n_local, n_dim_physical)
    w = element.weights  # (n_local,)
    C_e = element.extraction_operator  # (n_local, n_bernstein)

    # Apply extraction operator: N = C_e @ B (spline basis from Bernstein)
    N_vals = C_e @ B_vals
    dN_dxi = C_e @ dB_dxi
    dN_deta = C_e @ dB_deta

    # Weighted spline basis values for NURBS
    Nw = N_vals * w
    W = np.sum(Nw)  # Denominator of NURBS

    # Weighted spline basis derivatives
    dNw_dxi = dN_dxi * w
    dNw_deta = dN_deta * w
    dW_dxi = np.sum(dNw_dxi)
    dW_deta = np.sum(dNw_deta)

    # Rational basis: R = Nw / W
    R = Nw / W

    # Rational basis derivatives (quotient rule):
    # dR/dxi = (dNw/dxi * W - Nw * dW/dxi) / W^2
    dR_dxi = (dNw_dxi * W - Nw * dW_dxi) / (W * W)
    dR_deta = (dNw_deta * W - Nw * dW_deta) / (W * W)

    # Physical point: x = sum_i R_i * P_i
    point = R @ P  # (n_dim_physical,)

    # Jacobian: dx/dxi = sum_i dR_i/dxi * P_i
    n_dim_physical = P.shape[1]
    jacobian = np.zeros((n_dim_physical, 2))
    jacobian[:, 0] = dR_dxi @ P  # dx/dxi, dy/dxi
    jacobian[:, 1] = dR_deta @ P  # dx/deta, dy/deta

    # Determinant (for 2D parametric, 2D/3D physical)
    if n_dim_physical == 2:
        det_jac = jacobian[0, 0] * jacobian[1, 1] - jacobian[0, 1] * jacobian[1, 0]
    elif n_dim_physical == 3:
        # For surface in 3D, compute |J| from cross product of tangent vectors
        tau_xi = jacobian[:, 0]
        tau_eta = jacobian[:, 1]
        normal = np.cross(tau_xi, tau_eta)
        det_jac = np.linalg.norm(normal)
    else:
        det_jac = abs(np.linalg.det(jacobian))

    return point, jacobian, det_jac


def compute_shape_function_derivatives(element: Element,
                                        B_vals: np.ndarray,
                                        dB_dxi: np.ndarray,
                                        dB_deta: np.ndarray,
                                        jacobian: np.ndarray) -> Tuple[
                                            np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute shape functions and their physical derivatives.

    Transforms parametric derivatives to physical derivatives using:
    [dN/dx]   [dxi/dx  deta/dx] [dN/dxi ]
    [dN/dy] = [dxi/dy  deta/dy] [dN/deta]

    IMPORTANT: This function first applies the Bézier extraction operator
    to convert Bernstein basis to spline basis, then computes the rational
    (NURBS) basis and its physical derivatives.

    Parameters:
        element: Element with weights and extraction operator
        B_vals, dB_dxi, dB_deta: Bernstein basis and derivatives (w.r.t. parametric)
        jacobian: Jacobian matrix from eval_geometry_at_quadrature_point

    Returns:
        (R, dR_dx, dR_dy) where:
        - R: Rational basis values (n_local,)
        - dR_dx: x-derivatives of rational basis (n_local,)
        - dR_dy: y-derivatives of rational basis (n_local,)
    """
    w = element.weights
    C_e = element.extraction_operator

    # Apply extraction operator: N = C_e @ B (spline basis from Bernstein)
    N_vals = C_e @ B_vals
    dN_dxi = C_e @ dB_dxi
    dN_deta = C_e @ dB_deta

    # Compute rational basis and parametric derivatives
    Nw = N_vals * w
    W = np.sum(Nw)
    R = Nw / W

    dNw_dxi = dN_dxi * w
    dNw_deta = dN_deta * w
    dW_dxi = np.sum(dNw_dxi)
    dW_deta = np.sum(dNw_deta)

    dR_dxi = (dNw_dxi * W - Nw * dW_dxi) / (W * W)
    dR_deta = (dNw_deta * W - Nw * dW_deta) / (W * W)

    # Inverse Jacobian for mapping derivatives
    # [dxi/dx  dxi/dy ]   1   [ dy/deta  -dx/deta]
    # [deta/dx deta/dy] = --- [-dy/dxi   dx/dxi  ]
    #                     |J|

    det_jac = jacobian[0, 0] * jacobian[1, 1] - jacobian[0, 1] * jacobian[1, 0]

    inv_jac = np.array([
        [ jacobian[1, 1], -jacobian[0, 1]],
        [-jacobian[1, 0],  jacobian[0, 0]]
    ]) / det_jac

    # dR/dx = dR/dxi * dxi/dx + dR/deta * deta/dx
    dR_dx = dR_dxi * inv_jac[0, 0] + dR_deta * inv_jac[1, 0]
    dR_dy = dR_dxi * inv_jac[0, 1] + dR_deta * inv_jac[1, 1]

    return R, dR_dx, dR_dy

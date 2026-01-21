"""
NURBS (Non-Uniform Rational B-Spline) geometry representation.

NURBS extend B-splines by introducing weights for each control point,
enabling exact representation of conic sections (circles, ellipses, etc.).

A NURBS curve/surface point is computed as:

    C(xi) = sum_i (N_i(xi) * w_i * P_i) / sum_i (N_i(xi) * w_i)

where:
- N_i are B-spline basis functions
- w_i are weights (positive real numbers)
- P_i are control points

The rational basis functions R_i(xi) = N_i(xi) * w_i / sum_j N_j(xi) * w_j
form a partition of unity and are non-negative.

This module provides:
- NURBSGeometry: Abstract base for NURBS geometries
- NURBSCurve: 1D NURBS parametric curves
- NURBSSurface: 2D NURBS parametric surfaces
- Factory functions for common geometries

TODO: Extend for THB-NURBS (hierarchical)
TODO: Extend for T-NURBS (local topology)
"""

import math
import numpy as np
from typing import Tuple, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..discretization.knot_vector import KnotVector, make_open_knot_vector
from .bspline import BSplineBasis, TensorProductBasis, eval_basis_ders_1d


class NURBSGeometry(ABC):
    """
    Abstract base class for NURBS geometry objects.

    This defines the interface that all NURBS geometries must implement.
    The solver interacts with geometry through this interface.

    Key responsibilities:
    - Store control points and weights
    - Evaluate geometry at parameter values
    - Provide Jacobian information for mapping

    TODO: THB and T-spline geometries will inherit from this
    """

    @property
    @abstractmethod
    def n_dim_parametric(self) -> int:
        """Number of parametric dimensions (1=curve, 2=surface, 3=volume)."""
        pass

    @property
    @abstractmethod
    def n_dim_physical(self) -> int:
        """Number of physical/spatial dimensions."""
        pass

    @property
    @abstractmethod
    def n_control_points(self) -> int:
        """Total number of control points (= number of DOFs per field)."""
        pass

    @property
    @abstractmethod
    def control_points(self) -> np.ndarray:
        """Control point coordinates as (n_control_points, n_dim_physical) array."""
        pass

    @property
    @abstractmethod
    def weights(self) -> np.ndarray:
        """NURBS weights as (n_control_points,) array."""
        pass

    @abstractmethod
    def eval_point(self, xi: Union[float, Tuple[float, ...]]) -> np.ndarray:
        """Evaluate geometry at a parameter value."""
        pass

    @abstractmethod
    def eval_derivatives(self, xi: Union[float, Tuple[float, ...]],
                         n_ders: int = 1) -> Tuple[np.ndarray, ...]:
        """Evaluate geometry and derivatives at a parameter value."""
        pass


class NURBSCurve(NURBSGeometry):
    """
    NURBS curve in arbitrary dimensional space.

    A NURBS curve C(xi) is defined by:
    - Knot vector defining the parametric domain
    - Control points P_i in R^d (d = physical dimension)
    - Weights w_i > 0
    """

    def __init__(self, knot_vector: KnotVector,
                 control_points: np.ndarray,
                 weights: Optional[np.ndarray] = None):
        """
        Initialize a NURBS curve.

        Parameters:
            knot_vector: KnotVector defining the basis
            control_points: Array of shape (n, d) where n = n_basis functions
            weights: Array of shape (n,), defaults to 1.0 (B-spline)
        """
        self._knot_vector = knot_vector
        self._basis = BSplineBasis(knot_vector)
        self._control_points = np.atleast_2d(control_points).astype(np.float64)

        if self._control_points.shape[0] != knot_vector.n_basis:
            raise ValueError(
                f"Number of control points ({self._control_points.shape[0]}) "
                f"must match number of basis functions ({knot_vector.n_basis})"
            )

        if weights is None:
            self._weights = np.ones(knot_vector.n_basis)
        else:
            self._weights = np.asarray(weights, dtype=np.float64)
            if len(self._weights) != knot_vector.n_basis:
                raise ValueError("Weights array length must match number of control points")
            if np.any(self._weights <= 0):
                raise ValueError("All weights must be positive")

    @property
    def n_dim_parametric(self) -> int:
        return 1

    @property
    def n_dim_physical(self) -> int:
        return self._control_points.shape[1]

    @property
    def n_control_points(self) -> int:
        return self._knot_vector.n_basis

    @property
    def control_points(self) -> np.ndarray:
        return self._control_points.copy()

    @property
    def weights(self) -> np.ndarray:
        return self._weights.copy()

    @property
    def knot_vector(self) -> KnotVector:
        return self._knot_vector

    @property
    def basis(self) -> BSplineBasis:
        return self._basis

    @property
    def degree(self) -> int:
        return self._knot_vector.degree

    def eval_point(self, xi: float) -> np.ndarray:
        """
        Evaluate curve at parameter value.

        Parameters:
            xi: Parameter value

        Returns:
            Point coordinates as (d,) array
        """
        span = self._knot_vector.find_span(xi)
        N = self._basis.eval(xi, span)

        # Get active control points and weights
        start = span - self.degree
        P_local = self._control_points[start:start + self.degree + 1]
        w_local = self._weights[start:start + self.degree + 1]

        # Compute weighted sum
        Nw = N * w_local
        W = np.sum(Nw)  # Denominator
        point = np.dot(Nw, P_local) / W

        return point

    def eval_derivatives(self, xi: float, n_ders: int = 1) -> Tuple[np.ndarray, ...]:
        """
        Evaluate curve and derivatives at parameter value.

        Uses the formula for rational derivatives (Piegl & Tiller, Eq. 4.8).

        Parameters:
            xi: Parameter value
            n_ders: Number of derivatives

        Returns:
            Tuple (C, dC/dxi, d²C/dxi², ...) of arrays
        """
        span = self._knot_vector.find_span(xi)
        Nders = self._basis.eval_ders(xi, n_ders, span)

        start = span - self.degree
        P_local = self._control_points[start:start + self.degree + 1]
        w_local = self._weights[start:start + self.degree + 1]

        # Compute A^(k) = sum_i N_i^(k) * w_i * P_i  (numerator derivatives)
        # and w^(k) = sum_i N_i^(k) * w_i  (denominator derivatives)
        A_ders = np.zeros((n_ders + 1, self.n_dim_physical))
        w_ders = np.zeros(n_ders + 1)

        for k in range(n_ders + 1):
            Nk = Nders[k, :]
            Nkw = Nk * w_local
            A_ders[k] = np.dot(Nkw, P_local)
            w_ders[k] = np.sum(Nkw)

        # Compute rational derivatives using the quotient rule
        # C^(k) = (A^(k) - sum_{j=1}^{k} C(k,j) * w^(j) * C^(k-j)) / w^(0)
        C_ders = np.zeros((n_ders + 1, self.n_dim_physical))

        for k in range(n_ders + 1):
            v = A_ders[k].copy()
            for j in range(1, k + 1):
                binom = math.comb(k, j)
                v -= binom * w_ders[j] * C_ders[k - j]
            C_ders[k] = v / w_ders[0]

        return tuple(C_ders[k] for k in range(n_ders + 1))


class NURBSSurface(NURBSGeometry):
    """
    NURBS surface in 3D (or 2D) space.

    A NURBS surface S(xi, eta) is defined by:
    - Two knot vectors (xi and eta directions)
    - Control points P_{i,j} arranged in a grid
    - Weights w_{i,j} > 0

    The surface point is:
    S(xi, eta) = sum_{i,j} R_{i,j}(xi, eta) * P_{i,j}

    where R_{i,j} are the rational basis functions.

    Control points are stored in row-major order:
    [P_{0,0}, P_{0,1}, ..., P_{0,m}, P_{1,0}, ..., P_{n,m}]
    """

    def __init__(self,
                 knot_vector_xi: KnotVector,
                 knot_vector_eta: KnotVector,
                 control_points: np.ndarray,
                 weights: Optional[np.ndarray] = None):
        """
        Initialize a NURBS surface.

        Parameters:
            knot_vector_xi: KnotVector for xi direction
            knot_vector_eta: KnotVector for eta direction
            control_points: Array of shape (n_xi * n_eta, d) in row-major order
                           or (n_xi, n_eta, d) which will be reshaped
            weights: Array of shape (n_xi * n_eta,) or (n_xi, n_eta), defaults to 1.0
        """
        self._kv_xi = knot_vector_xi
        self._kv_eta = knot_vector_eta
        self._basis_xi = BSplineBasis(knot_vector_xi)
        self._basis_eta = BSplineBasis(knot_vector_eta)

        n_xi = knot_vector_xi.n_basis
        n_eta = knot_vector_eta.n_basis
        n_total = n_xi * n_eta

        # Handle control points shape
        control_points = np.asarray(control_points, dtype=np.float64)
        if control_points.ndim == 3:
            if control_points.shape[:2] != (n_xi, n_eta):
                raise ValueError(
                    f"Control points shape {control_points.shape} doesn't match "
                    f"expected ({n_xi}, {n_eta}, d)"
                )
            self._control_points = control_points.reshape(n_total, -1)
        else:
            if control_points.shape[0] != n_total:
                raise ValueError(
                    f"Number of control points ({control_points.shape[0]}) "
                    f"must equal n_xi * n_eta ({n_total})"
                )
            self._control_points = control_points

        # Handle weights
        if weights is None:
            self._weights = np.ones(n_total)
        else:
            weights = np.asarray(weights, dtype=np.float64).flatten()
            if len(weights) != n_total:
                raise ValueError(f"Weights length ({len(weights)}) must equal {n_total}")
            if np.any(weights <= 0):
                raise ValueError("All weights must be positive")
            self._weights = weights

        # Store dimensions
        self._n_xi = n_xi
        self._n_eta = n_eta

    @property
    def n_dim_parametric(self) -> int:
        return 2

    @property
    def n_dim_physical(self) -> int:
        return self._control_points.shape[1]

    @property
    def n_control_points(self) -> int:
        return self._n_xi * self._n_eta

    @property
    def n_control_points_per_dir(self) -> Tuple[int, int]:
        """Number of control points in each direction (n_xi, n_eta)."""
        return (self._n_xi, self._n_eta)

    @property
    def control_points(self) -> np.ndarray:
        return self._control_points.copy()

    @property
    def control_points_grid(self) -> np.ndarray:
        """Control points as (n_xi, n_eta, d) grid."""
        return self._control_points.reshape(self._n_xi, self._n_eta, -1)

    @property
    def weights(self) -> np.ndarray:
        return self._weights.copy()

    @property
    def weights_grid(self) -> np.ndarray:
        """Weights as (n_xi, n_eta) grid."""
        return self._weights.reshape(self._n_xi, self._n_eta)

    @property
    def knot_vectors(self) -> Tuple[KnotVector, KnotVector]:
        return (self._kv_xi, self._kv_eta)

    @property
    def degrees(self) -> Tuple[int, int]:
        return (self._kv_xi.degree, self._kv_eta.degree)

    @property
    def domain(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Parametric domain as ((xi_min, xi_max), (eta_min, eta_max))."""
        return (self._kv_xi.domain, self._kv_eta.domain)

    def _local_to_global_index(self, i: int, j: int) -> int:
        """
        Convert (i, j) tensor index to global flat index.

        Ordering: x (xi) varies fastest, y (eta) varies slowest.
        For a 4x3 grid (n_xi=4, n_eta=3):
          j=0: [0,1,2,3], j=1: [4,5,6,7], j=2: [8,9,10,11]
        """
        return j * self._n_xi + i

    def _global_to_local_index(self, idx: int) -> Tuple[int, int]:
        """Convert global flat index to (i, j) tensor index."""
        return (idx % self._n_xi, idx // self._n_xi)

    def eval_point(self, xi: Tuple[float, float]) -> np.ndarray:
        """
        Evaluate surface at parameter values.

        Parameters:
            xi: Parameter values (xi, eta)

        Returns:
            Point coordinates as (d,) array
        """
        xi_val, eta_val = xi

        # Find spans
        span_xi = self._kv_xi.find_span(xi_val)
        span_eta = self._kv_eta.find_span(eta_val)

        # Evaluate 1D basis functions
        N_xi = self._basis_xi.eval(xi_val, span_xi)
        N_eta = self._basis_eta.eval(eta_val, span_eta)

        p_xi = self._kv_xi.degree
        p_eta = self._kv_eta.degree

        point = np.zeros(self.n_dim_physical)
        W = 0.0

        for j in range(p_eta + 1):
            for i in range(p_xi + 1):
                glob_i = span_xi - p_xi + i
                glob_j = span_eta - p_eta + j
                idx = self._local_to_global_index(glob_i, glob_j)

                Nij = N_xi[i] * N_eta[j]
                w = self._weights[idx]
                Nij_w = Nij * w

                point += Nij_w * self._control_points[idx]
                W += Nij_w

        return point / W

    def eval_derivatives(self, xi: Tuple[float, float],
                         n_ders: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate surface point and first derivatives.

        Parameters:
            xi: Parameter values (xi, eta)
            n_ders: Number of derivatives (currently only 1 is fully implemented)

        Returns:
            Tuple (S, dS/dxi, dS/deta) of (d,) arrays
        """
        xi_val, eta_val = xi

        # Find spans
        span_xi = self._kv_xi.find_span(xi_val)
        span_eta = self._kv_eta.find_span(eta_val)

        # Evaluate 1D basis functions and derivatives
        Nders_xi = eval_basis_ders_1d(self._kv_xi, xi_val, 1, span_xi)
        Nders_eta = eval_basis_ders_1d(self._kv_eta, eta_val, 1, span_eta)

        N_xi = Nders_xi[0, :]
        dN_xi = Nders_xi[1, :]
        N_eta = Nders_eta[0, :]
        dN_eta = Nders_eta[1, :]

        p_xi = self._kv_xi.degree
        p_eta = self._kv_eta.degree

        A = np.zeros(self.n_dim_physical)
        dA_dxi = np.zeros(self.n_dim_physical)
        dA_deta = np.zeros(self.n_dim_physical)
        W = 0.0
        dW_dxi = 0.0
        dW_deta = 0.0

        for j in range(p_eta + 1):
            for i in range(p_xi + 1):
                glob_i = span_xi - p_xi + i
                glob_j = span_eta - p_eta + j
                idx = self._local_to_global_index(glob_i, glob_j)

                P = self._control_points[idx]
                w = self._weights[idx]
                wP = w * P

                Nij = N_xi[i] * N_eta[j]
                dNij_dxi = dN_xi[i] * N_eta[j]
                dNij_deta = N_xi[i] * dN_eta[j]

                A += Nij * wP
                dA_dxi += dNij_dxi * wP
                dA_deta += dNij_deta * wP

                W += Nij * w
                dW_dxi += dNij_dxi * w
                dW_deta += dNij_deta * w

        # Apply quotient rule: d/dxi(A/W) = (dA/dxi * W - A * dW/dxi) / W^2
        S = A / W
        dS_dxi = (dA_dxi * W - A * dW_dxi) / (W * W)
        dS_deta = (dA_deta * W - A * dW_deta) / (W * W)

        return (S, dS_dxi, dS_deta)

    def eval_rational_basis(self, xi: Tuple[float, float],
                            return_derivatives: bool = False) -> Union[
                                Tuple[np.ndarray, np.ndarray],
                                Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Evaluate NURBS rational basis functions at a parameter point.

        This returns the basis function values for all non-zero basis functions
        at the given parameter point, along with their global indices.

        Parameters:
            xi: Parameter values (xi, eta)
            return_derivatives: If True, also return derivatives

        Returns:
            If return_derivatives=False:
                (R, indices) where R is (n_local,) rational basis values
                and indices is (n_local,) global DOF indices
            If return_derivatives=True:
                (R, dR_dxi, dR_deta, indices)
        """
        xi_val, eta_val = xi

        # Find spans
        span_xi = self._kv_xi.find_span(xi_val)
        span_eta = self._kv_eta.find_span(eta_val)

        p_xi = self._kv_xi.degree
        p_eta = self._kv_eta.degree
        n_local = (p_xi + 1) * (p_eta + 1)

        if return_derivatives:
            Nders_xi = eval_basis_ders_1d(self._kv_xi, xi_val, 1, span_xi)
            Nders_eta = eval_basis_ders_1d(self._kv_eta, eta_val, 1, span_eta)
            N_xi = Nders_xi[0, :]
            dN_xi = Nders_xi[1, :]
            N_eta = Nders_eta[0, :]
            dN_eta = Nders_eta[1, :]
        else:
            N_xi = self._basis_xi.eval(xi_val, span_xi)
            N_eta = self._basis_eta.eval(eta_val, span_eta)
            dN_xi = dN_eta = None

        Nw = np.zeros(n_local)
        indices = np.zeros(n_local, dtype=int)

        if return_derivatives:
            dNw_dxi = np.zeros(n_local)
            dNw_deta = np.zeros(n_local)

        W = 0.0
        dW_dxi = 0.0
        dW_deta = 0.0

        local_idx = 0
        for j in range(p_eta + 1):
            for i in range(p_xi + 1):
                glob_i = span_xi - p_xi + i
                glob_j = span_eta - p_eta + j
                global_idx = self._local_to_global_index(glob_i, glob_j)

                w = self._weights[global_idx]
                Nij = N_xi[i] * N_eta[j]
                Nij_w = Nij * w

                Nw[local_idx] = Nij_w
                indices[local_idx] = global_idx
                W += Nij_w

                if return_derivatives:
                    dNij_dxi = dN_xi[i] * N_eta[j]
                    dNij_deta = N_xi[i] * dN_eta[j]
                    dNw_dxi[local_idx] = dNij_dxi * w
                    dNw_deta[local_idx] = dNij_deta * w
                    dW_dxi += dNij_dxi * w
                    dW_deta += dNij_deta * w

                local_idx += 1

        # Compute rational basis: R = Nw / W
        R = Nw / W

        if return_derivatives:
            # dR/dxi = (dNw/dxi * W - Nw * dW/dxi) / W^2
            dR_dxi = (dNw_dxi * W - Nw * dW_dxi) / (W * W)
            dR_deta = (dNw_deta * W - Nw * dW_deta) / (W * W)
            return (R, dR_dxi, dR_deta, indices)

        return (R, indices)


# For backward compatibility, import factory functions from primitives
from .primitives import (
    make_nurbs_unit_square,
    make_nurbs_rectangle,
    make_nurbs_circle,
    make_nurbs_arc,
    make_nurbs_disk,
    make_nurbs_annulus,
)

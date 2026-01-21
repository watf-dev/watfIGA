"""
Gauss-Legendre quadrature for numerical integration.

Gauss quadrature provides optimal polynomial integration:
n points integrate exactly polynomials up to degree 2n-1.

For IGA with polynomial degree p, we typically need n_gauss = p+1
points per direction to integrate the mass matrix exactly, and
slightly more for accurate stiffness matrix integration with
rational functions (NURBS).

The reference domain is [0, 1] for consistency with Bernstein basis.
Standard Gauss points on [-1, 1] are mapped accordingly.

Usage:
    points, weights = gauss_legendre_1d(n)  # 1D quadrature on [0,1]
    points, weights = gauss_legendre_2d(n_xi, n_eta)  # 2D tensor-product
"""

import numpy as np
from typing import Tuple
from functools import lru_cache


@lru_cache(maxsize=16)
def gauss_legendre_1d(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gauss-Legendre quadrature points and weights on [0, 1].

    Parameters:
        n: Number of quadrature points

    Returns:
        (points, weights) where:
        - points: Array of n quadrature points in [0, 1]
        - weights: Array of n quadrature weights (sum to 1)

    Note:
        For polynomial degree p, use n >= (p+1) for mass matrix,
        and n >= (p+1) or (p+2) for stiffness matrix.
    """
    if n < 1:
        raise ValueError("Need at least 1 quadrature point")

    # Get standard Gauss points on [-1, 1]
    points_std, weights_std = np.polynomial.legendre.leggauss(n)

    # Map to [0, 1]: x = (xi + 1) / 2, dx = 1/2 * dxi
    points = 0.5 * (points_std + 1.0)
    weights = 0.5 * weights_std

    return points.copy(), weights.copy()


def gauss_legendre_2d(n_xi: int, n_eta: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tensor-product Gauss-Legendre quadrature on [0,1]².

    Parameters:
        n_xi: Number of points in xi direction
        n_eta: Number of points in eta direction

    Returns:
        (points, weights) where:
        - points: Array of shape (n_xi * n_eta, 2) with (xi, eta) coordinates
        - weights: Array of shape (n_xi * n_eta,) with weights
    """
    xi_pts, xi_wts = gauss_legendre_1d(n_xi)
    eta_pts, eta_wts = gauss_legendre_1d(n_eta)

    n_total = n_xi * n_eta
    points = np.zeros((n_total, 2))
    weights = np.zeros(n_total)

    idx = 0
    for i in range(n_xi):
        for j in range(n_eta):
            points[idx, 0] = xi_pts[i]
            points[idx, 1] = eta_pts[j]
            weights[idx] = xi_wts[i] * eta_wts[j]
            idx += 1

    return points, weights


def gauss_legendre_3d(n_xi: int, n_eta: int, n_zeta: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tensor-product Gauss-Legendre quadrature on [0,1]³.

    Parameters:
        n_xi, n_eta, n_zeta: Number of points in each direction

    Returns:
        (points, weights) where:
        - points: Array of shape (n_total, 3) with (xi, eta, zeta) coordinates
        - weights: Array of shape (n_total,) with weights
    """
    xi_pts, xi_wts = gauss_legendre_1d(n_xi)
    eta_pts, eta_wts = gauss_legendre_1d(n_eta)
    zeta_pts, zeta_wts = gauss_legendre_1d(n_zeta)

    n_total = n_xi * n_eta * n_zeta
    points = np.zeros((n_total, 3))
    weights = np.zeros(n_total)

    idx = 0
    for i in range(n_xi):
        for j in range(n_eta):
            for k in range(n_zeta):
                points[idx, 0] = xi_pts[i]
                points[idx, 1] = eta_pts[j]
                points[idx, 2] = zeta_pts[k]
                weights[idx] = xi_wts[i] * eta_wts[j] * zeta_wts[k]
                idx += 1

    return points, weights


class GaussQuadrature:
    """
    Encapsulates Gauss quadrature for element integration.

    This class provides a convenient interface for getting quadrature
    points and weights for elements of different dimensions.

    Attributes:
        n_points_per_dir: Number of quadrature points per parametric direction
    """

    def __init__(self, n_points_per_dir: Tuple[int, ...]):
        """
        Initialize Gauss quadrature.

        Parameters:
            n_points_per_dir: Number of points in each direction
        """
        self.n_points_per_dir = n_points_per_dir
        self.n_dim = len(n_points_per_dir)

        # Precompute points and weights
        if self.n_dim == 1:
            self._points, self._weights = gauss_legendre_1d(n_points_per_dir[0])
            self._points = self._points.reshape(-1, 1)
        elif self.n_dim == 2:
            self._points, self._weights = gauss_legendre_2d(
                n_points_per_dir[0], n_points_per_dir[1])
        elif self.n_dim == 3:
            self._points, self._weights = gauss_legendre_3d(
                n_points_per_dir[0], n_points_per_dir[1], n_points_per_dir[2])
        else:
            raise ValueError(f"Unsupported dimension: {self.n_dim}")

    @property
    def n_points(self) -> int:
        """Total number of quadrature points."""
        return len(self._weights)

    @property
    def points(self) -> np.ndarray:
        """
        Quadrature points on reference element [0,1]^d.

        Returns:
            Array of shape (n_points, n_dim)
        """
        return self._points

    @property
    def weights(self) -> np.ndarray:
        """
        Quadrature weights.

        Returns:
            Array of shape (n_points,)
        """
        return self._weights

    @classmethod
    def for_degree(cls, degrees: Tuple[int, ...],
                   rule: str = "full") -> 'GaussQuadrature':
        """
        Create quadrature rule appropriate for given polynomial degrees.

        Parameters:
            degrees: Polynomial degrees in each direction
            rule: "full" for (p+1) points, "reduced" for p points

        Returns:
            GaussQuadrature instance
        """
        if rule == "full":
            n_pts = tuple(p + 1 for p in degrees)
        elif rule == "reduced":
            n_pts = tuple(max(p, 1) for p in degrees)
        else:
            raise ValueError(f"Unknown rule: {rule}")

        return cls(n_pts)

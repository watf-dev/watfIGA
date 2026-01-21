"""
B-spline basis function evaluation.

B-splines are piecewise polynomial functions defined by:
1. A knot vector (non-decreasing sequence of parametric values)
2. A polynomial degree p

The i-th B-spline basis function of degree p is defined recursively:

    N_{i,0}(xi) = 1 if xi_i <= xi < xi_{i+1}, else 0

    N_{i,p}(xi) = (xi - xi_i)/(xi_{i+p} - xi_i) * N_{i,p-1}(xi)
                + (xi_{i+p+1} - xi)/(xi_{i+p+1} - xi_{i+1}) * N_{i+1,p-1}(xi)

Properties:
- Partition of unity: sum of all basis functions = 1
- Non-negativity: N_{i,p}(xi) >= 0
- Local support: N_{i,p} is non-zero only on [xi_i, xi_{i+p+1})
- Smoothness: C^{p-k} at a knot of multiplicity k

This module provides dimension-agnostic basis evaluation.
For tensor-product surfaces/volumes, use outer products of 1D evaluations.

TODO: THB-splines will override basis evaluation with hierarchical logic
TODO: T-splines will use local knot vectors per basis function
"""

import numpy as np
from typing import Tuple, Optional
from ..discretization.knot_vector import KnotVector


def eval_basis_1d(kv: KnotVector, xi: float,
                  span: Optional[int] = None) -> np.ndarray:
    """
    Evaluate all non-zero B-spline basis functions at a parameter value.

    Uses the Cox-de Boor algorithm optimized for evaluating only
    the p+1 non-zero basis functions at a given parameter value.

    Parameters:
        kv: Knot vector
        xi: Parameter value
        span: Optional pre-computed span index

    Returns:
        Array of shape (p+1,) containing N_{span-p,p}(xi) to N_{span,p}(xi)
    """
    p = kv.degree
    knots = kv.knots

    if span is None:
        span = kv.find_span(xi)

    # Initialize with degree 0
    N = np.zeros(p + 1)
    N[0] = 1.0

    # Build up to degree p
    left = np.zeros(p + 1)
    right = np.zeros(p + 1)

    for j in range(1, p + 1):
        left[j] = xi - knots[span + 1 - j]
        right[j] = knots[span + j] - xi

        saved = 0.0
        for r in range(j):
            temp = N[r] / (right[r + 1] + left[j - r])
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        N[j] = saved

    return N


def eval_basis_ders_1d(kv: KnotVector, xi: float, n_ders: int,
                       span: Optional[int] = None) -> np.ndarray:
    """
    Evaluate B-spline basis functions and derivatives at a parameter value.

    Uses the algorithm from Piegl & Tiller "The NURBS Book" (Algorithm A2.3).

    Parameters:
        kv: Knot vector
        xi: Parameter value
        n_ders: Number of derivatives to compute (0 = just values)
        span: Optional pre-computed span index

    Returns:
        Array of shape (n_ders+1, p+1) where result[k, j] is the k-th derivative
        of the j-th non-zero basis function (N_{span-p+j, p})
    """
    p = kv.degree
    knots = kv.knots

    if span is None:
        span = kv.find_span(xi)

    # Limit derivatives to degree
    n_ders = min(n_ders, p)

    # Result array
    ders = np.zeros((n_ders + 1, p + 1))

    # ndu[j][r] = N_{span-p+r, j} or knot differences
    ndu = np.zeros((p + 1, p + 1))
    ndu[0, 0] = 1.0

    left = np.zeros(p + 1)
    right = np.zeros(p + 1)

    # Compute basis functions and store knot differences
    for j in range(1, p + 1):
        left[j] = xi - knots[span + 1 - j]
        right[j] = knots[span + j] - xi

        saved = 0.0
        for r in range(j):
            # Upper triangle: knot differences
            ndu[j, r] = right[r + 1] + left[j - r]
            temp = ndu[r, j - 1] / ndu[j, r]

            # Lower triangle: basis functions
            ndu[r, j] = saved + right[r + 1] * temp
            saved = left[j - r] * temp

        ndu[j, j] = saved

    # Load basis functions
    for j in range(p + 1):
        ders[0, j] = ndu[j, p]

    # Compute derivatives
    a = np.zeros((2, p + 1))

    for r in range(p + 1):  # Loop over basis functions
        s1, s2 = 0, 1
        a[0, 0] = 1.0

        for k in range(1, n_ders + 1):  # Loop over derivatives
            d = 0.0
            rk = r - k
            pk = p - k

            if r >= k:
                a[s2, 0] = a[s1, 0] / ndu[pk + 1, rk]
                d = a[s2, 0] * ndu[rk, pk]

            j1 = 1 if rk >= -1 else -rk
            j2 = k - 1 if r - 1 <= pk else p - r

            for j in range(j1, j2 + 1):
                a[s2, j] = (a[s1, j] - a[s1, j - 1]) / ndu[pk + 1, rk + j]
                d += a[s2, j] * ndu[rk + j, pk]

            if r <= pk:
                a[s2, k] = -a[s1, k - 1] / ndu[pk + 1, r]
                d += a[s2, k] * ndu[r, pk]

            ders[k, r] = d
            s1, s2 = s2, s1

    # Multiply by factorial factors
    r = p
    for k in range(1, n_ders + 1):
        for j in range(p + 1):
            ders[k, j] *= r
        r *= (p - k)

    return ders


def eval_basis_nd(knot_vectors: Tuple[KnotVector, ...],
                  xi: Tuple[float, ...],
                  n_ders: int = 0) -> Tuple[np.ndarray, Tuple[np.ndarray, ...]]:
    """
    Evaluate tensor-product B-spline basis in n dimensions.

    For a tensor-product basis, the multivariate basis function is:
    N_{i,j,...}(xi, eta, ...) = N_i(xi) * N_j(eta) * ...

    Parameters:
        knot_vectors: Tuple of KnotVector objects, one per dimension
        xi: Tuple of parameter values, one per dimension
        n_ders: Number of derivatives to compute per direction

    Returns:
        (basis_values, derivatives) where:
        - basis_values: Flat array of all non-zero basis function values
        - derivatives: Tuple of derivative arrays, one per parametric direction
                      Each has shape (n_local_basis,) containing d N / d xi_k
    """
    n_dim = len(knot_vectors)

    # Evaluate 1D basis functions in each direction
    basis_1d = []
    ders_1d = []
    spans = []

    for d in range(n_dim):
        kv = knot_vectors[d]
        span = kv.find_span(xi[d])
        spans.append(span)

        if n_ders > 0:
            ders = eval_basis_ders_1d(kv, xi[d], n_ders, span)
            basis_1d.append(ders[0, :])
            ders_1d.append(ders)
        else:
            basis_1d.append(eval_basis_1d(kv, xi[d], span))
            ders_1d.append(None)

    # Compute tensor product of basis values
    # Start with first direction
    result = basis_1d[0].copy()

    # Tensor product with remaining directions
    for d in range(1, n_dim):
        result = np.outer(result, basis_1d[d]).flatten()

    # Compute derivatives if requested
    if n_ders > 0:
        derivatives = []
        for deriv_dir in range(n_dim):
            # For derivative in direction deriv_dir, use d/dxi in that direction
            # and N (values) in all other directions
            deriv_result = ders_1d[deriv_dir][1, :] if n_ders > 0 else np.zeros_like(basis_1d[deriv_dir])

            for d in range(n_dim):
                if d < deriv_dir:
                    deriv_result = np.outer(basis_1d[d], deriv_result).flatten()
                elif d > deriv_dir:
                    deriv_result = np.outer(deriv_result, basis_1d[d]).flatten()

            derivatives.append(deriv_result)

        return result, tuple(derivatives)

    return result, ()


class BSplineBasis:
    """
    Encapsulates a univariate B-spline basis.

    This class bundles a knot vector with methods for basis evaluation,
    providing a cleaner interface for higher-level code.

    Attributes:
        knot_vector: The underlying KnotVector
        degree: Polynomial degree
        n_basis: Number of basis functions
    """

    def __init__(self, knot_vector: KnotVector):
        """
        Initialize a B-spline basis.

        Parameters:
            knot_vector: KnotVector defining the basis
        """
        self.knot_vector = knot_vector

    @property
    def degree(self) -> int:
        return self.knot_vector.degree

    @property
    def n_basis(self) -> int:
        return self.knot_vector.n_basis

    @property
    def n_elements(self) -> int:
        return self.knot_vector.n_elements

    def eval(self, xi: float, span: Optional[int] = None) -> np.ndarray:
        """Evaluate non-zero basis functions at xi."""
        return eval_basis_1d(self.knot_vector, xi, span)

    def eval_ders(self, xi: float, n_ders: int,
                  span: Optional[int] = None) -> np.ndarray:
        """Evaluate basis functions and derivatives at xi."""
        return eval_basis_ders_1d(self.knot_vector, xi, n_ders, span)

    def active_basis_indices(self, element_idx: int) -> np.ndarray:
        """Get indices of basis functions active on an element."""
        return self.knot_vector.active_basis_indices(element_idx)


class TensorProductBasis:
    """
    Tensor-product B-spline basis for multiple dimensions.tensorproduct

    For 2D: N_{i,j}(xi, eta) = N_i(xi) * N_j(eta)
    For 3D: N_{i,j,k}(xi, eta, zeta) = N_i(xi) * N_j(eta) * N_k(zeta)

    TODO: THB-splines will not use pure tensor-product globally
    TODO: T-splines use local tensor-product with different knot vectors
    """

    def __init__(self, bases: Tuple[BSplineBasis, ...]):
        """
        Initialize a tensor-product basis.

        Parameters:
            bases: Tuple of BSplineBasis objects, one per parametric direction
        """
        self.bases = bases
        self.n_dim = len(bases)

    @property
    def degrees(self) -> Tuple[int, ...]:
        """Polynomial degrees in each direction."""
        return tuple(b.degree for b in self.bases)

    @property
    def n_basis_per_dir(self) -> Tuple[int, ...]:
        """Number of basis functions in each direction."""
        return tuple(b.n_basis for b in self.bases)

    @property
    def n_basis_total(self) -> int:
        """Total number of tensor-product basis functions."""
        result = 1
        for b in self.bases:
            result *= b.n_basis
        return result

    @property
    def n_elements_per_dir(self) -> Tuple[int, ...]:
        """Number of elements in each direction."""
        return tuple(b.n_elements for b in self.bases)

    @property
    def n_elements_total(self) -> int:
        """Total number of elements."""
        result = 1
        for b in self.bases:
            result *= b.n_elements
        return result

    def eval(self, xi: Tuple[float, ...]) -> np.ndarray:
        """
        Evaluate tensor-product basis at a parameter point.

        Parameters:
            xi: Parameter values (xi, eta, ...) or (xi, eta, zeta) etc.

        Returns:
            Flat array of non-zero basis function values
        """
        knot_vectors = tuple(b.knot_vector for b in self.bases)
        values, _ = eval_basis_nd(knot_vectors, xi, n_ders=0)
        return values

    def eval_ders(self, xi: Tuple[float, ...],
                  n_ders: int = 1) -> Tuple[np.ndarray, Tuple[np.ndarray, ...]]:
        """
        Evaluate tensor-product basis and derivatives.

        Parameters:
            xi: Parameter values
            n_ders: Number of derivatives per direction

        Returns:
            (values, derivatives) where derivatives is tuple of arrays
        """
        knot_vectors = tuple(b.knot_vector for b in self.bases)
        return eval_basis_nd(knot_vectors, xi, n_ders=n_ders)

    def global_to_tensor_index(self, global_idx: int) -> Tuple[int, ...]:
        """
        Convert flat global index to tensor indices.

        Parameters:
            global_idx: Flat index into all basis functions

        Returns:
            Tuple of indices, one per direction
        """
        indices = []
        remaining = global_idx
        for d in range(self.n_dim - 1, -1, -1):
            n = self.n_basis_per_dir[d]
            indices.insert(0, remaining % n)
            remaining //= n
        return tuple(indices)

    def tensor_to_global_index(self, tensor_idx: Tuple[int, ...]) -> int:
        """
        Convert tensor indices to flat global index.

        Parameters:
            tensor_idx: Tuple of indices, one per direction

        Returns:
            Flat global index
        """
        global_idx = 0
        stride = 1
        for d in range(self.n_dim - 1, -1, -1):
            global_idx += tensor_idx[d] * stride
            stride *= self.n_basis_per_dir[d]
        return global_idx

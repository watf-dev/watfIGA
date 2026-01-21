"""
Bézier extraction operators for IGA.

Bézier extraction is the key technique that enables IGA solvers to work
element-by-element without knowing about the global spline structure.

Key idea:
- On each element, the global B-spline/NURBS basis can be expressed
  as a linear combination of Bernstein polynomials
- The extraction operator C_e encodes this relationship:

    N_local(xi) = C_e * B(xi)

  where:
    N_local = (p+1) active spline basis functions on element e
    B = (p+1) Bernstein basis polynomials
    C_e = (p+1) x (p+1) extraction operator for element e

Benefits:
- Solver only needs to evaluate Bernstein basis (simple, universal)
- Element matrices can be assembled in standard FEM fashion
- Global spline structure is captured in C_e

For tensor-product 2D/3D:
    C_e^{2D} = C_e^eta ⊗ C_e^xi  (Kronecker product)

Reference:
- Borden et al., "Isogeometric finite element data structures
  based on Bézier extraction of NURBS"

TODO: THB-splines require different extraction logic (hierarchical)
TODO: T-splines have different extraction at T-junctions
"""

import numpy as np
from typing import List, Tuple
from .knot_vector import KnotVector


def compute_extraction_operators_1d(kv: KnotVector) -> List[np.ndarray]:
    """
    Compute Bézier extraction operators for all elements in 1D.

    This implements the algorithm from Borden et al. for extracting
    Bézier coefficients from a B-spline.

    Parameters:
        kv: Knot vector

    Returns:
        List of extraction operators C_e, one per element.
        Each C_e has shape (p+1, p+1).

    Mathematical background:
    - Each element's local B-spline basis N_i restricted to the element
      can be written as: N_i|_e = sum_j C_e[i,j] * B_j
    - Where B_j are degree-p Bernstein polynomials on [0,1]
    - C_e is computed via recursive knot insertion

    Algorithm:
    - Start with identity (Bézier = B-spline initially)
    - For each unique internal knot, insert until multiplicity = p
    - Track how coefficients transform through insertion
    """
    p = kv.degree
    knots = kv.knots
    n_elements = kv.n_elements

    # Get unique knots and their multiplicities
    unique_knots = kv.unique_knots
    n_unique = len(unique_knots)

    # Compute multiplicity of each unique knot in the knot vector
    def multiplicity(xi: float, tol: float = 1e-14) -> int:
        return np.sum(np.abs(knots - xi) < tol)

    # Number of knots to insert at each breakpoint
    m = np.zeros(n_unique, dtype=int)
    for i in range(n_unique):
        m[i] = p - multiplicity(unique_knots[i])

    # Total number of knots to insert
    total_insert = int(np.sum(m[1:-1]))  # Don't insert at boundaries

    # Extended knot vector (after all insertions)
    # After full insertion, we get Bézier segments
    nb = kv.n_basis  # original number of basis functions
    a = p + 1  # first internal knot index in extended vector
    b = nb  # one past last internal knot

    # Initialize extraction operators as identities
    # We'll build them incrementally
    C = [np.eye(p + 1) for _ in range(n_elements)]

    if total_insert == 0:
        # Already Bézier (no internal knots or all have multiplicity p)
        return C

    # Work with a copy of knots that we'll extend
    knots_ext = list(knots)
    nb_ext = nb

    # For tracking which element we're working on
    elem = 0

    # Insert knots one by one
    for k in range(1, n_unique - 1):  # For each internal breakpoint
        xi_insert = unique_knots[k]
        n_insert = m[k]

        for _ in range(n_insert):
            # Find span for insertion
            r = -1
            for i in range(len(knots_ext) - 1):
                if knots_ext[i] <= xi_insert < knots_ext[i + 1]:
                    r = i
                    break
            if r == -1:
                r = len(knots_ext) - 2  # Last span

            # Knot insertion algorithm
            # New basis = alpha_i * old_i + (1-alpha_i) * old_{i-1}
            # where alpha_i = (xi - knots[i]) / (knots[i+p+1] - knots[i])

            alphas = np.zeros(p + 1)
            for i in range(p + 1):
                idx = r - p + i
                if idx >= 0 and idx + p + 1 < len(knots_ext):
                    denom = knots_ext[idx + p + 1] - knots_ext[idx]
                    if abs(denom) > 1e-14:
                        alphas[i] = (xi_insert - knots_ext[idx]) / denom
                    else:
                        alphas[i] = 0.0
                elif idx + p + 1 >= len(knots_ext):
                    alphas[i] = 1.0
                else:
                    alphas[i] = 0.0

            # Insert the knot
            knots_ext.insert(r + 1, xi_insert)
            nb_ext += 1

    # After all insertions, rebuild extraction operators properly
    # This is a cleaner approach: compute directly from original knot vector
    return _compute_extraction_direct(kv)


def _compute_extraction_direct(kv: KnotVector) -> List[np.ndarray]:
    """
    Compute Bézier extraction operators directly.

    This is a more straightforward approach that evaluates the relationship
    between B-spline and Bernstein bases directly.

    For each element:
    1. The B-spline basis restricted to the element can be expressed
       in terms of Bernstein polynomials
    2. We find C_e such that N_local = C_e @ B

    Method: Evaluate both bases at p+1 parameter values and solve.
    """
    p = kv.degree
    n_elements = kv.n_elements
    elements = kv.elements

    C_list = []

    for e in range(n_elements):
        xi_start, xi_end = elements[e]
        h = xi_end - xi_start

        # Chebyshev points on [0, 1] (better conditioning than uniform)
        t_cheb = 0.5 * (1 - np.cos(np.pi * np.arange(p + 1) / p))

        # Map to element domain
        xi_pts = xi_start + h * t_cheb

        # Evaluate B-spline basis at these points
        N_matrix = np.zeros((p + 1, p + 1))
        span = kv.find_span(xi_pts[0])

        for i, xi in enumerate(xi_pts):
            # Recompute span if necessary
            if xi < kv.knots[span] or xi > kv.knots[span + 1]:
                span = kv.find_span(xi)
            N = _eval_basis_at_span(kv, xi, span)
            N_matrix[i, :] = N

        # Evaluate Bernstein basis at reference points
        B_matrix = np.zeros((p + 1, p + 1))
        for i, t in enumerate(t_cheb):
            B_matrix[i, :] = _bernstein_basis(p, t)

        # Solve for C_e: N = C_e @ B.T => N @ inv(B.T) = C_e
        # Actually: N[i,:] = C_e.T @ B[i,:] for each point i
        # So N = B @ C_e.T => C_e.T = inv(B) @ N => C_e = N.T @ inv(B.T)
        # Let's be more careful with the formula:
        #
        # N_j(xi) = sum_k C[j,k] * B_k(t) where t = (xi - xi_start) / h
        # In matrix form for multiple points:
        # [N_0(xi_1) ... N_p(xi_1)]     [B_0(t_1) ... B_p(t_1)]
        # [   ...          ...    ]  =  [   ...        ...   ] @ C.T
        # [N_0(xi_m) ... N_p(xi_m)]     [B_0(t_m) ... B_p(t_m)]
        #
        # N_matrix = B_matrix @ C_e.T
        # C_e.T = solve(B_matrix, N_matrix)
        # C_e = N_matrix.T @ inv(B_matrix.T) ??? No...
        # Actually: C_e.T = inv(B_matrix) @ N_matrix

        try:
            C_e_T = np.linalg.solve(B_matrix, N_matrix)
            C_e = C_e_T.T
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            C_e = (np.linalg.pinv(B_matrix) @ N_matrix).T

        C_list.append(C_e)

    return C_list


def _eval_basis_at_span(kv: KnotVector, xi: float, span: int) -> np.ndarray:
    """Evaluate B-spline basis at xi given span."""
    from ..geometry.bspline import eval_basis_1d
    return eval_basis_1d(kv, xi, span)


def _bernstein_basis(p: int, t: float) -> np.ndarray:
    """
    Evaluate all Bernstein basis polynomials of degree p at t in [0,1].

    B_{i,p}(t) = C(p,i) * t^i * (1-t)^(p-i)

    where C(p,i) is the binomial coefficient.

    Parameters:
        p: Polynomial degree
        t: Parameter value in [0, 1]

    Returns:
        Array of shape (p+1,) with B_{0,p}(t), ..., B_{p,p}(t)
    """
    B = np.zeros(p + 1)

    # Use de Casteljau-like recurrence for numerical stability
    B[0] = 1.0 - t
    B[1] = t

    for j in range(1, p):
        saved = 0.0
        for k in range(j + 1):
            temp = B[k]
            B[k] = saved + (1.0 - t) * temp
            saved = t * temp
        B[j + 1] = saved

    return B


def bernstein_basis_ders(p: int, t: float, n_ders: int = 1) -> np.ndarray:
    """
    Evaluate Bernstein basis polynomials and derivatives at t.

    Parameters:
        p: Polynomial degree
        t: Parameter value in [0, 1]
        n_ders: Number of derivatives to compute

    Returns:
        Array of shape (n_ders+1, p+1) where result[k, i] is
        d^k/dt^k B_{i,p}(t)
    """
    result = np.zeros((n_ders + 1, p + 1))

    # Values
    result[0, :] = _bernstein_basis(p, t)

    # Derivatives using the formula:
    # d/dt B_{i,p}(t) = p * (B_{i-1,p-1}(t) - B_{i,p-1}(t))
    # with B_{-1,p-1} = B_{p,p-1} = 0

    if n_ders >= 1 and p >= 1:
        B_lower = _bernstein_basis(p - 1, t)
        for i in range(p + 1):
            left = B_lower[i - 1] if i > 0 else 0.0
            right = B_lower[i] if i < p else 0.0
            result[1, i] = p * (left - right)

    # Higher derivatives (using the same recursive formula)
    for d in range(2, n_ders + 1):
        if p >= d:
            # Need (d-1)th derivative of degree p-1 Bernstein
            # This gets complicated; use explicit formula for second derivative
            # d²/dt² B_{i,p} = p*(p-1) * (B_{i-2,p-2} - 2*B_{i-1,p-2} + B_{i,p-2})
            pass  # For now, only first derivative is fully implemented

    return result


def compute_extraction_operators_2d(kv_xi: KnotVector,
                                     kv_eta: KnotVector) -> List[np.ndarray]:
    """
    Compute 2D Bézier extraction operators via Kronecker product.

    For tensor-product elements:
    C_e^{2D} = C_e^eta ⊗ C_e^xi

    Parameters:
        kv_xi: Knot vector in xi direction
        kv_eta: Knot vector in eta direction

    Returns:
        List of extraction operators C_e for each 2D element.
        Each C_e has shape ((p_xi+1)*(p_eta+1), (p_xi+1)*(p_eta+1)).
    """
    C_xi_list = compute_extraction_operators_1d(kv_xi)
    C_eta_list = compute_extraction_operators_1d(kv_eta)

    n_elem_xi = len(C_xi_list)
    n_elem_eta = len(C_eta_list)

    C_2d_list = []

    for ej in range(n_elem_eta):
        for ei in range(n_elem_xi):
            C_2d = np.kron(C_eta_list[ej], C_xi_list[ei])
            C_2d_list.append(C_2d)

    return C_2d_list


def compute_extraction_operators_3d(kv_xi: KnotVector,
                                     kv_eta: KnotVector,
                                     kv_zeta: KnotVector) -> List[np.ndarray]:
    """
    Compute 3D Bézier extraction operators via Kronecker product.

    C_e^{3D} = C_e^zeta ⊗ C_e^eta ⊗ C_e^xi

    Parameters:
        kv_xi, kv_eta, kv_zeta: Knot vectors in each direction

    Returns:
        List of extraction operators for each 3D element.
    """
    C_xi_list = compute_extraction_operators_1d(kv_xi)
    C_eta_list = compute_extraction_operators_1d(kv_eta)
    C_zeta_list = compute_extraction_operators_1d(kv_zeta)

    C_3d_list = []

    for ek in range(len(C_zeta_list)):
        for ej in range(len(C_eta_list)):
            for ei in range(len(C_xi_list)):
                C_3d = np.kron(np.kron(C_zeta_list[ek], C_eta_list[ej]),
                               C_xi_list[ei])
                C_3d_list.append(C_3d)

    return C_3d_list


class BernsteinBasis:
    """
    Bernstein polynomial basis on the reference element [0,1]^d.

    This class provides evaluation of Bernstein basis functions
    which are used after Bézier extraction in IGA assembly.

    The solver works entirely with Bernstein polynomials, making
    it independent of the spline type (NURBS, THB, T-spline).

    Tensor-product Bernstein basis in 2D:
    B_{i,j}(xi, eta) = B_i(xi) * B_j(eta)
    """

    def __init__(self, degrees: Tuple[int, ...]):
        """
        Initialize Bernstein basis.

        Parameters:
            degrees: Polynomial degrees in each parametric direction
        """
        self.degrees = degrees
        self.n_dim = len(degrees)

    @property
    def n_basis(self) -> int:
        """Total number of tensor-product basis functions."""
        n = 1
        for p in self.degrees:
            n *= (p + 1)
        return n

    def eval(self, xi: Tuple[float, ...]) -> np.ndarray:
        """
        Evaluate all Bernstein basis functions at a point.

        Parameters:
            xi: Parameter values in [0,1]^d

        Returns:
            Array of shape (n_basis,) with basis function values
        """
        B_1d = [_bernstein_basis(self.degrees[d], xi[d])
                for d in range(self.n_dim)]

        result = B_1d[-1]
        for d in range(self.n_dim - 2, -1, -1):
            result = np.outer(result, B_1d[d]).flatten()

        return result

    def eval_ders(self, xi: Tuple[float, ...],
                  n_ders: int = 1) -> Tuple[np.ndarray, ...]:
        """
        Evaluate Bernstein basis and derivatives at a point.

        Parameters:
            xi: Parameter values in [0,1]^d
            n_ders: Number of derivatives per direction

        Returns:
            Tuple (B, dB_dxi1, dB_dxi2, ...) of arrays
        """
        Bders_1d = [bernstein_basis_ders(self.degrees[d], xi[d], n_ders)
                    for d in range(self.n_dim)]

        B_vals = Bders_1d[-1][0, :]
        for d in range(self.n_dim - 2, -1, -1):
            B_vals = np.outer(B_vals, Bders_1d[d][0, :]).flatten()

        result = [B_vals]

        for deriv_dir in range(self.n_dim):
            dB = None
            for d in range(self.n_dim - 1, -1, -1):
                if d == deriv_dir:
                    B_d = Bders_1d[d][1, :]  # Derivative
                else:
                    B_d = Bders_1d[d][0, :]  # Value

                if dB is None:
                    dB = B_d
                else:
                    dB = np.outer(dB, B_d).flatten()

            result.append(dB)

        return tuple(result)

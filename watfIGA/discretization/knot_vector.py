"""
Knot vector utilities for IGA.

A knot vector is a non-decreasing sequence of real numbers that defines
the parametric domain and basis function support for B-splines/NURBS.

Mathematical background:
- Open knot vectors have p+1 repeated knots at each end (interpolatory at boundaries)
- The number of basis functions n = len(knots) - p - 1
- Knot spans (elements) are unique intervals [xi_i, xi_{i+1}] where xi_i < xi_{i+1}

TODO: THB-splines will use hierarchical knot vectors
TODO: T-splines use local knot vectors per control point
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class KnotVector:
    """
    Represents a univariate knot vector.

    Attributes:
        knots: The knot values (non-decreasing sequence)
        degree: Polynomial degree p

    Properties computed:
        n_basis: Number of basis functions
        n_elements: Number of non-zero measure knot spans
        elements: List of (start, end) parametric coordinates for each element
    """
    knots: np.ndarray
    degree: int

    def __post_init__(self):
        self.knots = np.asarray(self.knots, dtype=np.float64)
        self._validate()
        self._compute_elements()

    def _validate(self):
        """Validate knot vector properties."""
        if len(self.knots) < 2 * (self.degree + 1):
            raise ValueError(
                f"Knot vector too short for degree {self.degree}. "
                f"Need at least {2 * (self.degree + 1)} knots, got {len(self.knots)}."
            )
        # Check non-decreasing
        if not np.all(np.diff(self.knots) >= 0):
            raise ValueError("Knot vector must be non-decreasing.")

    def _compute_elements(self):
        """
        Compute unique knot spans (elements).

        Elements are intervals [xi_i, xi_{i+1}] with non-zero measure,
        i.e., where xi_i < xi_{i+1}.
        """
        unique_knots = np.unique(self.knots)
        self._unique_knots = unique_knots
        self._elements = []
        self._element_indices = []  # Store span indices for each element

        for i in range(len(unique_knots) - 1):
            xi_start = unique_knots[i]
            xi_end = unique_knots[i + 1]
            if xi_end > xi_start:
                self._elements.append((xi_start, xi_end))
                # Find the span index (last occurrence of xi_start in knots)
                span_idx = np.searchsorted(self.knots, xi_start, side='right') - 1
                # Ensure span_idx is within valid range
                span_idx = max(self.degree, min(span_idx, self.n_basis - 1))
                self._element_indices.append(span_idx)

    @property
    def n_basis(self) -> int:
        """Number of basis functions."""
        return len(self.knots) - self.degree - 1

    @property
    def n_elements(self) -> int:
        """Number of non-zero measure knot spans (elements)."""
        return len(self._elements)

    @property
    def elements(self) -> List[Tuple[float, float]]:
        """List of element intervals as (xi_start, xi_end) tuples."""
        return self._elements.copy()

    @property
    def unique_knots(self) -> np.ndarray:
        """Unique knot values (breakpoints)."""
        return self._unique_knots.copy()

    @property
    def domain(self) -> Tuple[float, float]:
        """Parametric domain (first unique knot, last unique knot)."""
        return (self._unique_knots[0], self._unique_knots[-1])

    def find_span(self, xi: float) -> int:
        """
        Find the knot span index containing parameter value xi.

        For xi in [xi_i, xi_{i+1}), returns i.
        Uses the convention that the last span is closed: [xi_{n-1}, xi_n].

        Parameters:
            xi: Parameter value

        Returns:
            Span index i such that xi in [xi_i, xi_{i+1})

        Note:
            This returns the span index in the original knot vector,
            not the element index. Use find_element for that.
        """
        n = self.n_basis
        p = self.degree

        # Handle boundary cases
        if xi >= self.knots[n]:
            return n - 1
        if xi <= self.knots[p]:
            return p

        # Binary search
        low = p
        high = n
        mid = (low + high) // 2

        while xi < self.knots[mid] or xi >= self.knots[mid + 1]:
            if xi < self.knots[mid]:
                high = mid
            else:
                low = mid
            mid = (low + high) // 2

        return mid

    def find_element(self, xi: float) -> int:
        """
        Find which element contains parameter value xi.

        Uses half-open interval convention [xi_start, xi_end) for interior
        boundaries, matching the standard IGA convention. The last element
        includes its right boundary (closed at domain end).

        Parameters:
            xi: Parameter value

        Returns:
            Element index (0-based)
        """
        n_elements = len(self._elements)
        for e, (xi_start, xi_end) in enumerate(self._elements):
            if e == n_elements - 1:
                # Last element: closed interval [xi_start, xi_end]
                if xi_start <= xi <= xi_end:
                    return e
            else:
                # Interior element: half-open [xi_start, xi_end)
                if xi_start <= xi < xi_end:
                    return e
        raise ValueError(f"Parameter {xi} outside domain {self.domain}")

    def element_to_span(self, element_idx: int) -> int:
        """
        Convert element index to knot span index.

        Parameters:
            element_idx: Element index (0-based)

        Returns:
            Span index in the original knot vector
        """
        return self._element_indices[element_idx]

    def active_basis_indices(self, element_idx: int) -> np.ndarray:
        """
        Get indices of basis functions active on a given element.

        For a degree p, exactly p+1 basis functions are non-zero on each element.

        Parameters:
            element_idx: Element index

        Returns:
            Array of p+1 global basis function indices
        """
        span = self.element_to_span(element_idx)
        # Basis functions i with support overlapping the span
        # For span index i, active functions are i-p, i-p+1, ..., i
        return np.arange(span - self.degree, span + 1)

    def greville_abscissae(self) -> np.ndarray:
        """
        Compute Greville abscissae (nodal parameters for basis functions).

        The i-th Greville abscissa is the average of p consecutive knots:
        xi_i = (xi_{i+1} + xi_{i+2} + ... + xi_{i+p}) / p

        These are used for approximation and defining control point parameters.

        Returns:
            Array of n Greville abscissae
        """
        p = self.degree
        n = self.n_basis
        greville = np.zeros(n)

        for i in range(n):
            greville[i] = np.sum(self.knots[i + 1:i + p + 1]) / p

        return greville


def make_open_knot_vector(n_basis: int, degree: int,
                           domain: Tuple[float, float] = (0.0, 1.0)) -> KnotVector:
    """
    Create an open (clamped) uniform knot vector.

    Open knot vectors have the first and last knot repeated p+1 times,
    ensuring the basis interpolates the first and last control points.

    Parameters:
        n_basis: Number of basis functions desired
        degree: Polynomial degree p
        domain: Parametric domain (start, end)

    Returns:
        KnotVector with uniform internal knots
    """
    p = degree
    n = n_basis
    n_knots = n + p + 1
    n_internal = n_knots - 2 * (p + 1)

    if n_internal < 0:
        raise ValueError(
            f"Cannot create knot vector: n_basis={n_basis} too small for degree={degree}"
        )

    a, b = domain

    # Start with p+1 repeated knots at start
    knots = [a] * (p + 1)

    # Add uniform internal knots
    if n_internal > 0:
        internal = np.linspace(a, b, n_internal + 2)[1:-1]
        knots.extend(internal)

    # End with p+1 repeated knots at end
    knots.extend([b] * (p + 1))

    return KnotVector(np.array(knots), degree)


def insert_knot(kv: KnotVector, xi: float, times: int = 1) -> KnotVector:
    """
    Insert a knot value into the knot vector.

    This is a utility for refinement. Note that knot insertion
    also requires updating control points (handled elsewhere).

    Parameters:
        kv: Original knot vector
        xi: Knot value to insert
        times: Number of times to insert

    Returns:
        New KnotVector with inserted knots
    """
    new_knots = np.sort(np.concatenate([kv.knots, [xi] * times]))
    return KnotVector(new_knots, kv.degree)


def compute_multiplicity(kv: KnotVector, xi: float, tol: float = 1e-14) -> int:
    """
    Compute the multiplicity of a knot value.

    Parameters:
        kv: Knot vector
        xi: Knot value to check
        tol: Tolerance for equality

    Returns:
        Number of times xi appears in the knot vector
    """
    return np.sum(np.abs(kv.knots - xi) < tol)


def compute_knot_insertion_matrix(kv: KnotVector, xi: float) -> Tuple[KnotVector, np.ndarray]:
    """
    Compute the knot insertion matrix for inserting a single knot.

    When a knot is inserted, control points are updated by a linear transformation:
        P_new = A @ P_old

    Parameters:
        kv: Original knot vector
        xi: Knot value to insert

    Returns:
        Tuple of (new_knot_vector, insertion_matrix A)
        A has shape (n_new, n_old) where n_new = n_old + 1
    """
    p = kv.degree
    knots = kv.knots
    n_old = kv.n_basis

    # Find span containing xi
    k = kv.find_span(xi)

    # New knot vector
    new_knots = np.zeros(len(knots) + 1)
    new_knots[:k + 1] = knots[:k + 1]
    new_knots[k + 1] = xi
    new_knots[k + 2:] = knots[k + 1:]

    n_new = n_old + 1

    # Build insertion matrix
    A = np.zeros((n_new, n_old))

    for i in range(n_new):
        if i <= k - p:
            # Control points before affected region
            A[i, i] = 1.0
        elif i >= k + 1:
            # Control points after affected region
            A[i, i - 1] = 1.0
        else:
            # Affected control points: linear combination
            # alpha_i = (xi - knots[i]) / (knots[i+p] - knots[i])
            denom = knots[i + p] - knots[i]
            if abs(denom) > 1e-14:
                alpha = (xi - knots[i]) / denom
            else:
                alpha = 0.0
            A[i, i - 1] = 1.0 - alpha
            A[i, i] = alpha

    return KnotVector(new_knots, p), A


def refine_knot_vector_dyadic(kv: KnotVector) -> Tuple[KnotVector, np.ndarray]:
    """
    Refine a knot vector by inserting midpoints of all non-zero spans (dyadic refinement).

    This is the standard refinement for hierarchical B-splines.

    Parameters:
        kv: Original knot vector

    Returns:
        Tuple of (refined_knot_vector, refinement_matrix)
        The refinement_matrix transforms old control points to new ones.
    """
    # Get midpoints of all non-zero spans
    midpoints = []
    for xi_start, xi_end in kv.elements:
        mid = 0.5 * (xi_start + xi_end)
        midpoints.append(mid)

    # Sort midpoints
    midpoints = sorted(midpoints)

    # Apply knot insertions sequentially
    current_kv = kv
    A_total = np.eye(kv.n_basis)

    for xi in midpoints:
        new_kv, A = compute_knot_insertion_matrix(current_kv, xi)
        A_total = A @ A_total
        current_kv = new_kv

    return current_kv, A_total


def compute_refinement_matrix(kv_coarse: KnotVector, kv_fine: KnotVector) -> np.ndarray:
    """
    Compute the refinement matrix between two nested knot vectors.

    The refinement matrix T satisfies:
        N_coarse(xi) = T @ N_fine(xi)

    where N_coarse and N_fine are the basis function vectors.

    Parameters:
        kv_coarse: Coarse level knot vector
        kv_fine: Fine level knot vector (must contain all knots from coarse)

    Returns:
        Refinement matrix T with shape (n_coarse, n_fine)
    """
    # Verify nesting: all coarse knots must be in fine
    for knot in kv_coarse.knots:
        if not np.any(np.abs(kv_fine.knots - knot) < 1e-14):
            raise ValueError("Fine knot vector must contain all coarse knots")

    # Find knots to insert (knots in fine but not in coarse)
    knots_to_insert = []
    for knot in kv_fine.knots:
        mult_fine = compute_multiplicity(kv_fine, knot)
        mult_coarse = compute_multiplicity(kv_coarse, knot)
        for _ in range(mult_fine - mult_coarse):
            knots_to_insert.append(knot)

    # Sort and apply insertions
    knots_to_insert = sorted(knots_to_insert)

    current_kv = kv_coarse
    A_total = np.eye(kv_coarse.n_basis)

    for xi in knots_to_insert:
        new_kv, A = compute_knot_insertion_matrix(current_kv, xi)
        A_total = A @ A_total
        current_kv = new_kv

    # T is the transpose relationship: coarse = T @ fine coefficients
    # But we computed fine = A @ coarse, so T = A
    return A_total

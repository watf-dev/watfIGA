#!/usr/bin/env python3
"""
THB-splines (Truncated Hierarchical B-splines).

THB-splines are a hierarchical refinement technique that allows
local mesh refinement while maintaining the analysis-suitable
properties of B-splines.

Key concepts:
1. Hierarchical basis: Multiple levels of B-spline bases
2. Truncation: Modified basis functions at refinement boundaries
3. Linear independence: Truncation ensures basis functions are LI

Implementation approach: Hierarchical Bézier extraction
- Each element has an extraction operator mapping Bernstein to THB basis
- The solver remains unchanged (only works with extraction operators)
- Truncation is encoded in the extraction operators

Created: 2025-01-19
Author: Wataru Fukuda
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field

from ..discretization.knot_vector import (
    KnotVector, make_open_knot_vector,
    compute_knot_insertion_matrix, refine_knot_vector_dyadic,
    compute_refinement_matrix
)


@dataclass
class THBHierarchy1D:
    """
    1D hierarchical knot vector structure for THB-splines.

    Manages multiple levels of nested knot vectors and tracks
    which basis functions are active at each level.

    Attributes:
        knot_vectors: List of knot vectors, one per level
        refinement_matrices: Matrices relating adjacent levels
        active_basis: Set of active basis function indices per level
        degree: Polynomial degree (same for all levels)
    """
    degree: int
    knot_vectors: List[KnotVector] = field(default_factory=list)
    refinement_matrices: List[np.ndarray] = field(default_factory=list)
    active_basis: List[Set[int]] = field(default_factory=list)

    @classmethod
    def from_knot_vector(cls, kv: KnotVector) -> 'THBHierarchy1D':
        """
        Create hierarchy from initial (level 0) knot vector.

        Parameters:
            kv: Initial knot vector

        Returns:
            THBHierarchy1D with single level
        """
        hierarchy = cls(degree=kv.degree)
        hierarchy.knot_vectors = [kv]
        hierarchy.refinement_matrices = []
        # All basis functions active at level 0 initially
        hierarchy.active_basis = [set(range(kv.n_basis))]
        return hierarchy

    @property
    def n_levels(self) -> int:
        """Number of refinement levels."""
        return len(self.knot_vectors)

    @property
    def max_level(self) -> int:
        """Maximum level index (0-based)."""
        return self.n_levels - 1

    def get_knot_vector(self, level: int) -> KnotVector:
        """Get knot vector at specified level."""
        return self.knot_vectors[level]

    def get_n_basis(self, level: int) -> int:
        """Get number of basis functions at specified level."""
        return self.knot_vectors[level].n_basis

    def get_active_basis(self, level: int) -> Set[int]:
        """Get set of active basis function indices at specified level."""
        return self.active_basis[level].copy()

    def get_n_active_basis(self, level: int) -> int:
        """Get number of active basis functions at specified level."""
        return len(self.active_basis[level])

    def get_total_active_basis(self) -> int:
        """Get total number of active basis functions across all levels."""
        return sum(len(ab) for ab in self.active_basis)

    def add_level(self) -> None:
        """
        Add a new refinement level using dyadic refinement.

        Creates level n+1 from level n by inserting midpoints.
        """
        current_kv = self.knot_vectors[-1]
        new_kv, T = refine_knot_vector_dyadic(current_kv)

        self.knot_vectors.append(new_kv)
        self.refinement_matrices.append(T)
        # New level starts with no active basis (activated during refinement)
        self.active_basis.append(set())

    def ensure_level(self, level: int) -> None:
        """Ensure hierarchy has at least the specified level."""
        while self.n_levels <= level:
            self.add_level()

    def get_refinement_matrix(self, from_level: int) -> np.ndarray:
        """
        Get refinement matrix from level to level+1.

        The matrix T satisfies: N_l = T @ N_{l+1}
        (coarse basis as linear combination of fine basis)
        """
        return self.refinement_matrices[from_level]

    def activate_basis(self, level: int, indices: Set[int]) -> None:
        """Activate basis functions at specified level."""
        self.active_basis[level].update(indices)

    def deactivate_basis(self, level: int, indices: Set[int]) -> None:
        """Deactivate basis functions at specified level."""
        self.active_basis[level] -= indices

    def get_basis_support(self, level: int, basis_idx: int) -> Tuple[float, float]:
        """
        Get parametric support of a basis function.

        Parameters:
            level: Refinement level
            basis_idx: Basis function index at that level

        Returns:
            (xi_min, xi_max) support interval
        """
        kv = self.knot_vectors[level]
        p = self.degree
        # Support is [knots[i], knots[i+p+1]]
        return (kv.knots[basis_idx], kv.knots[basis_idx + p + 1])

    def find_overlapping_fine_basis(self, level: int, basis_idx: int) -> Set[int]:
        """
        Find fine-level basis functions that overlap with a coarse basis function.

        Used for truncation: when refining, we need to know which fine basis
        functions overlap with each coarse basis function.

        Parameters:
            level: Coarse level
            basis_idx: Basis function index at coarse level

        Returns:
            Set of basis indices at level+1 that have overlapping support
        """
        if level >= self.max_level:
            return set()

        xi_min, xi_max = self.get_basis_support(level, basis_idx)
        fine_kv = self.knot_vectors[level + 1]

        overlapping = set()
        for j in range(fine_kv.n_basis):
            fine_min, fine_max = self.get_basis_support(level + 1, j)
            # Check overlap
            if fine_min < xi_max and fine_max > xi_min:
                overlapping.add(j)

        return overlapping


@dataclass
class THBHierarchy2D:
    """
    2D hierarchical structure for THB-splines (tensor product).

    Manages hierarchies in both parametric directions.
    """
    hierarchy_xi: THBHierarchy1D
    hierarchy_eta: THBHierarchy1D

    @classmethod
    def from_knot_vectors(cls, kv_xi: KnotVector, kv_eta: KnotVector) -> 'THBHierarchy2D':
        """Create 2D hierarchy from initial knot vectors."""
        return cls(
            hierarchy_xi=THBHierarchy1D.from_knot_vector(kv_xi),
            hierarchy_eta=THBHierarchy1D.from_knot_vector(kv_eta)
        )

    @property
    def n_levels(self) -> int:
        """Number of refinement levels (must be same in both directions)."""
        return min(self.hierarchy_xi.n_levels, self.hierarchy_eta.n_levels)

    @property
    def degrees(self) -> Tuple[int, int]:
        """Polynomial degrees in each direction."""
        return (self.hierarchy_xi.degree, self.hierarchy_eta.degree)

    def ensure_level(self, level: int) -> None:
        """Ensure both hierarchies have at least the specified level."""
        self.hierarchy_xi.ensure_level(level)
        self.hierarchy_eta.ensure_level(level)

    def get_n_basis_per_dir(self, level: int) -> Tuple[int, int]:
        """Get number of basis functions per direction at specified level."""
        return (
            self.hierarchy_xi.get_n_basis(level),
            self.hierarchy_eta.get_n_basis(level)
        )

    def get_n_basis(self, level: int) -> int:
        """Get total number of basis functions at specified level."""
        n_xi, n_eta = self.get_n_basis_per_dir(level)
        return n_xi * n_eta

    def tensor_to_global(self, level: int, i: int, j: int) -> int:
        """Convert tensor indices (i, j) to global index at specified level."""
        n_xi = self.hierarchy_xi.get_n_basis(level)
        return j * n_xi + i

    def global_to_tensor(self, level: int, idx: int) -> Tuple[int, int]:
        """Convert global index to tensor indices (i, j) at specified level."""
        n_xi = self.hierarchy_xi.get_n_basis(level)
        i = idx % n_xi
        j = idx // n_xi
        return (i, j)

    def get_element_bounds(self, level: int, elem_i: int, elem_j: int) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Get parametric bounds for element at specified level."""
        kv_xi = self.hierarchy_xi.get_knot_vector(level)
        kv_eta = self.hierarchy_eta.get_knot_vector(level)
        xi_bounds = kv_xi.elements[elem_i]
        eta_bounds = kv_eta.elements[elem_j]
        return (xi_bounds, eta_bounds)

    def get_n_elements_per_dir(self, level: int) -> Tuple[int, int]:
        """Get number of elements per direction at specified level."""
        return (
            self.hierarchy_xi.get_knot_vector(level).n_elements,
            self.hierarchy_eta.get_knot_vector(level).n_elements
        )


class THBSurface:
    """
    THB-spline surface with hierarchical refinement.

    This class manages a NURBS/B-spline surface with local refinement
    capabilities using THB-splines.

    The key design: Bézier extraction operators encode the truncation,
    so the solver only sees standard extraction operators.

    Basis function states:
    1. Active & untruncated: Support entirely outside refined region
    2. Active & truncated: Support partially overlaps refined region
    3. Inactive (deactivated): Support entirely inside refined region
    """

    def __init__(
        self,
        hierarchy: THBHierarchy2D,
        control_points: np.ndarray,
        weights: Optional[np.ndarray] = None
    ):
        """
        Initialize THB surface.

        Parameters:
            hierarchy: 2D hierarchical structure
            control_points: Initial control points (level 0), shape (n_cp, n_dim)
            weights: NURBS weights (optional), shape (n_cp,)
        """
        self.hierarchy = hierarchy
        self._control_points = {0: control_points.copy()}

        n_cp = control_points.shape[0]
        if weights is None:
            self._weights = {0: np.ones(n_cp)}
        else:
            self._weights = {0: weights.copy()}

        # Track active control points per level
        # Initially all level 0 control points are active
        self._active_cps: Dict[int, Set[int]] = {0: set(range(n_cp))}

        # Track refined elements (elements that have been subdivided)
        self._refined_elements: Dict[int, Set[Tuple[int, int]]] = {0: set()}

        # Track truncated basis functions and their truncation coefficients
        # Key: (level, local_idx), Value: dict of fine basis contributions to subtract
        # truncation_coeffs[(l, i)][(l+1, j)] = coefficient c_ij
        self._truncated_basis: Dict[int, Set[int]] = {}  # level -> set of truncated basis indices
        self._truncation_coeffs: Dict[Tuple[int, int], Dict[Tuple[int, int], float]] = {}

    @classmethod
    def from_nurbs_surface(cls, surface) -> 'THBSurface':
        """
        Create THB surface from a standard NURBS surface.

        Parameters:
            surface: NURBSSurface object

        Returns:
            THBSurface with the same geometry
        """
        kv_xi, kv_eta = surface.knot_vectors
        hierarchy = THBHierarchy2D.from_knot_vectors(kv_xi, kv_eta)

        return cls(
            hierarchy=hierarchy,
            control_points=surface.control_points,
            weights=surface.weights
        )

    @property
    def n_levels(self) -> int:
        """Number of refinement levels."""
        return self.hierarchy.n_levels

    @property
    def degrees(self) -> Tuple[int, int]:
        """Polynomial degrees."""
        return self.hierarchy.degrees

    @property
    def n_dim_parametric(self) -> int:
        """Number of parametric dimensions."""
        return 2

    @property
    def n_dim_physical(self) -> int:
        """Number of physical dimensions."""
        return self._control_points[0].shape[1]

    def get_control_points(self, level: int) -> np.ndarray:
        """Get control points at specified level."""
        return self._control_points[level]

    def get_weights(self, level: int) -> np.ndarray:
        """Get weights at specified level."""
        return self._weights[level]

    def get_active_control_points(self, level: int) -> Set[int]:
        """Get active control point indices at specified level."""
        return self._active_cps.get(level, set()).copy()

    def get_total_active_dofs(self) -> int:
        """Get total number of active DOFs across all levels."""
        return sum(len(cps) for cps in self._active_cps.values())

    def get_truncated_basis(self, level: int) -> Set[int]:
        """
        Get set of truncated basis function indices at specified level.

        Truncated basis functions are active but have their support
        modified (reduced) in the refined region.
        """
        return self._truncated_basis.get(level, set()).copy()

    def is_basis_truncated(self, level: int, local_idx: int) -> bool:
        """Check if a basis function is truncated."""
        return local_idx in self._truncated_basis.get(level, set())

    def get_truncation_coefficients(self, level: int, local_idx: int) -> Dict[Tuple[int, int], float]:
        """
        Get truncation coefficients for a truncated basis function.

        Returns a dictionary mapping (fine_level, fine_idx) -> coefficient.
        The truncated function is: trunc(N_i^l) = N_i^l - sum_j c_ij * N_j^{l+1}

        Parameters:
            level: Level of the basis function
            local_idx: Local index at that level

        Returns:
            Dictionary of truncation coefficients (empty if not truncated)
        """
        return self._truncation_coeffs.get((level, local_idx), {}).copy()

    def is_element_refined(self, level: int, elem_i: int, elem_j: int) -> bool:
        """Check if an element has been refined."""
        return (elem_i, elem_j) in self._refined_elements.get(level, set())

    def refine_element(self, level: int, elem_i: int, elem_j: int) -> None:
        """
        Refine a single element (mark for subdivision).

        This implements element-wise refinement for THB-splines.
        After calling this, the mesh needs to be rebuilt.

        Parameters:
            level: Current level of the element
            elem_i: Element index in xi direction
            elem_j: Element index in eta direction
        """
        # Ensure next level exists
        self.hierarchy.ensure_level(level + 1)

        # Mark element as refined
        if level not in self._refined_elements:
            self._refined_elements[level] = set()
        self._refined_elements[level].add((elem_i, elem_j))

        # Compute control points for level+1 if not already done
        if level + 1 not in self._control_points:
            self._compute_refined_control_points(level)

        # Update active control points
        self._update_active_basis_for_refinement(level, elem_i, elem_j)

    def _compute_refined_control_points(self, from_level: int) -> None:
        """
        Compute control points for level+1 from level.

        Uses knot insertion to compute the refined control points.
        """
        to_level = from_level + 1

        # Get refinement matrices
        T_xi = self.hierarchy.hierarchy_xi.get_refinement_matrix(from_level)
        T_eta = self.hierarchy.hierarchy_eta.get_refinement_matrix(from_level)

        # 2D refinement matrix (Kronecker product)
        T_2d = np.kron(T_eta, T_xi)

        # Current control points (flattened in x-fastest order)
        cp_coarse = self._control_points[from_level]
        w_coarse = self._weights[from_level]

        # Refine control points: P_fine = T @ P_coarse
        # For NURBS, we work with weighted control points
        weighted_cp = cp_coarse * w_coarse[:, np.newaxis]
        weighted_cp_fine = T_2d @ weighted_cp
        w_fine = T_2d @ w_coarse

        # Unweight
        cp_fine = weighted_cp_fine / w_fine[:, np.newaxis]

        self._control_points[to_level] = cp_fine
        self._weights[to_level] = w_fine

        # Initialize active set for new level (empty until elements are refined)
        if to_level not in self._active_cps:
            self._active_cps[to_level] = set()

    def _update_active_basis_for_refinement(
        self, level: int, elem_i: int, elem_j: int
    ) -> None:
        """
        Update active basis functions when an element is refined.

        THB rule: When an element is refined, the coarse basis functions
        whose support is entirely covered by refined elements are deactivated,
        and the corresponding fine basis functions are activated.
        """
        to_level = level + 1
        kv_xi = self.hierarchy.hierarchy_xi.get_knot_vector(level)
        kv_eta = self.hierarchy.hierarchy_eta.get_knot_vector(level)

        # Get parametric bounds of refined element
        xi_min, xi_max = kv_xi.elements[elem_i]
        eta_min, eta_max = kv_eta.elements[elem_j]

        p_xi, p_eta = self.degrees

        # Find which coarse basis functions to potentially deactivate
        # A coarse basis function is deactivated if its support is ENTIRELY
        # covered by refined elements
        # For now (single-level), we deactivate coarse basis in the element
        # and activate corresponding fine basis

        # Compute fine basis functions that cover this element
        kv_xi_fine = self.hierarchy.hierarchy_xi.get_knot_vector(to_level)
        kv_eta_fine = self.hierarchy.hierarchy_eta.get_knot_vector(to_level)
        n_xi_fine = kv_xi_fine.n_basis

        # Find fine elements that cover the coarse element
        # With dyadic refinement, each coarse element splits into 4 fine elements
        fine_elem_xi_start = 2 * elem_i
        fine_elem_xi_end = 2 * elem_i + 2
        fine_elem_eta_start = 2 * elem_j
        fine_elem_eta_end = 2 * elem_j + 2

        # Collect all fine basis functions active on these fine elements
        fine_basis_to_activate = set()
        for fej in range(fine_elem_eta_start, fine_elem_eta_end):
            span_eta_fine = kv_eta_fine.element_to_span(fej)
            active_eta_fine = range(span_eta_fine - p_eta, span_eta_fine + 1)

            for fei in range(fine_elem_xi_start, fine_elem_xi_end):
                span_xi_fine = kv_xi_fine.element_to_span(fei)
                active_xi_fine = range(span_xi_fine - p_xi, span_xi_fine + 1)

                for j in active_eta_fine:
                    for i in active_xi_fine:
                        global_idx = j * n_xi_fine + i
                        fine_basis_to_activate.add(global_idx)

        # Activate fine basis functions
        self._active_cps[to_level].update(fine_basis_to_activate)

        # For single-level refinement, we need to track which coarse basis
        # to deactivate. A coarse basis is deactivated when its entire support
        # is covered by refined elements.
        # For simplicity, we mark coarse basis on this element for potential
        # deactivation (full check done in finalize_refinement)

    def finalize_refinement(self) -> None:
        """
        Finalize refinement after marking elements.

        This computes the final active basis sets based on THB truncation rules:
        1. Filter fine basis functions - only those with support entirely in refined region
        2. Apply truncation to coarse basis functions
        """
        for level in range(self.n_levels - 1):
            # Step 1: Filter fine basis functions
            self._filter_fine_basis_functions(level)
            # Step 2: Apply truncation rules
            self._apply_truncation_at_level(level)

    def _filter_fine_basis_functions(self, coarse_level: int) -> None:
        """
        Filter fine basis functions to only include those with support entirely
        within the refined region.

        In THB-splines, only fine basis functions whose support is ENTIRELY
        contained in Ω^{l+1} (the refined domain) are active. Functions at
        the boundary (with support extending outside) are NOT included.
        """
        fine_level = coarse_level + 1
        if fine_level >= self.n_levels:
            return

        if coarse_level not in self._refined_elements:
            return

        refined_elems = self._refined_elements[coarse_level]
        if not refined_elems:
            return

        # Convert coarse refined elements to fine elements
        fine_refined_elements = set()
        for (ei, ej) in refined_elems:
            for di in range(2):
                for dj in range(2):
                    fine_refined_elements.add((2*ei + di, 2*ej + dj))

        # Get fine knot vectors
        kv_xi_fine = self.hierarchy.hierarchy_xi.get_knot_vector(fine_level)
        kv_eta_fine = self.hierarchy.hierarchy_eta.get_knot_vector(fine_level)
        n_xi_fine = kv_xi_fine.n_basis

        # Filter active fine basis functions
        active_fine = self._active_cps.get(fine_level, set())
        filtered_active = set()

        for global_idx in active_fine:
            i_fine = global_idx % n_xi_fine
            j_fine = global_idx // n_xi_fine

            if self._is_fine_basis_interior(fine_level, i_fine, j_fine, fine_refined_elements):
                filtered_active.add(global_idx)

        self._active_cps[fine_level] = filtered_active

    def _apply_truncation_at_level(self, level: int) -> None:
        """
        Apply THB truncation rules at a level.

        For each coarse basis function, determine its state:
        1. Untruncated: Support has NO overlap with refined region
        2. Truncated: Support PARTIALLY overlaps refined region
        3. Deactivated: Support FULLY inside refined region

        For truncated basis functions, compute the truncation coefficients
        that define how to subtract the fine-level contribution.
        """
        if level not in self._refined_elements:
            return

        refined_elems = self._refined_elements[level]
        if not refined_elems:
            return

        kv_xi = self.hierarchy.hierarchy_xi.get_knot_vector(level)
        kv_eta = self.hierarchy.hierarchy_eta.get_knot_vector(level)
        n_xi = kv_xi.n_basis
        p_xi, p_eta = self.degrees

        # Initialize truncated basis set for this level
        if level not in self._truncated_basis:
            self._truncated_basis[level] = set()

        coarse_to_deactivate = set()
        coarse_to_truncate = set()

        for global_idx in list(self._active_cps.get(level, set())):
            i = global_idx % n_xi
            j = global_idx // n_xi

            # Get support of this basis function
            xi_supp = self.hierarchy.hierarchy_xi.get_basis_support(level, i)
            eta_supp = self.hierarchy.hierarchy_eta.get_basis_support(level, j)

            # Count elements in support and how many are refined
            elements_in_support = []
            refined_in_support = []

            for elem_j, (eta_start, eta_end) in enumerate(kv_eta.elements):
                if eta_end <= eta_supp[0] or eta_start >= eta_supp[1]:
                    continue  # Element not in eta support

                for elem_i, (xi_start, xi_end) in enumerate(kv_xi.elements):
                    if xi_end <= xi_supp[0] or xi_start >= xi_supp[1]:
                        continue  # Element not in xi support

                    elements_in_support.append((elem_i, elem_j))
                    if (elem_i, elem_j) in refined_elems:
                        refined_in_support.append((elem_i, elem_j))

            n_total = len(elements_in_support)
            n_refined = len(refined_in_support)

            if n_refined == 0:
                # No overlap with refined region - stays untruncated
                pass
            elif n_refined == n_total:
                # Fully inside refined region - deactivate
                coarse_to_deactivate.add(global_idx)
            else:
                # Partially overlaps - truncate
                coarse_to_truncate.add(global_idx)
                # Compute truncation coefficients using ALL refined elements
                # (not just those in this basis function's support)
                self._compute_truncation_coefficients(level, global_idx, list(refined_elems))

        # Deactivate fully-covered basis functions
        self._active_cps[level] -= coarse_to_deactivate

        # Mark truncated basis functions
        self._truncated_basis[level].update(coarse_to_truncate)

    def _compute_truncation_coefficients(
        self,
        level: int,
        coarse_idx: int,
        refined_elements: List[Tuple[int, int]]
    ) -> None:
        """
        Compute truncation coefficients for a coarse basis function.

        The truncated function is defined as:
            trunc(N_i^l) = N_i^l - sum_{j in Omega_refined} c_ij * N_j^{l+1}

        where c_ij are the coefficients from the refinement relation:
            N_i^l = sum_j c_ij * N_j^{l+1}

        IMPORTANT: The sum is only over fine basis functions whose support is
        ENTIRELY within the refined region. Functions at the boundary (with
        support extending outside) are NOT included in the truncation.

        Parameters:
            level: Coarse level
            coarse_idx: Global index of coarse basis function
            refined_elements: List of ALL refined elements at this level (defines Ω^{l+1})
        """
        fine_level = level + 1
        if fine_level >= self.n_levels:
            return

        # Get refinement matrices
        T_xi = self.hierarchy.hierarchy_xi.get_refinement_matrix(level)
        T_eta = self.hierarchy.hierarchy_eta.get_refinement_matrix(level)

        # Decompose coarse index to tensor indices
        n_xi_coarse = self.hierarchy.hierarchy_xi.get_n_basis(level)
        n_xi_fine = self.hierarchy.hierarchy_xi.get_n_basis(fine_level)
        n_eta_fine = self.hierarchy.hierarchy_eta.get_n_basis(fine_level)

        i_coarse = coarse_idx % n_xi_coarse
        j_coarse = coarse_idx // n_xi_coarse

        fine_kv_xi = self.hierarchy.hierarchy_xi.get_knot_vector(fine_level)
        fine_kv_eta = self.hierarchy.hierarchy_eta.get_knot_vector(fine_level)

        # Get all fine elements that correspond to the refined coarse elements
        fine_refined_elements = set()
        for (ei, ej) in refined_elements:
            # Each coarse element maps to 2x2 fine elements
            for di in range(2):
                for dj in range(2):
                    fine_refined_elements.add((2*ei + di, 2*ej + dj))

        truncation_coeffs = {}

        # Iterate over fine basis functions
        # T has shape (n_fine, n_coarse): T[j_fine, i_coarse] = coefficient
        for j_fine in range(n_eta_fine):
            # Check if this fine basis in eta is contributed by coarse basis
            if T_eta[j_fine, j_coarse] == 0:
                continue

            for i_fine in range(n_xi_fine):
                # Check if this fine basis in xi is contributed by coarse basis
                if T_xi[i_fine, i_coarse] == 0:
                    continue

                fine_global_idx = j_fine * n_xi_fine + i_fine

                # Check if this fine basis function is active
                if fine_global_idx not in self._active_cps.get(fine_level, set()):
                    continue

                # CRITICAL: Only include fine basis functions whose support is
                # ENTIRELY within the refined region
                if not self._is_fine_basis_interior(fine_level, i_fine, j_fine, fine_refined_elements):
                    continue

                # The coefficient is the product of 1D refinement coefficients
                coeff = T_xi[i_fine, i_coarse] * T_eta[j_fine, j_coarse]

                if abs(coeff) > 1e-14:
                    truncation_coeffs[(fine_level, fine_global_idx)] = coeff

        if truncation_coeffs:
            self._truncation_coeffs[(level, coarse_idx)] = truncation_coeffs

    def _is_fine_basis_interior(
        self,
        fine_level: int,
        i_fine: int,
        j_fine: int,
        fine_refined_elements: Set[Tuple[int, int]]
    ) -> bool:
        """
        Check if a fine basis function's support is entirely within the refined region.

        A basis function is "interior" if ALL elements in its support are refined.

        Parameters:
            fine_level: Fine level
            i_fine, j_fine: Tensor indices of fine basis function
            fine_refined_elements: Set of fine element indices that are refined

        Returns:
            True if support is entirely within refined region
        """
        fine_kv_xi = self.hierarchy.hierarchy_xi.get_knot_vector(fine_level)
        fine_kv_eta = self.hierarchy.hierarchy_eta.get_knot_vector(fine_level)

        # Get support of fine basis function
        xi_supp = self.hierarchy.hierarchy_xi.get_basis_support(fine_level, i_fine)
        eta_supp = self.hierarchy.hierarchy_eta.get_basis_support(fine_level, j_fine)

        # Check all fine elements in the support
        for elem_j, (eta_start, eta_end) in enumerate(fine_kv_eta.elements):
            if eta_end <= eta_supp[0] or eta_start >= eta_supp[1]:
                continue  # Element not in eta support

            for elem_i, (xi_start, xi_end) in enumerate(fine_kv_xi.elements):
                if xi_end <= xi_supp[0] or xi_start >= xi_supp[1]:
                    continue  # Element not in xi support

                # This element is in the support - check if it's refined
                if (elem_i, elem_j) not in fine_refined_elements:
                    return False  # Found an unrefined element in support

        return True  # All elements in support are refined

    def get_knot_vectors(self, level: int = 0) -> Tuple[KnotVector, KnotVector]:
        """Get knot vectors at specified level."""
        return (
            self.hierarchy.hierarchy_xi.get_knot_vector(level),
            self.hierarchy.hierarchy_eta.get_knot_vector(level)
        )

    @property
    def n_control_points_per_dir(self) -> Tuple[int, int]:
        """Number of control points per direction at level 0."""
        return self.hierarchy.get_n_basis_per_dir(0)

    @property
    def n_control_points(self) -> int:
        """Total number of control points at level 0."""
        return self.hierarchy.get_n_basis(0)

    @property
    def control_points(self) -> np.ndarray:
        """Control points at level 0 (for compatibility)."""
        return self._control_points[0]

    @property
    def weights(self) -> np.ndarray:
        """Weights at level 0 (for compatibility)."""
        return self._weights[0]

    @property
    def domain(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Parametric domain."""
        kv_xi = self.hierarchy.hierarchy_xi.get_knot_vector(0)
        kv_eta = self.hierarchy.hierarchy_eta.get_knot_vector(0)
        return (kv_xi.domain, kv_eta.domain)

    def eval_point(self, xi: Tuple[float, float]) -> np.ndarray:
        """
        Evaluate surface at parameter values.

        For THB-splines, the geometry is defined by the level-0 (coarsest)
        basis functions and control points. Refinement adds DOFs for better
        solution approximation but preserves the geometry.

        Parameters:
            xi: Parameter values (xi, eta)

        Returns:
            Point coordinates as (d,) array
        """
        from ..geometry.bspline import eval_basis_ders_1d

        xi_val, eta_val = xi

        kv_xi, kv_eta = self.get_knot_vectors(0)
        p_xi, p_eta = self.degrees
        n_xi = kv_xi.n_basis

        control_points = self._control_points[0]
        weights = self._weights[0]

        # Find spans
        span_xi = kv_xi.find_span(xi_val)
        span_eta = kv_eta.find_span(eta_val)

        # Evaluate basis functions
        N_xi = eval_basis_ders_1d(kv_xi, xi_val, 0, span_xi)[0, :]
        N_eta = eval_basis_ders_1d(kv_eta, eta_val, 0, span_eta)[0, :]

        # Compute NURBS point
        point = np.zeros(self.n_dim_physical)
        W = 0.0

        for jj in range(p_eta + 1):
            for ii in range(p_xi + 1):
                glob_i = span_xi - p_xi + ii
                glob_j = span_eta - p_eta + jj
                idx = glob_j * n_xi + glob_i

                Nij = N_xi[ii] * N_eta[jj]
                w = weights[idx]
                Rij = Nij * w

                point += Rij * control_points[idx]
                W += Rij

        return point / W

"""
Element abstraction for IGA.

In IGA, an element corresponds to a non-zero measure knot span.
Each element:
- Has a parametric domain (e.g., [xi_min, xi_max] x [eta_min, eta_max])
- Knows its active control points (bidirectional linking)
- Holds a Bézier extraction operator
- Has a refinement level (for hierarchical splines)
- Can be active or inactive

Key design principles:
1. Elements explicitly reference control points by ID (not just DOF indices)
2. Elements can be deactivated (THB refinement)
3. The element is solver-ready: contains everything needed for assembly
4. No knowledge of global knot vectors or specific spline type

Bidirectional linking invariant:
    cp_id in element.control_point_ids  <=>  element.id in control_points[cp_id].supported_elements

This design enables:
- THB-splines: Child elements can replace parent elements
- T-splines: Non-tensor-product connectivity
- Solver: Iterates over active elements only
"""

import numpy as np
from typing import Tuple, List, Set, Optional
from dataclasses import dataclass, field


@dataclass
class Element:
    """
    IGA element with explicit control point linking.

    Attributes:
        id: Unique element identifier
        level: Refinement level (0 = coarsest)
        parametric_bounds: ((xi_min, xi_max), ...) bounds in each direction
        control_point_ids: List of active control point IDs for this element
        extraction_operator: C_e matrix mapping Bernstein to spline basis
        degrees: Polynomial degrees in each direction
        active: Whether this element is active in the analysis
        parent_id: ID of parent element at coarser level (for THB-splines)
        children_ids: IDs of child elements at finer level (for THB-splines)

    Design notes:
        - control_point_ids replaces the old global_dof_indices
        - The order in control_point_ids matches extraction_operator rows
        - For scalar problems: DOF index = control point ID
        - For vector problems: multiple DOFs per control point
    """
    id: int
    level: int
    parametric_bounds: Tuple[Tuple[float, float], ...]
    control_point_ids: List[int]
    extraction_operator: np.ndarray
    degrees: Tuple[int, ...]
    active: bool = True

    # THB-spline hierarchical relationships
    parent_id: Optional[int] = None
    children_ids: List[int] = field(default_factory=list)

    # Cached geometry data (populated by mesh builder if needed)
    _local_coordinates: Optional[np.ndarray] = field(default=None, repr=False)
    _local_weights: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def n_dim(self) -> int:
        """Number of parametric dimensions."""
        return len(self.parametric_bounds)

    @property
    def n_local_basis(self) -> int:
        """Number of local basis functions on this element."""
        return self.extraction_operator.shape[0]

    @property
    def n_control_points(self) -> int:
        """Number of control points for this element."""
        return len(self.control_point_ids)

    @property
    def global_dof_indices(self) -> np.ndarray:
        """
        Global DOF indices (for backward compatibility).

        For scalar problems, DOF index = control point ID.
        """
        return np.array(self.control_point_ids, dtype=int)

    @property
    def control_points(self) -> Optional[np.ndarray]:
        """Local control point coordinates (if cached)."""
        return self._local_coordinates

    @property
    def weights(self) -> Optional[np.ndarray]:
        """Local NURBS weights (if cached)."""
        return self._local_weights

    def set_geometry_cache(self, coordinates: np.ndarray, weights: np.ndarray) -> None:
        """
        Cache local geometry data for this element.

        Parameters:
            coordinates: Control point coordinates, shape (n_local, n_dim_physical)
            weights: NURBS weights, shape (n_local,)
        """
        self._local_coordinates = coordinates
        self._local_weights = weights

    def clear_geometry_cache(self) -> None:
        """Clear cached geometry data."""
        self._local_coordinates = None
        self._local_weights = None

    def parametric_to_reference(self, xi: Tuple[float, ...]) -> Tuple[float, ...]:
        """
        Map parametric coordinates to reference element [0,1]^d.

        Parameters:
            xi: Parametric coordinates

        Returns:
            Reference coordinates in [0,1]^d
        """
        ref = []
        for d in range(self.n_dim):
            xi_min, xi_max = self.parametric_bounds[d]
            t = (xi[d] - xi_min) / (xi_max - xi_min)
            ref.append(t)
        return tuple(ref)

    def reference_to_parametric(self, t: Tuple[float, ...]) -> Tuple[float, ...]:
        """
        Map reference coordinates [0,1]^d to parametric coordinates.

        Parameters:
            t: Reference coordinates in [0,1]^d

        Returns:
            Parametric coordinates
        """
        xi = []
        for d in range(self.n_dim):
            xi_min, xi_max = self.parametric_bounds[d]
            xi.append(xi_min + t[d] * (xi_max - xi_min))
        return tuple(xi)

    def jacobian_ref_to_param(self) -> np.ndarray:
        """
        Compute Jacobian of reference-to-parametric mapping.

        Since the mapping is linear (affine), this is constant on each element.

        Returns:
            Diagonal matrix with (xi_max - xi_min) entries
        """
        jac = np.zeros((self.n_dim, self.n_dim))
        for d in range(self.n_dim):
            xi_min, xi_max = self.parametric_bounds[d]
            jac[d, d] = xi_max - xi_min
        return jac

    def det_jacobian_ref_to_param(self) -> float:
        """
        Compute determinant of reference-to-parametric Jacobian.

        Returns:
            Product of all (xi_max - xi_min) values
        """
        det = 1.0
        for d in range(self.n_dim):
            xi_min, xi_max = self.parametric_bounds[d]
            det *= (xi_max - xi_min)
        return det

    def deactivate(self) -> None:
        """Deactivate this element."""
        self.active = False

    def activate(self) -> None:
        """Activate this element."""
        self.active = True

    def is_leaf(self) -> bool:
        """Check if this element has no children (is a leaf in the hierarchy)."""
        return len(self.children_ids) == 0

    def is_root(self) -> bool:
        """Check if this element has no parent (is at coarsest level)."""
        return self.parent_id is None

    def contains(self, other_bounds: Tuple[Tuple[float, float], ...], tol: float = 1e-10) -> bool:
        """
        Check if this element's parametric domain contains another domain.

        Parameters:
            other_bounds: Parametric bounds to check ((xi_min, xi_max), ...)
            tol: Tolerance for boundary comparison

        Returns:
            True if other_bounds is contained within this element's bounds
        """
        for d in range(self.n_dim):
            self_min, self_max = self.parametric_bounds[d]
            other_min, other_max = other_bounds[d]
            if other_min < self_min - tol or other_max > self_max + tol:
                return False
        return True

    def add_child(self, child_id: int) -> None:
        """Add a child element ID."""
        if child_id not in self.children_ids:
            self.children_ids.append(child_id)

    def remove_child(self, child_id: int) -> None:
        """Remove a child element ID."""
        if child_id in self.children_ids:
            self.children_ids.remove(child_id)

    def __hash__(self) -> int:
        """Hash based on ID for use in sets/dicts."""
        return hash(self.id)

    def __eq__(self, other) -> bool:
        """Equality based on ID."""
        if isinstance(other, Element):
            return self.id == other.id
        return False


# Backward compatibility alias
@dataclass
class ElementWithGeometry(Element):
    """
    Backward compatibility wrapper.

    New code should use Element with set_geometry_cache() instead.
    """

    def __init__(self,
                 element_id: int,
                 parametric_bounds: Tuple[Tuple[float, float], ...],
                 extraction_operator: np.ndarray,
                 global_dof_indices: np.ndarray,
                 degrees: Tuple[int, ...],
                 control_points: np.ndarray,
                 weights: np.ndarray):
        """Create element with geometry (backward compatible signature)."""
        super().__init__(
            id=element_id,
            level=0,
            parametric_bounds=parametric_bounds,
            control_point_ids=list(global_dof_indices),
            extraction_operator=extraction_operator,
            degrees=degrees,
            active=True
        )
        self.set_geometry_cache(control_points, weights)

    @property
    def element_id(self) -> int:
        """Backward compatibility alias for id."""
        return self.id

    @property
    def n_dim_physical(self) -> int:
        """Number of physical/spatial dimensions."""
        if self._local_coordinates is not None:
            return self._local_coordinates.shape[1]
        return 0

    @property
    def n_local_dof(self) -> int:
        """Number of local DOFs (backward compatibility)."""
        return len(self.control_point_ids)


def create_element_2d(
    element_id: int,
    xi_bounds: Tuple[float, float],
    eta_bounds: Tuple[float, float],
    extraction_operator: np.ndarray,
    control_point_ids: List[int],
    degrees: Tuple[int, int],
    level: int = 0
) -> Element:
    """
    Create a 2D element.

    Parameters:
        element_id: Unique element ID
        xi_bounds: (xi_min, xi_max) for this element
        eta_bounds: (eta_min, eta_max) for this element
        extraction_operator: 2D extraction operator (Kronecker product)
        control_point_ids: List of control point IDs for this element
        degrees: (p_xi, p_eta) polynomial degrees
        level: Refinement level (default 0)

    Returns:
        Element instance
    """
    return Element(
        id=element_id,
        level=level,
        parametric_bounds=((xi_bounds[0], xi_bounds[1]),
                          (eta_bounds[0], eta_bounds[1])),
        control_point_ids=control_point_ids,
        extraction_operator=extraction_operator,
        degrees=degrees,
        active=True
    )


def create_element_with_geometry_2d(
    element_id: int,
    xi_bounds: Tuple[float, float],
    eta_bounds: Tuple[float, float],
    extraction_operator: np.ndarray,
    control_point_ids: List[int],
    degrees: Tuple[int, int],
    coordinates: np.ndarray,
    weights: np.ndarray,
    level: int = 0
) -> Element:
    """
    Create a 2D element with cached geometry.

    Parameters:
        element_id: Unique element ID
        xi_bounds, eta_bounds: Parametric bounds
        extraction_operator: Extraction operator
        control_point_ids: List of control point IDs
        degrees: Polynomial degrees
        coordinates: Local control point coordinates
        weights: NURBS weights
        level: Refinement level (default 0)

    Returns:
        Element instance with cached geometry
    """
    elem = create_element_2d(
        element_id, xi_bounds, eta_bounds,
        extraction_operator, control_point_ids,
        degrees, level
    )
    elem.set_geometry_cache(coordinates, weights)
    return elem

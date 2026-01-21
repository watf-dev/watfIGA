"""
Control point abstraction for IGA.

Control points are the fundamental DOFs in isogeometric analysis.
Unlike FEM nodes which only have coordinates, IGA control points have:
- Physical coordinates
- NURBS weight
- Refinement level (for hierarchical splines)
- Active/inactive state (for THB truncation)
- Knowledge of supporting elements (bidirectional linking)

This design enables:
- THB-splines: Control points can be deactivated when children exist
- T-splines: Control points can have irregular support
- Solver: Operates only on active control points

Key invariant (must always hold):
    cp in element.control_point_ids  <=>  element.id in cp.supported_elements
"""

import numpy as np
from typing import Set, Optional
from dataclasses import dataclass, field


@dataclass
class ControlPoint:
    """
    First-class control point object.

    Attributes:
        id: Unique identifier
        coordinates: Physical coordinates (x, y) or (x, y, z)
        weight: NURBS weight (1.0 for B-splines)
        level: Refinement level (0 = coarsest/base level)
        active: Whether this control point contributes to the solution
        supported_elements: Set of element IDs that this CP supports

    Design notes:
        - Control points know which elements they support (bidirectional link)
        - Deactivating a control point (THB truncation) sets active=False
        - Solver DOFs are attached only to active control points
        - The weight is used for NURBS rationalization
    """
    id: int
    coordinates: np.ndarray
    weight: float = 1.0
    level: int = 0
    active: bool = True
    supported_elements: Set[int] = field(default_factory=set)

    def __post_init__(self):
        """Ensure coordinates is a numpy array."""
        self.coordinates = np.asarray(self.coordinates, dtype=np.float64)
        # Ensure supported_elements is a set (in case a list was passed)
        if not isinstance(self.supported_elements, set):
            self.supported_elements = set(self.supported_elements)

    @property
    def n_dim(self) -> int:
        """Number of spatial dimensions."""
        return len(self.coordinates)

    @property
    def x(self) -> float:
        """X coordinate."""
        return self.coordinates[0]

    @property
    def y(self) -> float:
        """Y coordinate."""
        return self.coordinates[1]

    @property
    def z(self) -> Optional[float]:
        """Z coordinate (None for 2D)."""
        return self.coordinates[2] if self.n_dim > 2 else None

    def add_element(self, element_id: int) -> None:
        """
        Register that this control point supports an element.

        Parameters:
            element_id: ID of the element to add
        """
        self.supported_elements.add(element_id)

    def remove_element(self, element_id: int) -> None:
        """
        Unregister an element from this control point's support.

        Parameters:
            element_id: ID of the element to remove
        """
        self.supported_elements.discard(element_id)

    def deactivate(self) -> None:
        """
        Deactivate this control point (THB truncation).

        Note: Does not remove element links - those remain for
        potential reactivation or hierarchy tracking.
        """
        self.active = False

    def activate(self) -> None:
        """Reactivate this control point."""
        self.active = True

    def __hash__(self) -> int:
        """Hash based on ID for use in sets/dicts."""
        return hash(self.id)

    def __eq__(self, other) -> bool:
        """Equality based on ID."""
        if isinstance(other, ControlPoint):
            return self.id == other.id
        return False

    def __repr__(self) -> str:
        return (f"ControlPoint(id={self.id}, coord={self.coordinates}, "
                f"w={self.weight}, level={self.level}, active={self.active}, "
                f"n_elements={len(self.supported_elements)})")


def create_control_points_from_array(
    coordinates: np.ndarray,
    weights: Optional[np.ndarray] = None,
    level: int = 0
) -> dict:
    """
    Create ControlPoint objects from coordinate array.

    Parameters:
        coordinates: Array of shape (n_points, n_dim)
        weights: Optional array of shape (n_points,), defaults to 1.0
        level: Refinement level for all points (default 0)

    Returns:
        Dictionary mapping ID -> ControlPoint
    """
    n_points = coordinates.shape[0]

    if weights is None:
        weights = np.ones(n_points)

    control_points = {}
    for i in range(n_points):
        cp = ControlPoint(
            id=i,
            coordinates=coordinates[i].copy(),
            weight=weights[i],
            level=level,
            active=True,
            supported_elements=set()
        )
        control_points[i] = cp

    return control_points

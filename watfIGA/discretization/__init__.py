"""
Discretization module for IGA.

Provides:
- KnotVector: Knot vector representation
- ControlPoint: First-class control point object
- Element: IGA element with extraction operator
- Mesh: Analysis-ready mesh with active tracking
- Bézier extraction operators
"""

from .knot_vector import KnotVector, make_open_knot_vector
from .control_point import ControlPoint, create_control_points_from_array
from .element import Element, create_element_2d, create_element_with_geometry_2d
from .extraction import (
    compute_extraction_operators_1d,
    compute_extraction_operators_2d,
    BernsteinBasis
)

# Import mesh components separately to avoid circular imports
# Users should import these directly: from watfIGA.discretization.mesh import ...

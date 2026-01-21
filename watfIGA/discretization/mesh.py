"""
Analysis-ready mesh (discretization) for IGA.

The Mesh class is the central data structure that:
1. Owns all elements and control points
2. Tracks active subsets for solver
3. Maintains bidirectional Element ↔ ControlPoint linking
4. Provides filtered views for assembly

Key design principles:
- Elements and control points stored as dictionaries by ID
- Active subsets tracked explicitly (not derived)
- Bidirectional linking invariant always maintained
- Solver-agnostic: works with NURBS, THB-splines, T-splines

Bidirectional linking invariant:
    cp_id in element.control_point_ids  <=>  element.id in control_points[cp_id].supported_elements

This design enables:
- THB-splines: Deactivate parent elements/CPs when children exist
- T-splines: Non-tensor-product connectivity
- Solver: Iterate over active elements only
"""
from __future__ import annotations

import numpy as np
from typing import List, Dict, Set, Tuple, Optional, Iterator, TYPE_CHECKING
from dataclasses import dataclass, field

from .knot_vector import KnotVector
from .element import Element, create_element_with_geometry_2d
from .control_point import ControlPoint, create_control_points_from_array
from .extraction import compute_extraction_operators_1d

if TYPE_CHECKING:
    from ..geometry.nurbs import NURBSSurface, NURBSCurve
    from ..geometry.thb import THBSurface


class Mesh:
    """
    Analysis-ready mesh for IGA with explicit active tracking.

    Attributes:
        elements: Dictionary mapping element ID -> Element
        control_points: Dictionary mapping CP ID -> ControlPoint
        active_elements: Set of active element IDs
        active_control_points: Set of active control point IDs
        max_level: Maximum refinement level in the mesh
        n_dim_parametric: Number of parametric dimensions
        n_dim_physical: Number of physical dimensions

    Key invariant:
        The bidirectional linking between elements and control points
        is always consistent. Any modification must update both sides.
    """

    def __init__(self,
                 elements: Dict[int, Element],
                 control_points: Dict[int, ControlPoint],
                 n_dim_parametric: int,
                 n_dim_physical: int,
                 n_control_points_per_dir: Optional[Tuple[int, ...]] = None):
        """
        Initialize mesh with elements and control points.

        Parameters:
            elements: Dictionary of elements by ID
            control_points: Dictionary of control points by ID
            n_dim_parametric: Number of parametric dimensions
            n_dim_physical: Number of physical dimensions
            n_control_points_per_dir: Grid dimensions for tensor-product meshes (n_xi, n_eta, ...)
        """
        self._elements = elements
        self._control_points = control_points
        self.n_dim_parametric = n_dim_parametric
        self.n_dim_physical = n_dim_physical
        self._n_control_points_per_dir = n_control_points_per_dir

        # Initialize active sets from element/CP active flags
        self._active_elements: Set[int] = {
            eid for eid, elem in elements.items() if elem.active
        }
        self._active_control_points: Set[int] = {
            cpid for cpid, cp in control_points.items() if cp.active
        }

        # Compute max level
        self._max_level = max((e.level for e in elements.values()), default=0)

        # Verify bidirectional linking invariant
        self._verify_linking_invariant()

    @classmethod
    def build(cls, geometry, include_geometry: bool = True) -> 'Mesh':
        """
        Build a mesh from NURBS or THB geometry.

        Automatically detects geometry type and dimensionality:
        - NURBSCurve (1D) -> build_mesh_1d
        - NURBSSurface (2D) -> build_mesh_2d
        - THBSurface (2D hierarchical) -> build_thb_mesh

        Parameters:
            geometry: NURBSCurve, NURBSSurface, or THBSurface
            include_geometry: Whether to cache geometry data in elements

        Returns:
            Mesh ready for analysis

        Example:
            surface = make_nurbs_unit_square(p=2, n_elem_xi=4, n_elem_eta=4)
            mesh = Mesh.build(surface)

            curve = NURBSCurve(...)
            mesh = Mesh.build(curve)

            thb_surface = THBSurface.from_nurbs_surface(surface)
            mesh = Mesh.build(thb_surface)
        """
        # Check for THB surface (has hierarchy attribute)
        if hasattr(geometry, 'hierarchy'):
            return build_thb_mesh(geometry, include_geometry)

        # Check dimensionality by parametric dimension
        if geometry.n_dim_parametric == 1:
            return build_mesh_1d(geometry, include_geometry)
        elif geometry.n_dim_parametric == 2:
            return build_mesh_2d(geometry, include_geometry)
        else:
            raise ValueError(f"Unsupported parametric dimension: {geometry.n_dim_parametric}")

    def _verify_linking_invariant(self) -> None:
        """
        Verify the bidirectional linking invariant.

        Raises AssertionError if invariant is violated.
        """
        for eid, elem in self._elements.items():
            for cp_id in elem.control_point_ids:
                if cp_id not in self._control_points:
                    raise ValueError(
                        f"Element {eid} references non-existent control point {cp_id}"
                    )
                if eid not in self._control_points[cp_id].supported_elements:
                    raise ValueError(
                        f"Bidirectional linking violated: "
                        f"CP {cp_id} not linked back to element {eid}"
                    )

        for cp_id, cp in self._control_points.items():
            for eid in cp.supported_elements:
                if eid not in self._elements:
                    raise ValueError(
                        f"Control point {cp_id} references non-existent element {eid}"
                    )
                if cp_id not in self._elements[eid].control_point_ids:
                    raise ValueError(
                        f"Bidirectional linking violated: "
                        f"Element {eid} not linked back to CP {cp_id}"
                    )

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def elements(self) -> Dict[int, Element]:
        """All elements (active and inactive)."""
        return self._elements

    @property
    def control_points(self) -> Dict[int, ControlPoint]:
        """All control points (active and inactive)."""
        return self._control_points

    @property
    def active_elements(self) -> Set[int]:
        """Set of active element IDs."""
        return self._active_elements.copy()

    @property
    def active_control_points(self) -> Set[int]:
        """Set of active control point IDs."""
        return self._active_control_points.copy()

    @property
    def n_elements(self) -> int:
        """Total number of elements (active and inactive)."""
        return len(self._elements)

    @property
    def n_active_elements(self) -> int:
        """Number of active elements."""
        return len(self._active_elements)

    @property
    def n_control_points(self) -> int:
        """Total number of control points (active and inactive)."""
        return len(self._control_points)

    @property
    def n_active_control_points(self) -> int:
        """Number of active control points."""
        return len(self._active_control_points)

    @property
    def n_dof(self) -> int:
        """Number of DOFs (equals number of active control points for scalar problems)."""
        return self.n_active_control_points

    @property
    def max_level(self) -> int:
        """Maximum refinement level in the mesh."""
        return self._max_level

    @property
    def n_control_points_per_dir(self) -> Optional[Tuple[int, ...]]:
        """Grid dimensions for tensor-product meshes (n_xi, n_eta, ...)."""
        return self._n_control_points_per_dir

    # -------------------------------------------------------------------------
    # Accessors for solver
    # -------------------------------------------------------------------------

    def get_active_elements(self) -> Iterator[Element]:
        """
        Iterate over active elements.

        This is the primary method for solver assembly.

        Yields:
            Active Element objects
        """
        for eid in self._active_elements:
            yield self._elements[eid]

    def get_active_elements_list(self) -> List[Element]:
        """
        Get list of active elements.

        Returns:
            List of active Element objects
        """
        return [self._elements[eid] for eid in sorted(self._active_elements)]

    def get_active_control_points(self) -> Iterator[ControlPoint]:
        """
        Iterate over active control points.

        Yields:
            Active ControlPoint objects
        """
        for cpid in self._active_control_points:
            yield self._control_points[cpid]

    def get_active_control_points_list(self) -> List[ControlPoint]:
        """
        Get list of active control points.

        Returns:
            List of active ControlPoint objects
        """
        return [self._control_points[cpid] for cpid in sorted(self._active_control_points)]

    def get_element(self, element_id: int) -> Element:
        """Get element by ID."""
        return self._elements[element_id]


    def get_control_point(self, cp_id: int) -> ControlPoint:
        """Get control point by ID."""
        return self._control_points[cp_id]

    def get_control_point_coordinates(self, cp_ids: List[int]) -> np.ndarray:
        """
        Get coordinates for a list of control point IDs.

        Parameters:
            cp_ids: List of control point IDs

        Returns:
            Array of coordinates, shape (len(cp_ids), n_dim_physical)
        """
        return np.array([self._control_points[cpid].coordinates for cpid in cp_ids])

    def get_control_point_weights(self, cp_ids: List[int]) -> np.ndarray:
        """
        Get weights for a list of control point IDs.

        Parameters:
            cp_ids: List of control point IDs

        Returns:
            Array of weights, shape (len(cp_ids),)
        """
        return np.array([self._control_points[cpid].weight for cpid in cp_ids])

    # -------------------------------------------------------------------------
    # Active management (for THB-splines)
    # -------------------------------------------------------------------------

    def deactivate_element(self, element_id: int) -> None:
        """
        Deactivate an element.

        Parameters:
            element_id: ID of element to deactivate
        """
        if element_id in self._elements:
            self._elements[element_id].deactivate()
            self._active_elements.discard(element_id)

    def activate_element(self, element_id: int) -> None:
        """
        Activate an element.

        Parameters:
            element_id: ID of element to activate
        """
        if element_id in self._elements:
            self._elements[element_id].activate()
            self._active_elements.add(element_id)

    def deactivate_control_point(self, cp_id: int) -> None:
        """
        Deactivate a control point (THB truncation).

        Parameters:
            cp_id: ID of control point to deactivate
        """
        if cp_id in self._control_points:
            self._control_points[cp_id].deactivate()
            self._active_control_points.discard(cp_id)

    def activate_control_point(self, cp_id: int) -> None:
        """
        Activate a control point.

        Parameters:
            cp_id: ID of control point to activate
        """
        if cp_id in self._control_points:
            self._control_points[cp_id].activate()
            self._active_control_points.add(cp_id)

    def get_elements_at_level(self, level: int) -> List[Element]:
        """
        Get all elements at a specific refinement level.

        Parameters:
            level: Refinement level

        Returns:
            List of elements at that level
        """
        return [e for e in self._elements.values() if e.level == level]

    def get_control_points_at_level(self, level: int) -> List[ControlPoint]:
        """
        Get all control points at a specific refinement level.

        Parameters:
            level: Refinement level

        Returns:
            List of control points at that level
        """
        return [cp for cp in self._control_points.values() if cp.level == level]

    # -------------------------------------------------------------------------
    # Linking management
    # -------------------------------------------------------------------------

    def link_element_to_control_point(self, element_id: int, cp_id: int) -> None:
        """
        Create bidirectional link between element and control point.

        Parameters:
            element_id: Element ID
            cp_id: Control point ID
        """
        if cp_id not in self._elements[element_id].control_point_ids:
            self._elements[element_id].control_point_ids.append(cp_id)
        self._control_points[cp_id].add_element(element_id)

    def unlink_element_from_control_point(self, element_id: int, cp_id: int) -> None:
        """
        Remove bidirectional link between element and control point.

        Parameters:
            element_id: Element ID
            cp_id: Control point ID
        """
        elem = self._elements[element_id]
        if cp_id in elem.control_point_ids:
            elem.control_point_ids.remove(cp_id)
        self._control_points[cp_id].remove_element(element_id)

    # -------------------------------------------------------------------------
    # Connectivity queries
    # -------------------------------------------------------------------------

    def get_element_connectivity(self, element_id: int) -> List[int]:
        """
        Get the control point IDs for an element (connectivity data).

        Parameters:
            element_id: Element ID

        Returns:
            List of control point IDs in local ordering
        """
        return self._elements[element_id].control_point_ids.copy()

    def get_control_point_support(self, cp_id: int) -> Set[int]:
        """
        Get the element IDs that a control point supports.

        Parameters:
            cp_id: Control point ID

        Returns:
            Set of element IDs where this CP's basis function is non-zero
        """
        return self._control_points[cp_id].supported_elements.copy()

    def get_connectivity_matrix(self) -> np.ndarray:
        """
        Get the full connectivity matrix (element to control point).

        Returns:
            Boolean matrix C where C[e, cp] = True if CP contributes to element e.
            Shape: (n_elements, n_control_points)

        Note:
            This is mainly for visualization/debugging. For assembly,
            use element.control_point_ids directly.
        """
        n_elem = self.n_elements
        n_cp = self.n_control_points
        connectivity = np.zeros((n_elem, n_cp), dtype=bool)

        for elem_id, elem in self._elements.items():
            for cp_id in elem.control_point_ids:
                connectivity[elem_id, cp_id] = True

        return connectivity

    def get_active_connectivity_matrix(self) -> np.ndarray:
        """
        Get connectivity matrix for active elements and control points only.

        Returns:
            Boolean matrix C where C[e, cp] = True if active CP contributes to active element.
            Shape: (n_active_elements, n_active_control_points)

        Note:
            Row/column indices correspond to sorted active element/CP IDs.
        """
        active_elem_ids = sorted(self._active_elements)
        active_cp_ids = sorted(self._active_control_points)

        n_elem = len(active_elem_ids)
        n_cp = len(active_cp_ids)

        # Create mapping from global ID to local index
        elem_to_local = {eid: i for i, eid in enumerate(active_elem_ids)}
        cp_to_local = {cpid: j for j, cpid in enumerate(active_cp_ids)}

        connectivity = np.zeros((n_elem, n_cp), dtype=bool)

        for elem_id in active_elem_ids:
            elem = self._elements[elem_id]
            local_elem_idx = elem_to_local[elem_id]
            for cp_id in elem.control_point_ids:
                if cp_id in cp_to_local:
                    local_cp_idx = cp_to_local[cp_id]
                    connectivity[local_elem_idx, local_cp_idx] = True

        return connectivity

    def print_connectivity_summary(self) -> str:
        """
        Generate a human-readable connectivity summary.

        Returns:
            String summary of element-control point connectivity
        """
        lines = []
        lines.append("=" * 60)
        lines.append("MESH CONNECTIVITY SUMMARY")
        lines.append("=" * 60)
        lines.append(f"Total elements: {self.n_elements} (active: {self.n_active_elements})")
        lines.append(f"Total control points: {self.n_control_points} (active: {self.n_active_control_points})")
        lines.append(f"Max refinement level: {self.max_level}")
        lines.append("")

        lines.append("Element Connectivity:")
        lines.append("-" * 40)
        for elem_id in sorted(self._elements.keys()):
            elem = self._elements[elem_id]
            status = "active" if elem.active else "INACTIVE"
            cp_ids = elem.control_point_ids
            lines.append(f"  E[{elem_id}] (level={elem.level}, {status}): "
                        f"{len(cp_ids)} CPs -> {cp_ids[:5]}{'...' if len(cp_ids) > 5 else ''}")
        lines.append("")

        lines.append("Control Point Support:")
        lines.append("-" * 40)
        for cp_id in sorted(self._control_points.keys()):
            cp = self._control_points[cp_id]
            status = "active" if cp.active else "INACTIVE"
            elem_ids = sorted(cp.supported_elements)
            lines.append(f"  CP[{cp_id}] (level={cp.level}, {status}): "
                        f"{len(elem_ids)} elements -> {elem_ids}")
        lines.append("=" * 60)

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Backward compatibility
    # -------------------------------------------------------------------------

    def get_elements_for_solver(self) -> List[Element]:
        """
        Get elements for solver (backward compatible).

        Returns:
            List of active elements
        """
        return self.get_active_elements_list()

    @property
    def control_points_array(self) -> np.ndarray:
        """
        Get control point coordinates as array (for backward compatibility).

        Returns:
            Array of shape (n_control_points, n_dim_physical)
        """
        n_cp = len(self._control_points)
        coords = np.zeros((n_cp, self.n_dim_physical))
        for cpid, cp in self._control_points.items():
            coords[cpid] = cp.coordinates
        return coords

    @property
    def weights_array(self) -> np.ndarray:
        """
        Get weights as array (for backward compatibility).

        Returns:
            Array of shape (n_control_points,)
        """
        n_cp = len(self._control_points)
        weights = np.ones(n_cp)
        for cpid, cp in self._control_points.items():
            weights[cpid] = cp.weight
        return weights


# Backward compatibility alias
class MeshWithGeometry(Mesh):
    """Backward compatibility alias for Mesh."""

    def __init__(self,
                 elements: List[Element],
                 n_dof: int,
                 n_dim_parametric: int,
                 n_dim_physical: int,
                 control_points: np.ndarray,
                 weights: np.ndarray):
        """
        Create mesh from old-style arguments.

        Parameters:
            elements: List of elements
            n_dof: Number of DOFs (ignored, computed from control points)
            n_dim_parametric: Number of parametric dimensions
            n_dim_physical: Number of physical dimensions
            control_points: Array of control point coordinates
            weights: Array of weights
        """
        # Convert elements list to dict
        elements_dict = {elem.id: elem for elem in elements}

        # Create control points from arrays
        cp_dict = create_control_points_from_array(control_points, weights)

        # Populate bidirectional links
        for elem in elements:
            for cp_id in elem.control_point_ids:
                cp_dict[cp_id].add_element(elem.id)

        super().__init__(
            elements=elements_dict,
            control_points=cp_dict,
            n_dim_parametric=n_dim_parametric,
            n_dim_physical=n_dim_physical
        )

        # Store original arrays for backward compatibility
        self._control_points_array = control_points
        self._weights_array = weights

    @property
    def control_points_array(self) -> np.ndarray:
        """Original control points array."""
        return self._control_points_array

    @property
    def weights_array(self) -> np.ndarray:
        """Original weights array."""
        return self._weights_array


def build_mesh_1d(curve: 'NURBSCurve',
                  include_geometry: bool = True) -> Mesh:
    """
    Build a 1D analysis mesh from a NURBS curve.

    Parameters:
        curve: NURBSCurve geometry
        include_geometry: Whether to cache geometry data in elements

    Returns:
        Mesh ready for analysis
    """
    kv = curve.knot_vector
    p = curve.degree
    n_basis = curve.n_control_points

    # Get element bounds
    elements = kv.elements
    n_elem = len(elements)

    # Compute extraction operators
    C_list = compute_extraction_operators_1d(kv)

    # Create control points from curve data
    control_points_global = curve.control_points
    weights_global = curve.weights
    cp_dict = create_control_points_from_array(control_points_global, weights_global, level=0)

    # Build elements with bidirectional links
    elements_dict: Dict[int, Element] = {}

    for element_id in range(n_elem):
        xi_min, xi_max = elements[element_id]
        C_e = C_list[element_id]

        # Active basis indices
        span = kv.element_to_span(element_id)
        active = np.arange(span - p, span + 1)

        # Build control point IDs
        control_point_ids = [int(i) for i in active]
        for cp_id in control_point_ids:
            cp_dict[cp_id].add_element(element_id)

        # Create element
        elem = Element(
            id=element_id,
            level=0,
            parametric_bounds=((xi_min, xi_max),),
            control_point_ids=control_point_ids,
            extraction_operator=C_e,
            degrees=(p,),
            active=True
        )

        # Cache geometry if requested
        if include_geometry:
            local_coords = control_points_global[control_point_ids]
            local_weights = weights_global[control_point_ids]
            elem.set_geometry_cache(local_coords, local_weights)

        elements_dict[element_id] = elem

    return Mesh(
        elements=elements_dict,
        control_points=cp_dict,
        n_dim_parametric=1,
        n_dim_physical=curve.n_dim_physical,
        n_control_points_per_dir=(n_basis,)
    )


def build_mesh_2d(surface: NURBSSurface,
                  include_geometry: bool = True) -> Mesh:
    """
    Build a 2D analysis mesh from a NURBS surface.

    This is the main entry point for creating an IGA mesh from geometry.
    It computes:
    1. Control points with bidirectional element links
    2. Elements with control point references
    3. Bézier extraction operators
    4. Local geometry caching for each element

    Parameters:
        surface: NURBSSurface geometry
        include_geometry: Whether to cache geometry data in elements

    Returns:
        Mesh ready for analysis
    """
    kv_xi, kv_eta = surface.knot_vectors
    p_xi, p_eta = surface.degrees

    n_basis_xi, n_basis_eta = surface.n_control_points_per_dir

    # Get element bounds
    elements_xi = kv_xi.elements
    elements_eta = kv_eta.elements
    n_elem_xi = len(elements_xi)
    n_elem_eta = len(elements_eta)

    # Compute extraction operators
    C_xi_list = compute_extraction_operators_1d(kv_xi)
    C_eta_list = compute_extraction_operators_1d(kv_eta)

    # Create control points from surface data
    control_points_global = surface.control_points
    weights_global = surface.weights
    cp_dict = create_control_points_from_array(control_points_global, weights_global, level=0)

    # Build elements with bidirectional links
    elements_dict: Dict[int, Element] = {}
    element_id = 0

    for ej in range(n_elem_eta):
        eta_min, eta_max = elements_eta[ej]
        C_eta = C_eta_list[ej]

        # Active basis indices in eta direction
        span_eta = kv_eta.element_to_span(ej)
        active_eta = np.arange(span_eta - p_eta, span_eta + 1)

        for ei in range(n_elem_xi):
            xi_min, xi_max = elements_xi[ei]
            C_xi = C_xi_list[ei]

            # Active basis indices in xi direction
            span_xi = kv_xi.element_to_span(ei)
            active_xi = np.arange(span_xi - p_xi, span_xi + 1)

            # 2D extraction operator (Kronecker product)
            C_e = np.kron(C_eta, C_xi)

            # Build control point IDs in tensor-product order
            control_point_ids = []
            for j in active_eta:
                for i in active_xi:
                    cp_id = int(j * n_basis_xi + i)
                    control_point_ids.append(cp_id)
                    # Create bidirectional link
                    cp_dict[cp_id].add_element(element_id)

            # Create element
            elem = Element(
                id=element_id,
                level=0,
                parametric_bounds=((xi_min, xi_max), (eta_min, eta_max)),
                control_point_ids=control_point_ids,
                extraction_operator=C_e,
                degrees=(p_xi, p_eta),
                active=True
            )

            # Cache geometry if requested
            if include_geometry:
                local_coords = control_points_global[control_point_ids]
                local_weights = weights_global[control_point_ids]
                elem.set_geometry_cache(local_coords, local_weights)

            elements_dict[element_id] = elem
            element_id += 1

    return Mesh(
        elements=elements_dict,
        control_points=cp_dict,
        n_dim_parametric=2,
        n_dim_physical=surface.n_dim_physical,
        n_control_points_per_dir=(n_basis_xi, n_basis_eta)
    )


def build_thb_mesh(thb_surface: 'THBSurface',
                   include_geometry: bool = True) -> Mesh:
    """
    Build a mesh from a THB-spline surface with hierarchical refinement.

    For each element:
    - If the element is NOT refined: use the coarse-level element
    - If the element IS refined: use the fine-level child elements

    The extraction operators encode the THB basis, allowing the solver
    to work unchanged.

    Parameters:
        thb_surface: THBSurface with refinement information
        include_geometry: Whether to cache geometry data in elements

    Returns:
        Mesh ready for analysis
    """
    # Finalize refinement to update active basis sets
    thb_surface.finalize_refinement()

    hierarchy = thb_surface.hierarchy
    p_xi, p_eta = thb_surface.degrees

    # We need to build a unified global DOF numbering for all active basis functions
    # across all levels. This is the key to THB-spline mesh construction.

    # Build global DOF map: (level, local_idx) -> global_dof
    global_dof_map: Dict[Tuple[int, int], int] = {}
    global_dof = 0

    for level in range(thb_surface.n_levels):
        active_cps = thb_surface.get_active_control_points(level)
        for local_idx in sorted(active_cps):
            global_dof_map[(level, local_idx)] = global_dof
            global_dof += 1

    n_total_dofs = global_dof

    # Create control points dictionary
    cp_dict: Dict[int, ControlPoint] = {}

    for level in range(thb_surface.n_levels):
        cp_coords = thb_surface.get_control_points(level)
        cp_weights = thb_surface.get_weights(level)
        active_cps = thb_surface.get_active_control_points(level)

        for local_idx in active_cps:
            global_idx = global_dof_map[(level, local_idx)]
            cp_dict[global_idx] = ControlPoint(
                id=global_idx,
                coordinates=cp_coords[local_idx].copy(),
                weight=cp_weights[local_idx],
                level=level,
                active=True
            )

    # Build elements
    elements_dict: Dict[int, Element] = {}
    element_id = 0

    # Process level 0 elements
    kv_xi_0 = hierarchy.hierarchy_xi.get_knot_vector(0)
    kv_eta_0 = hierarchy.hierarchy_eta.get_knot_vector(0)
    n_elem_xi_0 = kv_xi_0.n_elements
    n_elem_eta_0 = kv_eta_0.n_elements

    # Compute extraction operators for level 0
    C_xi_list_0 = compute_extraction_operators_1d(kv_xi_0)
    C_eta_list_0 = compute_extraction_operators_1d(kv_eta_0)

    # Compute extraction operators for level 1 if it exists
    C_xi_list_1 = None
    C_eta_list_1 = None
    if thb_surface.n_levels > 1:
        kv_xi_1 = hierarchy.hierarchy_xi.get_knot_vector(1)
        kv_eta_1 = hierarchy.hierarchy_eta.get_knot_vector(1)
        C_xi_list_1 = compute_extraction_operators_1d(kv_xi_1)
        C_eta_list_1 = compute_extraction_operators_1d(kv_eta_1)

    for ej in range(n_elem_eta_0):
        for ei in range(n_elem_xi_0):
            if thb_surface.is_element_refined(0, ei, ej):
                # Element is refined - create child elements at level 1
                child_elements = _create_thb_child_elements(
                    thb_surface, ei, ej, element_id, global_dof_map,
                    C_xi_list_1, C_eta_list_1, cp_dict, include_geometry
                )
                for child_elem in child_elements:
                    elements_dict[child_elem.id] = child_elem
                    element_id += 1
            else:
                # Element not refined - create level 0 element
                elem = _create_thb_element_level0(
                    thb_surface, ei, ej, element_id, global_dof_map,
                    C_xi_list_0, C_eta_list_0, cp_dict, include_geometry
                )
                elements_dict[element_id] = elem
                element_id += 1

    # Update bidirectional links
    for elem in elements_dict.values():
        for cp_id in elem.control_point_ids:
            if cp_id in cp_dict:
                cp_dict[cp_id].add_element(elem.id)

    return Mesh(
        elements=elements_dict,
        control_points=cp_dict,
        n_dim_parametric=2,
        n_dim_physical=thb_surface.n_dim_physical,
        n_control_points_per_dir=None  # Not applicable for THB
    )


def _create_thb_element_level0(
    thb_surface: 'THBSurface',
    ei: int,
    ej: int,
    element_id: int,
    global_dof_map: Dict[Tuple[int, int], int],
    C_xi_list: List[np.ndarray],
    C_eta_list: List[np.ndarray],
    cp_dict: Dict[int, ControlPoint],
    include_geometry: bool
) -> Element:
    """Create a level-0 THB element (not refined)."""
    hierarchy = thb_surface.hierarchy
    p_xi, p_eta = thb_surface.degrees

    kv_xi = hierarchy.hierarchy_xi.get_knot_vector(0)
    kv_eta = hierarchy.hierarchy_eta.get_knot_vector(0)
    n_xi = kv_xi.n_basis

    xi_bounds = kv_xi.elements[ei]
    eta_bounds = kv_eta.elements[ej]

    # Standard extraction operator
    C_xi = C_xi_list[ei]
    C_eta = C_eta_list[ej]
    C_e = np.kron(C_eta, C_xi)

    # Get active basis functions for this element
    span_xi = kv_xi.element_to_span(ei)
    span_eta = kv_eta.element_to_span(ej)
    active_xi = range(span_xi - p_xi, span_xi + 1)
    active_eta = range(span_eta - p_eta, span_eta + 1)

    # Map local basis to global DOFs
    control_point_ids = []
    active_local_indices = []

    for j in active_eta:
        for i in active_xi:
            local_idx = j * n_xi + i
            # Check if this basis is active at level 0
            if (0, local_idx) in global_dof_map:
                control_point_ids.append(global_dof_map[(0, local_idx)])
                active_local_indices.append(len(control_point_ids) - 1)

    # Adjust extraction operator if some basis functions are deactivated
    # (truncated to finer level)
    if len(control_point_ids) < C_e.shape[0]:
        # Need to modify extraction operator for THB truncation
        # For now, use the rows corresponding to active basis functions
        C_e_thb = C_e[active_local_indices, :]
    else:
        C_e_thb = C_e

    elem = Element(
        id=element_id,
        level=0,
        parametric_bounds=(xi_bounds, eta_bounds),
        control_point_ids=control_point_ids,
        extraction_operator=C_e_thb,
        degrees=(p_xi, p_eta),
        active=True
    )

    if include_geometry and control_point_ids:
        local_coords = np.array([cp_dict[cpid].coordinates for cpid in control_point_ids])
        local_weights = np.array([cp_dict[cpid].weight for cpid in control_point_ids])
        elem.set_geometry_cache(local_coords, local_weights)

    return elem


def _create_thb_child_elements(
    thb_surface: 'THBSurface',
    ei_coarse: int,
    ej_coarse: int,
    start_element_id: int,
    global_dof_map: Dict[Tuple[int, int], int],
    C_xi_list: List[np.ndarray],
    C_eta_list: List[np.ndarray],
    cp_dict: Dict[int, ControlPoint],
    include_geometry: bool
) -> List[Element]:
    """
    Create level-1 child elements for a refined coarse element.

    For elements inside the refined region, the active basis functions are:
    1. Interior L1 basis functions (support entirely in refined region)
    2. Truncated L0 basis functions (support overlaps this element)

    The extraction operator maps from active global DOFs to local Bernstein basis.
    For truncated L0 functions, the extraction accounts for the truncation formula:
        trunc(N^L0) = N^L0 - sum_k c_k * N_k^L1
    """
    hierarchy = thb_surface.hierarchy
    p_xi, p_eta = thb_surface.degrees
    n_local = (p_xi + 1) * (p_eta + 1)

    kv_xi_0 = hierarchy.hierarchy_xi.get_knot_vector(0)
    kv_eta_0 = hierarchy.hierarchy_eta.get_knot_vector(0)
    n_xi_0 = kv_xi_0.n_basis

    kv_xi_1 = hierarchy.hierarchy_xi.get_knot_vector(1)
    kv_eta_1 = hierarchy.hierarchy_eta.get_knot_vector(1)
    n_xi_1 = kv_xi_1.n_basis

    # Get truncated L0 basis functions
    truncated_l0 = thb_surface.get_truncated_basis(0)

    # Each coarse element splits into 2x2 = 4 fine elements
    fine_elem_xi_start = 2 * ei_coarse
    fine_elem_eta_start = 2 * ej_coarse

    elements = []
    elem_id = start_element_id

    for dej in range(2):
        fej = fine_elem_eta_start + dej
        for dei in range(2):
            fei = fine_elem_xi_start + dei

            xi_bounds = kv_xi_1.elements[fei]
            eta_bounds = kv_eta_1.elements[fej]

            # Full extraction operator for fine element (L1 basis)
            C_xi = C_xi_list[fei]
            C_eta = C_eta_list[fej]
            C_e_l1_full = np.kron(C_eta, C_xi)

            # Active L1 basis functions for this element
            span_xi = kv_xi_1.element_to_span(fei)
            span_eta = kv_eta_1.element_to_span(fej)
            active_xi_l1 = list(range(span_xi - p_xi, span_xi + 1))
            active_eta_l1 = list(range(span_eta - p_eta, span_eta + 1))

            # Collect L1 DOFs
            control_point_ids = []
            l1_local_indices = []
            local_idx_counter = 0

            for j in active_eta_l1:
                for i in active_xi_l1:
                    local_idx = j * n_xi_1 + i
                    if (1, local_idx) in global_dof_map:
                        control_point_ids.append(global_dof_map[(1, local_idx)])
                        l1_local_indices.append(local_idx_counter)
                    local_idx_counter += 1

            # Find truncated L0 basis functions that overlap this fine element
            l0_dofs_for_element = []
            for l0_idx in truncated_l0:
                i_l0 = l0_idx % n_xi_0
                j_l0 = l0_idx // n_xi_0

                # Get support of this L0 basis function
                xi_supp = hierarchy.hierarchy_xi.get_basis_support(0, i_l0)
                eta_supp = hierarchy.hierarchy_eta.get_basis_support(0, j_l0)

                # Check if support overlaps with this fine element
                # Element bounds: [xi_bounds[0], xi_bounds[1]] x [eta_bounds[0], eta_bounds[1]]
                xi_overlap = (xi_supp[0] < xi_bounds[1]) and (xi_supp[1] > xi_bounds[0])
                eta_overlap = (eta_supp[0] < eta_bounds[1]) and (eta_supp[1] > eta_bounds[0])

                if xi_overlap and eta_overlap:
                    if (0, l0_idx) in global_dof_map:
                        l0_dofs_for_element.append((l0_idx, global_dof_map[(0, l0_idx)]))

            # Add L0 DOFs to control_point_ids
            for l0_idx, global_dof in l0_dofs_for_element:
                control_point_ids.append(global_dof)

            # Build extraction operator
            # For now, we build a combined extraction operator:
            # - First columns: L1 extraction (subset of C_e_l1_full)
            # - Additional columns for truncated L0: identity-like mapping
            #   (actual truncation is handled during basis evaluation)
            n_l1_dofs = len(l1_local_indices)
            n_l0_dofs = len(l0_dofs_for_element)

            if l1_local_indices:
                C_e_l1 = C_e_l1_full[:, l1_local_indices]
            else:
                C_e_l1 = np.zeros((n_local, 0))

            # For truncated L0, we need to compute the extraction on the fine element
            # This is a simplified approach - for full THB evaluation, the solver
            # should use the THB basis evaluation with truncation
            if n_l0_dofs > 0:
                # Placeholder extraction for L0 DOFs on fine element
                # The actual values should be computed based on L0 basis restriction
                # to the fine element. For now, use identity to indicate these DOFs exist.
                C_e_l0 = np.zeros((n_local, n_l0_dofs))
                # Mark that L0 DOFs contribute (diagonal-like for identification)
                for k in range(min(n_local, n_l0_dofs)):
                    C_e_l0[k, k] = 1.0
            else:
                C_e_l0 = np.zeros((n_local, 0))

            # Combined extraction operator
            C_e = np.hstack([C_e_l1, C_e_l0]) if n_l0_dofs > 0 else C_e_l1

            elem = Element(
                id=elem_id,
                level=1,
                parametric_bounds=(xi_bounds, eta_bounds),
                control_point_ids=control_point_ids,
                extraction_operator=C_e,
                degrees=(p_xi, p_eta),
                active=True
            )

            if include_geometry and control_point_ids:
                local_coords = np.array([cp_dict[cpid].coordinates for cpid in control_point_ids])
                local_weights = np.array([cp_dict[cpid].weight for cpid in control_point_ids])
                elem.set_geometry_cache(local_coords, local_weights)

            elements.append(elem)
            elem_id += 1

    return elements


def get_boundary_dofs_2d(surface: NURBSSurface,
                          boundary: str) -> np.ndarray:
    """
    Get global DOF indices on a boundary.

    For a tensor-product surface with control points indexed as
    P[i,j] -> global index j * n_xi + i (x-fastest), the boundaries are:

    - "left": xi = 0, i.e., i=0, j varies
    - "right": xi = 1, i.e., i=n_xi-1, j varies
    - "bottom": eta = 0, i.e., j=0, i varies
    - "top": eta = 1, i.e., j=n_eta-1, i varies

    Parameters:
        surface: NURBS surface
        boundary: One of "left", "right", "bottom", "top"

    Returns:
        Array of global DOF indices on the boundary
    """
    n_xi, n_eta = surface.n_control_points_per_dir

    if boundary == "left":
        return np.array([j * n_xi + 0 for j in range(n_eta)])
    elif boundary == "right":
        return np.array([j * n_xi + (n_xi - 1) for j in range(n_eta)])
    elif boundary == "bottom":
        return np.array([0 * n_xi + i for i in range(n_xi)])
    elif boundary == "top":
        return np.array([(n_eta - 1) * n_xi + i for i in range(n_xi)])
    else:
        raise ValueError(f"Unknown boundary: {boundary}. "
                        f"Use 'left', 'right', 'bottom', or 'top'.")


def get_all_boundary_dofs_2d(surface: NURBSSurface) -> np.ndarray:
    """
    Get all DOF indices on the boundary (union of all four sides).

    Parameters:
        surface: NURBS surface

    Returns:
        Array of unique global DOF indices on any boundary
    """
    left = get_boundary_dofs_2d(surface, "left")
    right = get_boundary_dofs_2d(surface, "right")
    bottom = get_boundary_dofs_2d(surface, "bottom")
    top = get_boundary_dofs_2d(surface, "top")

    return np.unique(np.concatenate([left, right, bottom, top]))


def get_interior_dofs_2d(surface: NURBSSurface) -> np.ndarray:
    """
    Get all interior (non-boundary) DOF indices.

    Parameters:
        surface: NURBS surface

    Returns:
        Array of global DOF indices not on any boundary
    """
    all_dofs = np.arange(surface.n_control_points)
    boundary_dofs = get_all_boundary_dofs_2d(surface)
    return np.setdiff1d(all_dofs, boundary_dofs)


class BoundaryCondition:
    """Base class for boundary conditions."""
    pass


@dataclass
class DirichletBC(BoundaryCondition):
    """
    Dirichlet boundary condition: u = g on boundary.

    Attributes:
        dof_indices: Global DOF indices where BC is applied
        values: Prescribed values at those DOFs
    """
    dof_indices: np.ndarray
    values: np.ndarray

    @classmethod
    def homogeneous(cls, dof_indices: np.ndarray) -> 'DirichletBC':
        """Create homogeneous Dirichlet BC (u = 0)."""
        return cls(dof_indices, np.zeros(len(dof_indices)))

    @classmethod
    def from_function(cls, surface: NURBSSurface,
                      boundary: str,
                      func) -> 'DirichletBC':
        """
        Create Dirichlet BC from a function.

        Parameters:
            surface: NURBS surface
            boundary: Boundary identifier
            func: Function f(x, y) -> value

        Returns:
            DirichletBC with values evaluated at boundary control points
        """
        dof_indices = get_boundary_dofs_2d(surface, boundary)
        control_points = surface.control_points[dof_indices]
        values = np.array([func(cp[0], cp[1]) for cp in control_points])
        return cls(dof_indices, values)

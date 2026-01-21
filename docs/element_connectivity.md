# Element-Control Point Connectivity in IGA

This document explains the data structures for element-control point linking, which is fundamental for IGA and essential for THB-splines and T-splines.

## Overview

In Isogeometric Analysis, unlike classical FEM where elements have fixed node connectivity, the relationship between elements and basis functions (control points) is more complex:

- Each element has a **variable** number of contributing basis functions (depends on polynomial degree)
- Basis functions have **overlapping support** across multiple elements
- For hierarchical splines (THB), connectivity changes with refinement level
- For T-splines, connectivity can be **irregular** (non-tensor-product)

## Data Structures

### Element

```python
@dataclass
class Element:
    id: int                           # Unique identifier
    level: int                        # Refinement level (0 = coarsest)
    parametric_bounds: Tuple[...]     # Domain in parameter space
    control_point_ids: List[int]      # <-- CONNECTIVITY DATA
    extraction_operator: np.ndarray   # Bézier extraction C_e
    degrees: Tuple[int, ...]          # Polynomial degrees
    active: bool                      # For THB truncation
```

**Key field: `control_point_ids`**
- Ordered list of control point IDs that contribute to this element
- For degree `p` in 2D: contains `(p+1)² = 9` entries (for p=2)
- Order matches rows of `extraction_operator`

### ControlPoint

```python
@dataclass
class ControlPoint:
    id: int                           # Unique identifier
    coordinates: np.ndarray           # Physical location
    weight: float                     # NURBS weight
    level: int                        # Refinement level
    active: bool                      # For THB truncation
    supported_elements: Set[int]      # <-- INVERSE CONNECTIVITY
```

**Key field: `supported_elements`**
- Set of element IDs where this control point's basis function is non-zero
- Enables efficient queries: "which elements does this CP affect?"

### Mesh

```python
class Mesh:
    elements: Dict[int, Element]
    control_points: Dict[int, ControlPoint]
    active_elements: Set[int]
    active_control_points: Set[int]
```

## Bidirectional Linking Invariant

The following invariant **must always hold**:

```
cp_id ∈ element.control_point_ids  ⟺  element.id ∈ control_points[cp_id].supported_elements
```

This means:
- If element E lists control point C in its connectivity → C must list E in its support
- If control point C lists element E in its support → E must list C in its connectivity

The `Mesh` class verifies this invariant on construction.

## Connectivity Example

For a 2×2 mesh with degree p=2 on [0,1]²:

```
Control Points (4×4 = 16 total):
    12──13──14──15
    │   │   │   │
    8───9──10──11
    │   │   │   │
    4───5───6───7
    │   │   │   │
    0───1───2───3

Elements (2×2 = 4 total):
    ┌───────┬───────┐
    │ E[2]  │ E[3]  │
    │       │       │
    ├───────┼───────┤
    │ E[0]  │ E[1]  │
    │       │       │
    └───────┴───────┘
```

### Element Connectivity

Each element has (p+1)² = 9 control points:

| Element | control_point_ids |
|---------|-------------------|
| E[0]    | [0, 1, 2, 4, 5, 6, 8, 9, 10] |
| E[1]    | [1, 2, 3, 5, 6, 7, 9, 10, 11] |
| E[2]    | [4, 5, 6, 8, 9, 10, 12, 13, 14] |
| E[3]    | [5, 6, 7, 9, 10, 11, 13, 14, 15] |

### Control Point Support

| Control Point | supported_elements | Location |
|---------------|-------------------|----------|
| CP[0]  | {0}           | Corner (1 element) |
| CP[1]  | {0, 1}        | Edge (2 elements) |
| CP[5]  | {0, 1, 2, 3}  | Interior (4 elements) |
| CP[15] | {3}           | Corner (1 element) |

## Why This Matters for THB-Splines

In Truncated Hierarchical B-splines (THB):

1. **Refinement creates new elements** at higher levels
2. **Child control points** replace parent control points locally
3. **Parent control points are truncated** (deactivated) where children exist
4. Connectivity **changes dynamically** with refinement

### THB Refinement Scenario

```
Before refinement (level 0):
┌───────────────┐
│               │
│    E[0]       │  CP[0,1,2,3,4,5,6,7,8] all active
│               │
└───────────────┘

After refining E[0] (level 1 children):
┌───────┬───────┐
│ E[4]  │ E[5]  │  New elements at level 1
├───────┼───────┤  New control points at level 1
│ E[6]  │ E[7]  │  Some level-0 CPs truncated (deactivated)
└───────┴───────┘
```

The data structures support this:
- `Element.level` tracks refinement level
- `ControlPoint.level` tracks when CP was introduced
- `ControlPoint.active` can be set to False for truncation
- `Mesh.active_control_points` gives the current active set

## Why This Matters for T-Splines

In T-splines:
- Control points can have **local knot vectors** (not global)
- Connectivity can be **irregular** (T-junctions)
- Number of supporting elements varies per control point

The explicit connectivity data (`control_point_ids` and `supported_elements`) handles this naturally without assuming tensor-product structure.

## API Usage

### Get connectivity for an element

```python
mesh = build_mesh_2d(surface)

# Get element's control points
elem = mesh.get_element(0)
cp_ids = elem.control_point_ids  # [0, 1, 2, 4, 5, 6, 8, 9, 10]

# Get coordinates for these CPs
coords = mesh.get_control_point_coordinates(cp_ids)
weights = mesh.get_control_point_weights(cp_ids)
```

### Get support for a control point

```python
# Which elements does CP 5 affect?
cp = mesh.get_control_point(5)
elem_ids = cp.supported_elements  # {0, 1, 2, 3}

# Get those elements
for eid in elem_ids:
    elem = mesh.get_element(eid)
    # ... process element
```

### Query active subsets

```python
# Iterate over active elements only (for solver)
for elem in mesh.get_active_elements():
    # elem.control_point_ids gives connectivity
    pass

# Get active control points (DOFs for solver)
active_cps = mesh.get_active_control_points_list()
n_dof = mesh.n_dof  # = number of active CPs
```

### Modify connectivity (for refinement)

```python
# Add a link
mesh.link_element_to_control_point(element_id, cp_id)

# Remove a link
mesh.unlink_element_from_control_point(element_id, cp_id)

# Deactivate (THB truncation)
mesh.deactivate_control_point(cp_id)
mesh.deactivate_element(elem_id)
```

## Solver Integration

The solver uses connectivity data during assembly:

```python
for element in mesh.get_active_elements():
    # 1. Get element's control points
    cp_ids = element.control_point_ids

    # 2. Compute element matrices in Bernstein basis
    K_bern, f_bern = compute_element_matrices(element)

    # 3. Apply extraction: K_e = C @ K_bern @ C^T
    C_e = element.extraction_operator
    K_e = C_e @ K_bern @ C_e.T
    f_e = C_e @ f_bern

    # 4. Scatter to global using connectivity
    for i_local, i_global in enumerate(cp_ids):
        f[i_global] += f_e[i_local]
        for j_local, j_global in enumerate(cp_ids):
            K[i_global, j_global] += K_e[i_local, j_local]
```

## Data Structure Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                            MESH                                 │
│  ┌──────────────────────┐    ┌──────────────────────┐          │
│  │ elements: Dict       │    │ control_points: Dict │          │
│  │  0 → Element         │    │  0 → ControlPoint    │          │
│  │  1 → Element         │    │  1 → ControlPoint    │          │
│  │  ...                 │    │  ...                 │          │
│  └──────────────────────┘    └──────────────────────┘          │
│                                                                 │
│  active_elements: {0, 1, 2, ...}                               │
│  active_control_points: {0, 1, 2, ...}                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────┐         ┌─────────────────────────┐
│ Element                 │         │ ControlPoint            │
│  id: 0                  │         │  id: 5                  │
│  level: 0               │         │  level: 0               │
│  active: True           │         │  active: True           │
│  control_point_ids:     │◄───────►│  supported_elements:    │
│    [0,1,2,4,5,6,8,9,10] │         │    {0, 1, 2, 3}         │
│  extraction_operator    │         │  coordinates: [x, y]    │
│  parametric_bounds      │         │  weight: 1.0            │
└─────────────────────────┘         └─────────────────────────┘
         │                                    │
         │  Bidirectional Invariant:          │
         │  5 ∈ E[0].cp_ids ⟺ 0 ∈ CP[5].elems │
         └────────────────────────────────────┘
```

## Future Extensions

### THB-Spline Implementation

When implementing THB-splines, the following operations will be needed:

```python
def refine_element(mesh, elem_id):
    """Refine an element, creating children at next level."""
    parent = mesh.get_element(elem_id)
    new_level = parent.level + 1

    # 1. Create child elements at new_level
    # 2. Create new control points at new_level
    # 3. Compute new extraction operators
    # 4. Set up connectivity for children
    # 5. Truncate (deactivate) parent CPs where children exist
    # 6. Deactivate parent element

    mesh.deactivate_element(elem_id)
    for cp_id in truncated_cps:
        mesh.deactivate_control_point(cp_id)
```

### T-Spline Implementation

For T-splines, additional data may be needed:

```python
@dataclass
class TSplineControlPoint(ControlPoint):
    local_knot_vectors: Tuple[np.ndarray, ...]  # Per-direction
    # Connectivity via supported_elements still works
```

The current architecture supports this extension without changing the solver.

# THB-Splines: Truncated Hierarchical B-Splines

A comprehensive guide to understanding THB-splines for local mesh refinement in IGA.

**Author:** Wataru Fukuda
**Date:** 2025-01-19

---

# Table of Contents

1. [Introduction](#1-introduction)
2. [The Problem: Why Local Refinement?](#2-the-problem-why-local-refinement)
3. [Hierarchical B-Splines (HB-Splines)](#3-hierarchical-b-splines-hb-splines)
4. [Truncated Hierarchical B-Splines (THB-Splines)](#4-truncated-hierarchical-b-splines-thb-splines)
5. [Implementation Walkthrough](#5-implementation-walkthrough)
6. [Code Reference](#6-code-reference)

---

## 1. Introduction

**THB-splines** (Truncated Hierarchical B-Splines) are an extension of B-splines that allow **local mesh refinement** while maintaining:

- **Partition of unity** (basis functions sum to 1)
- **Linear independence** (no redundant DOFs)
- **Analysis-suitable properties** for IGA

In standard IGA, refining the mesh means adding knots everywhere (global refinement), which is wasteful when you only need more resolution in a specific region (e.g., near a crack tip or boundary layer).

THB-splines solve this by allowing you to refine only where needed.

---

## 2. The Problem: Why Local Refinement?

### Standard B-Spline Refinement

In a standard B-spline/NURBS surface, adding a knot affects the entire row or column of control points:

```
Level 0 (4x4 = 16 elements):
+---+---+---+---+
| 0 | 1 | 2 | 3 |
+---+---+---+---+
| 4 | 5 | 6 | 7 |
+---+---+---+---+
| 8 | 9 | 10| 11|
+---+---+---+---+
| 12| 13| 14| 15|
+---+---+---+---+

To refine element 0, you must add knots that propagate
across the entire mesh → all elements in row 0 and column 0 are affected.
```

This leads to **unnecessary DOFs** in regions that don't need refinement.

### The Solution: Hierarchical Refinement

THB-splines use multiple **refinement levels**:

```
Level 0 (coarse):           Level 1 (fine, local):
+-------+-------+           +--+--+
|       |       |           |  |  |
|  E0   |  E1   |  →        +--+--+  (only where needed)
|       |       |           |  |  |
+-------+-------+           +--+--+
```

Elements at level 0 that are refined get replaced by 4 child elements at level 1.

---

## 3. Hierarchical B-Splines (HB-Splines)

### The Basic Idea

Hierarchical B-splines use **nested** knot vectors:

- **Level 0**: Coarse mesh with knot vector `Ξ₀`
- **Level 1**: Fine mesh with knot vector `Ξ₁ ⊃ Ξ₀` (contains all knots from level 0 plus new ones)

The key relationship: **every coarse basis function can be written as a linear combination of fine basis functions**.

This is expressed by the **refinement matrix** `T`:

```
N_coarse(ξ) = T · N_fine(ξ)
```

See: [`refine_knot_vector_dyadic`](../watfIGA/discretization/knot_vector.py#L349-L380)

### Dyadic Refinement

Our implementation uses **dyadic refinement**: each element is split into 2×2 = 4 child elements by inserting midpoints.

```python
# From knot_vector.py:362-366
# Get midpoints of all non-zero spans
midpoints = []
for xi_start, xi_end in kv.elements:
    mid = 0.5 * (xi_start + xi_end)
    midpoints.append(mid)
```

### The Problem with HB-Splines

Plain hierarchical B-splines have a problem: **linear dependence**.

When you have both coarse and fine basis functions active in the same region, they are not linearly independent because coarse functions can be expressed as combinations of fine functions.

This causes **ill-conditioned** or **singular stiffness matrices**.

---

## 4. Truncated Hierarchical B-Splines (THB-Splines)

### The Three States of Basis Functions

In THB-splines, each coarse basis function has one of **three states** depending on how its support relates to the refined region:

| State | Condition | Behavior |
|-------|-----------|----------|
| **1. Active & Untruncated** | Support entirely OUTSIDE refined region | Full original basis function |
| **2. Active & Truncated** | Support PARTIALLY overlaps refined region | Modified function with reduced support |
| **3. Inactive (Deactivated)** | Support entirely INSIDE refined region | Replaced by fine basis functions |

### The Truncation Formula

For a truncated basis function, the THB basis is defined as:

```
trunc(N_i^l) = N_i^l - Σ_j c_ij · N_j^{l+1}
```

Where:
- `N_i^l` is the coarse basis function at level `l`
- `N_j^{l+1}` are fine basis functions at level `l+1` that are **active in the refined region**
- `c_ij` are the refinement coefficients from `N_i^l = Σ_j c_ij · N_j^{l+1}`

This subtraction **removes** the contribution of the coarse function inside the refined region.

See: [`_compute_truncation_coefficients`](../watfIGA/geometry/thb.py#L618)

### Visual Example

```
Level 0 mesh (coarse):
+-------+-------+-------+-------+
|       |       |       |       |
|  E00  |  E10  |  E20  |  E30  |
|       |       |       |       |
+-------+-------+-------+-------+
|       |       |       |       |
|  E01  |  E11  |  E21  |  E31  |
|       |       |       |       |
+-------+-------+-------+-------+

Mark elements E00, E10, E01 for refinement.

After refinement:
+---+---+---+---+-------+-------+
|   |   |   |   |       |       |
+---+---+---+---+  E20  |  E30  |   ← L0 elements (coarse)
|   |   |   |   |       |       |
+---+---+---+---+-------+-------+
|   |   |       |       |       |
+---+---+  E11  |  E21  |  E31  |   ← L0 elements
|   |   |       |       |       |
+---+---+-------+-------+-------+
    ↑
    L1 elements (fine)
```

### Basis Function States Example

Consider the basis functions for the mesh above:

```
Basis function support visualization (p=2, support = 3×3 elements):

  L0 basis fully inside refined region → DEACTIVATED
  ┌───────────────┐
  │ ■ ■ ■ │       │     Support entirely in refined area
  │ ■ ■ ■ │       │     → Replaced by L1 basis functions
  │ ■ ■ ■ │       │
  └───────┴───────┘

  L0 basis partially overlapping → TRUNCATED
  ┌───────────────┐
  │ ■ ■ │ ■       │     Support crosses boundary
  │ ■ ■ │ ■       │     → trunc(N) = N - Σ c_ij N^fine
  │     │         │     → Zero inside, nonzero outside
  └─────┴─────────┘

  L0 basis fully outside refined region → UNTRUNCATED
  ┌───────────────┐
  │     │ ■ ■ ■   │     Support entirely outside
  │     │ ■ ■ ■   │     → Original basis unchanged
  │     │ ■ ■ ■   │
  └─────┴─────────┘
```

### Why Truncation Works

The key insight is that truncation ensures:

1. **Partition of unity**: `Σ N_i = 1` is preserved everywhere
2. **Linear independence**: No redundant DOFs
3. **Smooth transition**: Truncated functions smoothly transition between refined and unrefined regions

See: [`_apply_truncation_at_level`](../watfIGA/geometry/thb.py#L542)

### Important: The Transition Zone

A common misconception is that truncated L0 functions are exactly zero **everywhere** inside the refined region. This is **NOT** always true!

The truncation formula only subtracts **interior** L1 contributions:

```
trunc(N_i^l) = N_i^l - Σ_{j: supp(N_j) ⊂ Ω} c_ij · N_j^{l+1}
```

The sum is only over L1 basis functions whose support is **entirely contained** in the refined domain Ω. At the boundary of the refined domain, there may be L1 functions whose support extends outside - these are NOT subtracted.

**Example**: For an L-shaped refinement pattern:

```
Refined coarse elements: (0,0), (1,0), (0,1)

At eta=0.0 (bottom row):
- Refined region in xi: [0, 0.5]
- Interior L1 at j=0: i=0,1,2,3 with max xi coverage = 0.5

At xi=0.4, eta=0.0:
- This point IS inside the refined coarse element (1,0)
- BUT truncated L0 functions may be NON-ZERO here!

Why? Because:
- L0 #2 has refinement contributions from L1 (3,0), L1 (4,0), L1 (5,0)
- L1 (3,0) is interior (support [0.125, 0.5] ⊂ [0, 0.5])
- L1 (4,0) is NOT interior (support [0.25, 0.625] extends outside!)
- The L1 (4,0) contribution is NOT subtracted → leaves residual
```

**Key Points**:
1. The partition of unity is **always** satisfied (sum of all active basis = 1)
2. Truncated functions are zero where **fully** covered by interior L1 functions
3. At the "transition zone" near the refined region boundary, truncated L0 functions may be non-zero
4. This is **correct** THB-spline behavior - it ensures smooth geometric representation

---

## 5. Implementation Walkthrough

### Step 1: Mark Elements for Refinement

First, you identify which elements need refinement:

```python
# From thb_mesh.py example
thb = THBSurface.from_nurbs_surface(surface)

# Mark specific elements for refinement
elements_to_refine = [(0, 0), (1, 0), (0, 1)]
for ei, ej in elements_to_refine:
    thb.refine_element(0, ei, ej)  # level=0, element indices
```

See: [`THBSurface.refine_element`](../watfIGA/geometry/thb.py#L359-L384)

### Step 2: Compute Refined Control Points

When an element is refined, we need control points at the finer level. These are computed using the **refinement matrix**:

```
P_fine = T · P_coarse
```

For NURBS, we work with weighted control points:

```python
# From thb.py:405-415
# Refine control points: P_fine = T @ P_coarse
# For NURBS, we work with weighted control points
weighted_cp = cp_coarse * w_coarse[:, np.newaxis]
weighted_cp_fine = T_2d @ weighted_cp
w_fine = T_2d @ w_coarse

# Unweight
cp_fine = weighted_cp_fine / w_fine[:, np.newaxis]
```

See: [`_compute_refined_control_points`](../watfIGA/geometry/thb.py#L386-L419)

The 2D refinement matrix is the **Kronecker product** of the 1D matrices:

```python
# From thb.py:398-399
T_2d = np.kron(T_eta, T_xi)
```

### Step 3: Activate Fine Basis Functions

When an element is refined, the fine basis functions that cover the refined region are activated:

```python
# From thb.py:470-485
# Collect all fine basis functions active on these fine elements
fine_basis_to_activate = set()
for fej in range(fine_elem_eta_start, fine_elem_eta_end):
    # ... find active basis functions on fine elements
    for j in active_eta_fine:
        for i in active_xi_fine:
            global_idx = j * n_xi_fine + i
            fine_basis_to_activate.add(global_idx)

# Activate fine basis functions
self._active_cps[to_level].update(fine_basis_to_activate)
```

See: [`_update_active_basis_for_refinement`](../watfIGA/geometry/thb.py#L421-L491)

### Step 4: Apply Truncation (Classify Coarse Basis)

The **crucial step** - classifying each coarse basis function into one of three states:

```python
# From thb.py - simplified logic
for global_idx in list(self._active_cps.get(level, set())):
    # Count elements in support and how many are refined
    elements_in_support = [...]  # all elements touching this basis
    refined_in_support = [...]   # subset that are refined

    n_total = len(elements_in_support)
    n_refined = len(refined_in_support)

    if n_refined == 0:
        # No overlap with refined region - stays UNTRUNCATED
        pass
    elif n_refined == n_total:
        # Fully inside refined region - DEACTIVATE
        coarse_to_deactivate.add(global_idx)
    else:
        # Partially overlaps - TRUNCATE
        coarse_to_truncate.add(global_idx)
        # Compute truncation coefficients c_ij
        self._compute_truncation_coefficients(level, global_idx, refined_in_support)

# Deactivate fully-covered basis functions
self._active_cps[level] -= coarse_to_deactivate

# Mark truncated basis functions
self._truncated_basis[level].update(coarse_to_truncate)
```

See: [`_apply_truncation_at_level`](../watfIGA/geometry/thb.py#L542)

### Step 5: Finalize Refinement

Call `finalize_refinement()` to apply truncation across all levels:

```python
# From thb.py
def finalize_refinement(self) -> None:
    """
    Finalize refinement after marking elements.
    This computes the final active basis sets based on THB truncation rules:
    1. Filter fine basis functions - only those with support entirely in refined region
    2. Apply truncation to coarse basis functions
    """
    for level in range(self.n_levels - 1):
        self._filter_fine_basis_functions(level)
        self._apply_truncation_at_level(level)
```

**Important**: The filtering step ensures that only fine basis functions whose support is **entirely contained** in the refined region Ω^{l+1} are active. Functions at the boundary (with support extending outside) are NOT included in the THB basis - their contribution is handled by the truncated coarse functions.

See: [`finalize_refinement`](../watfIGA/geometry/thb.py#L533)

---

## 6. Code Reference

### Key Files

| File | Description |
|------|-------------|
| [`watfIGA/geometry/thb.py`](../watfIGA/geometry/thb.py) | Main THB-spline implementation |
| [`watfIGA/discretization/knot_vector.py`](../watfIGA/discretization/knot_vector.py) | Knot vector operations, refinement matrices |
| [`watfIGA/discretization/mesh.py`](../watfIGA/discretization/mesh.py) | THB mesh building (`build_thb_mesh`) |
| [`watfIGA/visualization/basis.py`](../watfIGA/visualization/basis.py) | THB basis evaluation & visualization |
| [`examples/src/thb_mesh.py`](../examples/src/thb_mesh.py) | Example: THB mesh generation |
| [`examples/src/thb_basis_visualization.py`](../examples/src/thb_basis_visualization.py) | Example: Visualize THB basis functions |
| [`examples/src/thb_poisson.py`](../examples/src/thb_poisson.py) | Example: Poisson problem with THB |

### Key Classes

| Class | Location | Purpose |
|-------|----------|---------|
| `THBSurface` | [thb.py:258](../watfIGA/geometry/thb.py#L258) | Main THB surface class |
| `THBHierarchy2D` | [thb.py:186](../watfIGA/geometry/thb.py#L186) | 2D hierarchical structure |
| `THBHierarchy1D` | [thb.py:37](../watfIGA/geometry/thb.py#L37) | 1D hierarchical knot vectors |
| `THBBasisVisualizer` | [basis.py:155](../watfIGA/visualization/basis.py#L155) | Basis function evaluation & visualization |

### Key Methods

| Method | Location | Purpose |
|--------|----------|---------|
| `refine_element()` | [thb.py:399](../watfIGA/geometry/thb.py#L399) | Mark element for refinement |
| `finalize_refinement()` | [thb.py:533](../watfIGA/geometry/thb.py#L533) | Apply truncation rules |
| `_filter_fine_basis_functions()` | [thb.py:547](../watfIGA/geometry/thb.py#L547) | Filter L1 basis to interior only |
| `_compute_refined_control_points()` | [thb.py:426](../watfIGA/geometry/thb.py#L426) | Compute level n+1 control points |
| `_update_active_basis_for_refinement()` | [thb.py:461](../watfIGA/geometry/thb.py#L461) | Activate fine basis functions |
| `_apply_truncation_at_level()` | [thb.py:592](../watfIGA/geometry/thb.py#L592) | Classify & truncate coarse basis |
| `_compute_truncation_coefficients()` | [thb.py:669](../watfIGA/geometry/thb.py#L669) | Compute truncation coefficients |
| `_is_fine_basis_interior()` | [thb.py:754](../watfIGA/geometry/thb.py#L754) | Check if fine basis support is interior |
| `get_truncated_basis()` | [thb.py:366](../watfIGA/geometry/thb.py#L366) | Get truncated basis indices |
| `get_truncation_coefficients()` | [thb.py:379](../watfIGA/geometry/thb.py#L379) | Get truncation coefficients |
| `is_basis_truncated()` | [thb.py:375](../watfIGA/geometry/thb.py#L375) | Check if basis is truncated |
| `get_basis_support()` | [thb.py:139](../watfIGA/geometry/thb.py#L139) | Get support interval of basis function |
| `build_thb_mesh()` | [mesh.py:790](../watfIGA/discretization/mesh.py#L790) | Build mesh from THB surface |
| `refine_knot_vector_dyadic()` | [knot_vector.py:349](../watfIGA/discretization/knot_vector.py#L349) | Dyadic refinement |

---

## Summary: The THB Algorithm

```
1. Start with level-0 mesh (all L0 control points active)
       ↓
2. Mark elements for refinement
       ↓
3. For each marked element:
   a. Ensure level+1 knot vectors exist
   b. Compute level+1 control points via refinement matrix
   c. Activate fine basis functions covering the element
       ↓
4. Filter fine basis functions:
   - Keep only L1 basis whose support is ENTIRELY within refined region
   - L1 basis at boundary (support extends outside) are NOT active
       ↓
5. Apply truncation (classify each coarse basis function):
   For each active coarse basis function:
     - Count elements in support: total vs refined

     n_refined == 0        → UNTRUNCATED (keep as-is)
     n_refined == n_total  → DEACTIVATE  (remove from active set)
     0 < n_refined < total → TRUNCATE    (compute truncation coefficients)

   For truncated basis:
     - Compute coefficients: c_ij from refinement relation
     - Only subtract interior fine basis (support ⊂ Ω^{l+1})
       ↓
6. Result: Mixed-level mesh with:
   - Level-0 elements in unrefined regions (use L0 basis)
   - Level-1 elements in refined regions (use interior L1 basis only)
   - Truncated L0 basis provide smooth transition at boundary
   - Linear independence and partition of unity preserved

   Note: Truncated L0 functions are zero only where FULLY covered by
   interior L1 functions. At the "transition zone" they may be non-zero
   even inside refined coarse elements - this is correct behavior!
```

### Verifying the Implementation

Use `THBBasisVisualizer` from the visualization module to verify partition of unity:

```python
from watfIGA.visualization import THBBasisVisualizer

# Create visualizer
viz = THBBasisVisualizer(thb_surface, n_points=100)

# Check partition of unity
if viz.check_partition_of_unity():
    print("Partition of unity verified!")

# Get statistics
stats = viz.get_partition_of_unity_stats()
print(f"Sum range: [{stats['min']:.6f}, {stats['max']:.6f}]")

# Plot and save
viz.plot_3d(save_path="thb_basis_3d.png")
viz.plot_heatmaps(save_path="thb_basis_heatmaps.png")
```

Or run the example script:

```bash
python examples/src/thb_basis_visualization.py --save --all
```

This generates:
- `thb_basis_3d.png`: 3D surface plots of all basis functions
- `thb_basis_heatmaps.png`: 2D heatmaps of basis functions

The SUM plot should show constant 1.0 everywhere (partition of unity).

---

## Further Reading

1. **Giannelli, C., Jüttler, B., & Speleers, H.** (2012). THB-splines: The truncated basis for hierarchical splines. *Computer Aided Geometric Design*, 29(7), 485-498.

2. **Vuong, A. V., Giannelli, C., Jüttler, B., & Simeon, B.** (2011). A hierarchical approach to adaptive local refinement in isogeometric analysis. *Computer Methods in Applied Mechanics and Engineering*, 200(49-52), 3554-3567.

3. **Cottrell, J. A., Hughes, T. J., & Bazilevs, Y.** (2009). *Isogeometric Analysis: Toward Integration of CAD and FEA*. John Wiley & Sons.

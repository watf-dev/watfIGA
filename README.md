# watfIGA

A research-grade Isogeometric Analysis (IGA) library in Python.

## Overview

Isogeometric Analysis uses the same basis functions for geometry representation (from CAD) and solution approximation (for analysis), eliminating mesh generation and enabling exact geometry representation.

### Key Features

- **NURBS-based geometry**: Exact representation of conic sections and freeform surfaces
- **Bézier extraction**: Efficient element-level assembly using Bernstein polynomials
- **Spline-agnostic solver**: Solver code works unchanged with NURBS, THB-splines, or T-splines
- **Clean separation**: Geometry, discretization, solver, and post-processing are decoupled

### Architecture

```
iga/
├── geometry/          # NURBS curves/surfaces, B-spline basis
├── discretization/    # Knot vectors, Bézier extraction, mesh building
├── solver/            # PDE solvers (Poisson, elasticity, plates)
├── quadrature/        # Gauss-Legendre integration
├── postprocess/       # Solution sampling, VTK export
└── io/                # CAD import, configuration
```

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd watfIGA

# Install dependencies
pip install numpy scipy

# Optional: for visualization
pip install matplotlib
```

## Quick Start

```python
from watfIGA.geometry.nurbs import make_nurbs_unit_square
from watfIGA.discretization.mesh import build_mesh_2d, get_all_boundary_dofs_2d, DirichletBC
from watfIGA.solver.poisson import PoissonSolver

# Create geometry
surface = make_nurbs_unit_square(p=2, n_elem_xi=4, n_elem_eta=4)

# Build analysis mesh
mesh = build_mesh_2d(surface)

# Solve Poisson equation: -∇²u = f
solver = PoissonSolver(mesh, source=lambda x, y: 1.0)
bc = DirichletBC.homogeneous(get_all_boundary_dofs_2d(surface))
solver.add_dirichlet_bc(bc)
u = solver.run()
```

## Running Examples

```bash
./run_example.sh
# or
python3 examples/nurbs_poisson_2d.py
```

## Running Tests

```bash
# All tests
python3 -m pytest tests/ -v

# Specific module
python3 -m pytest tests/test_poisson.py -v

# With coverage
python3 -m pytest tests/ --cov=iga
```

## Project Structure

```
watfIGA/
├── iga/                # Core library
├── tests/              # Unit and integration tests
├── examples/           # Usage examples
├── run_example.sh      # Script to run examples
└── README.md
```

## Mathematical Background

### Bézier Extraction

The key insight enabling efficient IGA is Bézier extraction. Any B-spline/NURBS basis can be expressed as:

```
N(ξ) = C · B(ξ)
```

where `N` are the spline basis functions, `B` are Bernstein polynomials, and `C` is the extraction operator. This allows:

1. Element-level integration using simple Bernstein polynomials
2. Standard FEM assembly procedures
3. Spline-type agnostic solvers

### Supported PDEs

- **Poisson equation**: `-∇²u = f` (implemented)
- **Linear elasticity**: (placeholder)
- **Kirchhoff-Love plates**: (placeholder)
- **Reissner-Mindlin plates**: (placeholder)

## Roadmap

- [ ] THB-spline (Truncated Hierarchical B-spline) support
- [ ] T-spline support
- [ ] 3D solid mechanics
- [ ] CAD file import (STEP, IGES)
- [ ] Adaptive refinement

## References

1. Hughes, T.J.R., Cottrell, J.A., Bazilevs, Y. (2005). "Isogeometric analysis: CAD, finite elements, NURBS, exact geometry and mesh refinement"
2. Borden, M.J., Scott, M.A., Evans, J.A., Hughes, T.J.R. (2011). "Isogeometric finite element data structures based on Bézier extraction of NURBS"

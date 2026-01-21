"""
IGA - Isogeometric Analysis Library

A research-grade implementation of Isogeometric Analysis (IGA),
supporting NURBS and THB-splines (Truncated Hierarchical B-splines),
with architecture designed for future extension to T-splines.

Key modules:
- geometry: NURBS, THB-splines, B-spline basis functions
- discretization: Knot vectors, elements, Bézier extraction
- solver: PDE solvers (Poisson, elasticity, etc.)
- quadrature: Gauss-Legendre integration
- postprocess: Solution sampling and VTK export

Quick start (NURBS):
    from watfIGA.geometry.nurbs import make_nurbs_unit_square
    from watfIGA.discretization.mesh import Mesh, get_all_boundary_dofs_2d
    from watfIGA.solver.poisson import PoissonSolver

    # Create geometry and mesh
    surface = make_nurbs_unit_square(p=2, n_elem_xi=4, n_elem_eta=4)
    mesh = Mesh.build(surface)

    # Solve Poisson equation
    solver = PoissonSolver(mesh, source=lambda x, y: 1.0)
    bc = DirichletBC.homogeneous(get_all_boundary_dofs_2d(surface))
    solver.add_dirichlet_bc(bc)
    u = solver.run()

Quick start (THB-splines):
    from watfIGA.geometry.nurbs import make_nurbs_unit_square
    from watfIGA.geometry.thb import THBSurface
    from watfIGA.discretization.mesh import Mesh

    # Create THB surface from NURBS
    surface = make_nurbs_unit_square(p=2, n_elem_xi=4, n_elem_eta=4)
    thb = THBSurface.from_nurbs_surface(surface)

    # Refine specific elements
    thb.refine_element(0, 0, 0)  # Refine element at index (0, 0)

    # Build mesh with hierarchical refinement
    mesh = Mesh.build(thb)
"""

__version__ = "0.1.0"
__author__ = "Wataru Fukuda"

# Core imports for convenience
from .geometry.nurbs import NURBSSurface, make_nurbs_unit_square, make_nurbs_rectangle
from .geometry.thb import THBSurface, THBHierarchy1D, THBHierarchy2D
from .discretization.mesh import Mesh, build_mesh_2d, get_all_boundary_dofs_2d, DirichletBC
from .solver.poisson import PoissonSolver
from .postprocess.vtk import export_vtk_structured_2d
from .postprocess.sampling import sample_solution_2d

#!/usr/bin/env python3
"""
Example: Solve Poisson equation on a THB-spline mesh with local refinement.

This example demonstrates:
1. Creating a THB surface with local refinement
2. Building the hierarchical mesh
3. Solving the Poisson equation: -Laplacian(u) = f
4. Comparing solutions on uniform vs locally refined meshes

The problem:
    -Laplacian(u) = 2*pi^2 * sin(pi*x) * sin(pi*y)  in [0,1]^2
    u = 0  on boundary

Exact solution: u(x,y) = sin(pi*x) * sin(pi*y)

Created: 2025-01-19
Author: Wataru Fukuda
"""

import sys
import os

# Add project root to path (two levels up from examples/src/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from watfIGA.geometry.primitives import make_nurbs_unit_square
from watfIGA.geometry.thb import THBSurface
from watfIGA.discretization.mesh import Mesh, get_all_boundary_dofs_2d, DirichletBC
from watfIGA.solver.poisson import PoissonSolver
from watfIGA.postprocess.sampling import sample_solution_2d


def exact_solution(x, y):
    """Exact solution: u(x,y) = sin(pi*x) * sin(pi*y)"""
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def source_function(x, y):
    """Source term: f = 2*pi^2 * sin(pi*x) * sin(pi*y)"""
    return 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)


def compute_l2_error(surface, u_coeffs, n_sample=20):
    """Compute L2 error between numerical and exact solution."""
    # Sample numerical solution on a grid
    X, Y, U = sample_solution_2d(surface, u_coeffs, n_xi=n_sample, n_eta=n_sample)

    # Compute exact solution at physical points
    U_exact = exact_solution(X, Y)

    # Compute L2 error
    error = np.sqrt(np.mean((U - U_exact)**2))
    return error


def solve_uniform_mesh(p, n_elem):
    """Solve on a uniform mesh."""
    surface = make_nurbs_unit_square(p=p, n_elem_xi=n_elem, n_elem_eta=n_elem)
    mesh = Mesh.build(surface)

    # Set up solver
    solver = PoissonSolver(mesh, source=source_function)

    # Apply homogeneous Dirichlet BC on all boundaries
    boundary_dofs = get_all_boundary_dofs_2d(surface)
    bc = DirichletBC.homogeneous(boundary_dofs)
    solver.add_dirichlet_bc(bc)

    # Solve
    u = solver.run()

    # Compute error
    error = compute_l2_error(surface, u)

    return mesh, u, error, surface


def solve_thb_mesh(p, n_elem_base, refine_regions):
    """Solve on a THB mesh with local refinement."""
    # Create base surface
    surface = make_nurbs_unit_square(p=p, n_elem_xi=n_elem_base, n_elem_eta=n_elem_base)

    # Convert to THB
    thb = THBSurface.from_nurbs_surface(surface)

    # Refine specified elements
    for ei, ej in refine_regions:
        thb.refine_element(0, ei, ej)

    thb.finalize_refinement()

    # Build mesh
    mesh = Mesh.build(thb)

    # Set up solver
    solver = PoissonSolver(mesh, source=source_function)

    # For THB, we need to identify boundary DOFs differently
    # Get all control points and check if they're on boundary
    boundary_dofs = []
    for cp_id in mesh.active_control_points:
        cp = mesh.get_control_point(cp_id)
        x, y = cp.coordinates[:2]
        # Check if on boundary (within tolerance)
        tol = 1e-10
        if abs(x) < tol or abs(x - 1) < tol or abs(y) < tol or abs(y - 1) < tol:
            boundary_dofs.append(cp_id)

    bc = DirichletBC.homogeneous(np.array(boundary_dofs))
    solver.add_dirichlet_bc(bc)

    # Solve
    u = solver.run()

    # For THB meshes, we cannot directly use the standard error computation
    # since the DOF numbering is different. We'll compute error manually.
    # For now, return None for error (THB error requires special handling)
    error = None

    return mesh, u, error, thb, surface


def main():
    import argparse
    parser = argparse.ArgumentParser(description="THB-spline Poisson solver example")
    parser.add_argument("--debug", action="store_true", help="enable debug output")
    options = parser.parse_args()

    print("=" * 70)
    print("THB-Spline Poisson Solver Example")
    print("=" * 70)
    print("\nProblem: -Laplacian(u) = f in [0,1]^2, u = 0 on boundary")
    print("Exact solution: u(x,y) = sin(pi*x) * sin(pi*y)")

    p = 2  # Polynomial degree

    # Solve on uniform meshes with increasing refinement
    print("\n" + "-" * 70)
    print("Uniform Mesh Convergence")
    print("-" * 70)
    print(f"{'Elements':<12} {'DOFs':<10} {'L2 Error':<15}")
    print("-" * 40)

    for n_elem in [2, 4, 8]:
        mesh, u, error, _ = solve_uniform_mesh(p, n_elem)
        print(f"{n_elem}x{n_elem:<9} {mesh.n_dof:<10} {error:<15.6e}")

    # Solve on THB mesh with corner refinement
    print("\n" + "-" * 70)
    print("THB Mesh with Local Refinement")
    print("-" * 70)

    # Base mesh: 4x4
    n_elem_base = 4

    # Refine corner region (bottom-left quarter)
    corner_elements = [(0, 0), (1, 0), (0, 1), (1, 1)]

    print(f"\nBase mesh: {n_elem_base}x{n_elem_base} elements")
    print(f"Refined elements: {corner_elements}")

    mesh_thb, u_thb, error_thb, thb, surface_base = solve_thb_mesh(p, n_elem_base, corner_elements)

    print(f"\nTHB Mesh Statistics:")
    print(f"  Total elements: {mesh_thb.n_elements}")
    print(f"  Level-0 elements: {len(mesh_thb.get_elements_at_level(0))}")
    print(f"  Level-1 elements: {len(mesh_thb.get_elements_at_level(1))}")
    print(f"  Active DOFs: {mesh_thb.n_dof}")
    print(f"  Solution computed: {u_thb.shape}")

    # Compare with uniform 4x4 mesh
    mesh_uniform, u_uniform, error_uniform, _ = solve_uniform_mesh(p, n_elem_base)

    print(f"\nComparison (same base resolution {n_elem_base}x{n_elem_base}):")
    print(f"  Uniform mesh DOFs: {mesh_uniform.n_dof}")
    print(f"  THB mesh DOFs: {mesh_thb.n_dof}")
    print(f"  Uniform L2 error: {error_uniform:.6e}")
    print(f"  THB mesh solves successfully with {mesh_thb.n_dof - mesh_uniform.n_dof} additional DOFs")

    if options.debug:
        print("\n" + "=" * 70)
        print("DEBUG: Solution Coefficients")
        print("=" * 70)

        print(f"\nTHB solution shape: {u_thb.shape}")
        print(f"First 10 coefficients: {u_thb[:10]}")

        print("\nElement-level breakdown:")
        for level in range(mesh_thb.max_level + 1):
            elems = mesh_thb.get_elements_at_level(level)
            cps = mesh_thb.get_control_points_at_level(level)
            print(f"  Level {level}: {len(elems)} elements, {len(cps)} control points")

    print("\n" + "=" * 70)
    print("Poisson solve completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()

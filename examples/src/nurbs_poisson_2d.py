#!/usr/bin/env python3
"""
Example: 2D Poisson equation on unit square using NURBS.

This example demonstrates the complete IGA pipeline:
1. Create NURBS geometry (unit square)
2. Build analysis mesh with Bézier extraction
3. Solve Poisson equation with homogeneous Dirichlet BCs
4. Visualize solution

Problem:
    -∇²u = f    in Ω = [0,1]²
        u = 0    on ∂Ω

Manufactured solution for verification:
    u_exact = sin(πx) * sin(πy)
    f = 2π² * sin(πx) * sin(πy)

Usage:
    ./examples/src/nurbs_poisson_2d.py

Created: 2025-01-18
Author: Wataru Fukuda
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path (two levels up from examples/src/)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from watfIGA.geometry.primitives import make_nurbs_unit_square
from watfIGA.discretization.mesh import build_mesh_2d, get_all_boundary_dofs_2d, DirichletBC
from watfIGA.solver.poisson import PoissonSolver
from watfIGA.postprocess.sampling import sample_solution_2d, compute_l2_error
from watfIGA.postprocess.vtk import export_vtk_structured_2d


def run(degree: int = 2,
        n_elements: int = 4,
        export_vtk: bool = True,
        verbose: bool = True):
    """
    Run the 2D Poisson example.

    Parameters:
        degree: Polynomial degree (p = q)
        n_elements: Number of elements per direction
        export_vtk: Whether to export VTK file
        verbose: Print progress information

    Returns:
        Dictionary with results (solution, error, mesh info)
    """
    if verbose:
        print("=" * 60)
        print("IGA 2D Poisson Example")
        print("=" * 60)
        print(f"Degree: {degree}")
        print(f"Elements: {n_elements} x {n_elements}")
        print()

    # ==========================================================================
    # 1. Create geometry
    # ==========================================================================
    if verbose:
        print("Creating NURBS geometry...")

    surface = make_nurbs_unit_square(p=degree, n_elem_xi=n_elements, n_elem_eta=n_elements)

    if verbose:
        print(f"  Domain: {surface.domain}")
        print(f"  Control points: {surface.n_control_points_per_dir}")
        print(f"  Total DOFs: {surface.n_control_points}")
        print()

    # ==========================================================================
    # 2. Build analysis mesh
    # ==========================================================================
    if verbose:
        print("Building analysis mesh...")

    mesh = build_mesh_2d(surface)

    if verbose:
        print(f"  Number of elements: {mesh.n_elements}")
        print(f"  DOFs: {mesh.n_dof}")
        print()

    # ==========================================================================
    # 3. Define problem
    # ==========================================================================
    # Manufactured solution
    def u_exact(x, y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    def grad_u_exact(x, y):
        dudx = np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
        dudy = np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)
        return (dudx, dudy)

    # Source term: f = -∇²u = 2π² * sin(πx) * sin(πy)
    def source(x, y):
        return 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

    # ==========================================================================
    # 4. Set up solver
    # ==========================================================================
    if verbose:
        print("Setting up Poisson solver...")

    solver = PoissonSolver(mesh, source=source)

    # Homogeneous Dirichlet BC on all boundaries
    boundary_dofs = get_all_boundary_dofs_2d(surface)
    bc = DirichletBC.homogeneous(boundary_dofs)
    solver.add_dirichlet_bc(bc)

    if verbose:
        print(f"  Boundary DOFs: {len(boundary_dofs)}")
        print(f"  Interior DOFs: {mesh.n_dof - len(boundary_dofs)}")
        print()

    # ==========================================================================
    # 5. Assemble and solve
    # ==========================================================================
    if verbose:
        print("Assembling system...")

    solver.assemble()

    if verbose:
        print(f"  Stiffness matrix: {solver.K.shape}, nnz = {solver.K.nnz}")
        print()
        print("Applying boundary conditions...")

    solver.apply_boundary_conditions()

    if verbose:
        print()
        print("Solving linear system...")

    u = solver.solve()

    if verbose:
        print(f"  Solution range: [{u.min():.6f}, {u.max():.6f}]")
        print()

    # ==========================================================================
    # 6. Compute error
    # ==========================================================================
    if verbose:
        print("Computing error...")

    l2_error = compute_l2_error(surface, u, u_exact, n_sample=100)

    if verbose:
        print(f"  L2 error: {l2_error:.6e}")
        print()

    # ==========================================================================
    # 7. Export visualization
    # ==========================================================================
    if export_vtk:
        if verbose:
            print("Exporting VTK file...")

        output_file = Path(__file__).parent / "poisson_2d_solution.vtk"
        export_vtk_structured_2d(str(output_file), surface, u, n_xi=100, n_eta=100)

        # Also export exact solution for comparison
        X, Y, U = sample_solution_2d(surface, u, 100, 100)
        U_exact = np.zeros_like(U)
        for i in range(100):
            for j in range(100):
                U_exact[i, j] = u_exact(X[i, j], Y[i, j])

        output_file_exact = Path(__file__).parent / "poisson_2d_exact.vtk"
        # Export exact solution using the helper function
        _export_grid_vtk(str(output_file_exact), X, Y, U_exact, "u_exact")

        if verbose:
            print()

    # ==========================================================================
    # 8. Summary
    # ==========================================================================
    if verbose:
        print("=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"  Degree: {degree}")
        print(f"  Elements: {n_elements} x {n_elements}")
        print(f"  Total DOFs: {mesh.n_dof}")
        print(f"  L2 error: {l2_error:.6e}")
        print(f"  Max |u_h|: {np.abs(u).max():.6f}")
        print(f"  Max |u_exact|: {1.0:.6f}")  # max of sin(πx)sin(πy) is 1
        print("=" * 60)

    return {
        'solution': u,
        'l2_error': l2_error,
        'mesh': mesh,
        'surface': surface,
        'n_dof': mesh.n_dof,
        'n_elements': mesh.n_elements
    }


def _export_grid_vtk(filename: str, X: np.ndarray, Y: np.ndarray,
                     U: np.ndarray, field_name: str):
    """Helper to export a pre-sampled grid to VTK."""
    n_xi, n_eta = X.shape
    Z = np.zeros_like(X)

    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("IGA Solution\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_GRID\n")
        f.write(f"DIMENSIONS {n_xi} {n_eta} 1\n")

        n_points = n_xi * n_eta
        f.write(f"POINTS {n_points} float\n")

        for j in range(n_eta):
            for i in range(n_xi):
                f.write(f"{X[i, j]} {Y[i, j]} {Z[i, j]}\n")

        f.write(f"\nPOINT_DATA {n_points}\n")
        f.write(f"SCALARS {field_name} float 1\n")
        f.write("LOOKUP_TABLE default\n")

        for j in range(n_eta):
            for i in range(n_xi):
                f.write(f"{U[i, j]}\n")


def convergence_study(degrees: list = None, n_elements_list: list = None):
    """
    Run convergence study over mesh refinements.

    Parameters:
        degrees: List of polynomial degrees to test
        n_elements_list: List of element counts per direction
    """
    if degrees is None:
        degrees = [2, 3]
    if n_elements_list is None:
        n_elements_list = [2, 4, 8, 16]

    print("=" * 70)
    print("Convergence Study: 2D Poisson on Unit Square")
    print("=" * 70)

    results = {}

    for p in degrees:
        print(f"\nDegree p = {p}")
        print("-" * 50)
        print(f"{'Elements':>10} {'DOFs':>10} {'L2 Error':>15} {'Rate':>10}")
        print("-" * 50)

        errors = []
        h_vals = []

        for n_elem in n_elements_list:
            result = run(degree=p, n_elements=n_elem, export_vtk=False, verbose=False)
            l2_error = result['l2_error']
            n_dof = result['n_dof']

            errors.append(l2_error)
            h = 1.0 / n_elem
            h_vals.append(h)

            if len(errors) > 1:
                rate = np.log(errors[-2] / errors[-1]) / np.log(h_vals[-2] / h_vals[-1])
                print(f"{n_elem:>10} {n_dof:>10} {l2_error:>15.6e} {rate:>10.2f}")
            else:
                print(f"{n_elem:>10} {n_dof:>10} {l2_error:>15.6e} {'--':>10}")

        results[p] = {'h': h_vals, 'errors': errors}

    print()
    print("Expected convergence rate: p + 1 for smooth solutions")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="2D Poisson IGA Example")
    parser.add_argument("--degree", "-p", type=int, default=2,
                        help="Polynomial degree (default: 2)")
    parser.add_argument("--elements", "-n", type=int, default=4,
                        help="Number of elements per direction (default: 4)")
    parser.add_argument("--convergence", "-c", action="store_true",
                        help="Run convergence study")
    parser.add_argument("--no-vtk", action="store_true",
                        help="Skip VTK export")

    args = parser.parse_args()

    if args.convergence:
        convergence_study()
    else:
        run(degree=args.degree, n_elements=args.elements, export_vtk=not args.no_vtk)

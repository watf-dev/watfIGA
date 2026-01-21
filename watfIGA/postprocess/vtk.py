"""
VTK export for visualization.

This module exports IGA solutions to VTK format for visualization
in ParaView, VisIt, or other VTK-compatible viewers.

Supported formats:
- VTK Legacy (.vtk) - ASCII, widely compatible
- VTK XML StructuredGrid (.vts) - For regular grids

The solution is sampled on a regular grid since VTK doesn't
natively support NURBS visualization.

TODO: Add UnstructuredGrid export for adaptive sampling
TODO: Add vector field export for gradients/stresses
"""

import numpy as np
from typing import Optional, Dict
from pathlib import Path

from .sampling import sample_solution_2d, sample_solution_gradient_2d
from ..geometry.nurbs import NURBSSurface


def export_vtk_structured_2d(filename: str,
                              surface: NURBSSurface,
                              u: np.ndarray,
                              n_xi: int = 50,
                              n_eta: int = 50,
                              field_name: str = "solution",
                              additional_fields: Optional[Dict[str, np.ndarray]] = None):
    """
    Export 2D solution to VTK StructuredGrid format.

    Parameters:
        filename: Output filename (will add .vtk extension if missing)
        surface: NURBS surface
        u: Solution vector (control point values)
        n_xi, n_eta: Number of sample points
        field_name: Name for the solution field in VTK
        additional_fields: Optional dict of additional scalar fields
                          Each value should be shape (n_xi, n_eta)
    """
    # Ensure .vtk extension
    path = Path(filename)
    if path.suffix != '.vtk':
        path = path.with_suffix('.vtk')

    # Sample solution
    X, Y, U = sample_solution_2d(surface, u, n_xi, n_eta)

    # For 2D in 3D VTK, set Z = 0 or Z = U for height visualization
    Z = np.zeros_like(X)

    # Write VTK file
    with open(path, 'w') as f:
        # Header
        f.write("# vtk DataFile Version 3.0\n")
        f.write("IGA Solution\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_GRID\n")
        f.write(f"DIMENSIONS {n_xi} {n_eta} 1\n")

        # Points
        n_points = n_xi * n_eta
        f.write(f"POINTS {n_points} float\n")

        for j in range(n_eta):
            for i in range(n_xi):
                f.write(f"{X[i, j]} {Y[i, j]} {Z[i, j]}\n")

        # Point data
        f.write(f"\nPOINT_DATA {n_points}\n")

        # Solution field
        f.write(f"SCALARS {field_name} float 1\n")
        f.write("LOOKUP_TABLE default\n")

        for j in range(n_eta):
            for i in range(n_xi):
                f.write(f"{U[i, j]}\n")

        # Additional fields
        if additional_fields:
            for name, values in additional_fields.items():
                f.write(f"\nSCALARS {name} float 1\n")
                f.write("LOOKUP_TABLE default\n")
                for j in range(n_eta):
                    for i in range(n_xi):
                        f.write(f"{values[i, j]}\n")

    print(f"Exported VTK file: {path}")


def export_vtk_with_gradient_2d(filename: str,
                                 surface: NURBSSurface,
                                 u: np.ndarray,
                                 n_xi: int = 50,
                                 n_eta: int = 50,
                                 field_name: str = "solution"):
    """
    Export 2D solution with gradient to VTK format.

    Parameters:
        filename: Output filename
        surface: NURBS surface
        u: Solution vector
        n_xi, n_eta: Number of sample points
        field_name: Name for solution field
    """
    # Sample solution and gradient
    X, Y, U, dU_dx, dU_dy = sample_solution_gradient_2d(surface, u, n_xi, n_eta)

    # Compute gradient magnitude
    grad_mag = np.sqrt(dU_dx**2 + dU_dy**2)

    # Export with additional fields
    export_vtk_structured_2d(
        filename, surface, u, n_xi, n_eta, field_name,
        additional_fields={
            "gradient_x": dU_dx,
            "gradient_y": dU_dy,
            "gradient_magnitude": grad_mag
        }
    )


def export_vtk_xml_2d(filename: str,
                       surface: NURBSSurface,
                       u: np.ndarray,
                       n_xi: int = 50,
                       n_eta: int = 50,
                       field_name: str = "solution"):
    """
    Export 2D solution to VTK XML StructuredGrid format (.vts).

    This format is more compact and supports compression.

    Parameters:
        filename: Output filename (will add .vts extension)
        surface: NURBS surface
        u: Solution vector
        n_xi, n_eta: Number of sample points
        field_name: Name for solution field
    """
    path = Path(filename)
    if path.suffix != '.vts':
        path = path.with_suffix('.vts')

    X, Y, U = sample_solution_2d(surface, u, n_xi, n_eta)
    Z = np.zeros_like(X)

    with open(path, 'w') as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="StructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        f.write(f'  <StructuredGrid WholeExtent="0 {n_xi-1} 0 {n_eta-1} 0 0">\n')
        f.write(f'    <Piece Extent="0 {n_xi-1} 0 {n_eta-1} 0 0">\n')

        # Point data
        f.write('      <PointData Scalars="{}">\n'.format(field_name))
        f.write(f'        <DataArray type="Float64" Name="{field_name}" format="ascii">\n')
        for j in range(n_eta):
            for i in range(n_xi):
                f.write(f'          {U[i, j]}\n')
        f.write('        </DataArray>\n')
        f.write('      </PointData>\n')

        # Points
        f.write('      <Points>\n')
        f.write('        <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        for j in range(n_eta):
            for i in range(n_xi):
                f.write(f'          {X[i, j]} {Y[i, j]} {Z[i, j]}\n')
        f.write('        </DataArray>\n')
        f.write('      </Points>\n')

        f.write('    </Piece>\n')
        f.write('  </StructuredGrid>\n')
        f.write('</VTKFile>\n')

    print(f"Exported VTK XML file: {path}")


def export_control_mesh_vtk(filename: str,
                             surface: NURBSSurface):
    """
    Export the control mesh (control points and their connectivity) to VTK.

    Useful for debugging and understanding the parametrization.

    Parameters:
        filename: Output filename
        surface: NURBS surface
    """
    path = Path(filename)
    if path.suffix != '.vtk':
        path = path.with_suffix('.vtk')

    control_points = surface.control_points
    n_xi, n_eta = surface.n_control_points_per_dir
    n_points = surface.n_control_points

    # For 2D control points, add z=0
    if control_points.shape[1] == 2:
        control_points_3d = np.zeros((n_points, 3))
        control_points_3d[:, :2] = control_points
    else:
        control_points_3d = control_points

    with open(path, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Control Mesh\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_GRID\n")
        f.write(f"DIMENSIONS {n_eta} {n_xi} 1\n")

        f.write(f"POINTS {n_points} float\n")
        for pt in control_points_3d:
            f.write(f"{pt[0]} {pt[1]} {pt[2]}\n")

        # Add weights as point data
        f.write(f"\nPOINT_DATA {n_points}\n")
        f.write("SCALARS weight float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for w in surface.weights:
            f.write(f"{w}\n")

    print(f"Exported control mesh: {path}")

"""
Visualization module for IGA.

Provides:
1. Export functionality for mesh data to binary format (XMF2/XDMF for ParaView)
2. THB basis function evaluation and visualization

Key classes:
- MeshExporter: High-level exporter for mesh data
- THBBasisVisualizer: Evaluate and visualize THB basis functions

Usage:
    from watfIGA.visualization import MeshExporter, generate_xmf

    # Export mesh and control point grid
    exporter = MeshExporter("MESH")
    exporter.export(mesh)
    generate_xmf("MESH", mode="cps")

    # Visualize THB basis functions
    from watfIGA.visualization import THBBasisVisualizer

    viz = THBBasisVisualizer(thb_surface, n_points=100)
    viz.plot_3d(save_path="basis_3d.png")
    print(f"Partition of unity: {viz.check_partition_of_unity()}")
"""

from .export import (
    export_mesh_data,
    export_iga_elements,
    MeshExporter,
)
from .xmf2 import generate_xmf, generate_thb_points_xmf
from .basis import (
    evaluate_bspline_basis_2d,
    evaluate_thb_basis,
    THBBasisVisualizer,
)

__all__ = [
    # Mesh export
    'export_mesh_data',
    'export_iga_elements',
    'MeshExporter',
    'generate_xmf',
    'generate_thb_points_xmf',
    # Basis visualization
    'evaluate_bspline_basis_2d',
    'evaluate_thb_basis',
    'THBBasisVisualizer',
]

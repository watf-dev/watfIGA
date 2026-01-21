"""
Export mesh data to binary format for visualization.

This module provides functions to export control points, connectivity,
and other mesh data to binary files that can be read by gen_xmf2.py
to generate XMF2 (XDMF) files for ParaView visualization.

File format conventions:
- All binary files use big-endian byte order
- Floating point data: float64
- Integer data: int32
- No file extensions (xyz, ien, etc.)

Directory structure:
    MESH/
        xyz         # Control point coordinates (nn x nsd, float64, big-endian)
        ien         # Element connectivity (ne x nen, int32, big-endian)
        weights     # Control point weights (nn, float64, big-endian) [optional]
    mesh.xmf        # XMF2 file (outside MESH folder)
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..discretization.mesh import Mesh
    from ..geometry.nurbs import NURBSSurface, NURBSCurve


def export_mesh_data(
    mesh: 'Mesh',
    output_dir: str | Path,
) -> Dict[str, Any]:
    """
    Export mesh data to binary files for XMF2 visualization.

    Creates the following structure:
        output_dir/
            xyz         # Coordinates (nn x 3, float64, big-endian) - always 3D
            ien         # Connectivity (ne x nen, int32, big-endian)
            weights     # Weights (nn, float64, big-endian)

    Parameters:
        mesh: Mesh object to export
        output_dir: Output directory (e.g., "MESH")

    Returns:
        Metadata dictionary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get control point data
    control_points = mesh.control_points_array
    weights = mesh.weights_array
    nn, nsd = control_points.shape

    # Pad to 3D for ParaView (XYZ geometry requires 3 components)
    if nsd < 3:
        xyz_3d = np.zeros((nn, 3), dtype=np.float64)
        xyz_3d[:, :nsd] = control_points
    else:
        xyz_3d = control_points

    # Export coordinates as float64 big-endian (always 3D)
    xyz_3d.astype('>f8').tofile(output_dir / "xyz")

    # Export weights as float64 big-endian
    weights.astype('>f8').tofile(output_dir / "weights")

    # Get connectivity data
    elements = mesh.get_active_elements_list()
    connectivity = [elem.control_point_ids for elem in elements]
    ne = len(connectivity)

    # Check if all elements have the same number of nodes
    # For THB meshes, elements may have different numbers of active basis functions
    if connectivity:
        nen_list = [len(c) for c in connectivity]
        nen_max = max(nen_list)
        nen_min = min(nen_list)

        if nen_max != nen_min:
            # Variable connectivity - pad with -1 to make rectangular
            padded_connectivity = []
            for conn in connectivity:
                padded = list(conn) + [-1] * (nen_max - len(conn))
                padded_connectivity.append(padded)
            connectivity = padded_connectivity
            nen = nen_max
        else:
            nen = nen_max
    else:
        nen = 0

    # Export connectivity as int32 big-endian
    conn_array = np.array(connectivity, dtype='>i4')
    conn_array.tofile(output_dir / "ien")

    # Build metadata
    metadata = {
        "nn": int(nn),      # Number of nodes (control points)
        "ne": int(ne),      # Number of elements
        "nen": int(nen),    # Nodes per element
        "nsd": int(nsd),    # Number of spatial dimensions
        "npd": mesh.n_dim_parametric,  # Number of parametric dimensions
        "mesh_dir": str(output_dir),
    }

    # Save metadata
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    return metadata


def export_cps_grid(
    n_xi: int,
    n_eta: int,
    output_dir: str | Path,
) -> Dict[str, Any]:
    """
    Export simple quad connectivity for control point grid visualization.

    Creates non-overlapping quads connecting neighboring control points.
    This is different from IGA elements which have overlapping support.

    For a grid of n_xi × n_eta control points:
    - Number of quads: (n_xi - 1) × (n_eta - 1)
    - Each quad has 4 nodes

    Parameters:
        n_xi: Number of control points in xi direction
        n_eta: Number of control points in eta direction
        output_dir: Output directory (e.g., "MESH")

    Returns:
        Metadata dictionary with grid info
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Number of simple quads
    ne_cps = (n_xi - 1) * (n_eta - 1)

    # Build connectivity for simple quads (4 nodes per element)
    # Using x-fastest ordering: cp_id = j * n_xi + i
    ien_cps = []
    for ej in range(n_eta - 1):
        for ei in range(n_xi - 1):
            # Four corners of the quad (counter-clockwise)
            n0 = ej * n_xi + ei           # bottom-left
            n1 = ej * n_xi + (ei + 1)     # bottom-right
            n2 = (ej + 1) * n_xi + (ei + 1)  # top-right
            n3 = (ej + 1) * n_xi + ei     # top-left
            ien_cps.append([n0, n1, n2, n3])

    # Export as int32 big-endian
    ien_cps_array = np.array(ien_cps, dtype='>i4')
    ien_cps_array.tofile(output_dir / "ien_cps")

    # Update metadata
    metadata_file = output_dir / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}

    metadata["n_xi"] = int(n_xi)
    metadata["n_eta"] = int(n_eta)
    metadata["ne_cps"] = int(ne_cps)
    metadata["nen_cps"] = 4

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    return metadata


class MeshExporter:
    """
    High-level mesh exporter for XMF2 visualization.

    Usage:
        # Basic export (control points and mesh data)
        exporter = MeshExporter("MESH")
        exporter.export(mesh)

        # With IGA element boundaries (requires geometry)
        exporter = MeshExporter("MESH")
        exporter.export(mesh, geometry=surface)
        exporter.export_iga_elements()

        # Chained
        exporter.export(mesh, surface).export_iga_elements()
    """

    def __init__(self, output_dir: str | Path):
        """
        Initialize exporter.

        Parameters:
            output_dir: Output directory path (e.g., "MESH")
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata: Dict[str, Any] = {}
        self._mesh: Optional['Mesh'] = None
        self._geometry = None  # NURBS/THB surface for element evaluation

    def export(self, mesh: 'Mesh', geometry=None) -> 'MeshExporter':
        """
        Export mesh data to binary files.

        Automatically exports:
        - Control point coordinates (xyz)
        - Weights
        - IGA element connectivity (ien)
        - CPS grid connectivity (ien_cps) if mesh has grid dimensions

        Parameters:
            mesh: Mesh object to export
            geometry: Optional geometry (NURBSSurface/THBSurface) for element export

        Returns:
            Self for chaining
        """
        self._mesh = mesh
        self._geometry = geometry

        self.metadata = export_mesh_data(mesh, self.output_dir)

        # Automatically export cps grid connectivity if mesh has grid dimensions
        if mesh.n_control_points_per_dir is not None:
            n_xi, n_eta = mesh.n_control_points_per_dir
            cps_metadata = export_cps_grid(n_xi, n_eta, self.output_dir)
            self.metadata.update(cps_metadata)

        return self

    def export_iga_elements(self, geometry=None) -> 'MeshExporter':
        """
        Export IGA element boundaries as quads for visualization.

        Each IGA element is mapped from parametric to physical space
        by evaluating the geometry at the 4 corners.

        Parameters:
            geometry: Optional geometry (uses stored geometry if not provided)

        Returns:
            Self for chaining

        Raises:
            ValueError: If no geometry available and mesh not exported yet
        """
        geom = geometry or self._geometry
        mesh = self._mesh

        if mesh is None:
            raise ValueError("Call export(mesh) first, or provide mesh")
        if geom is None:
            raise ValueError(
                "Geometry required for IGA element export. "
                "Pass geometry to export(mesh, geometry) or export_iga_elements(geometry)"
            )

        elem_metadata = export_iga_elements(geom, mesh, self.output_dir)
        self.metadata.update(elem_metadata)

        return self

    def export_cps_grid(self, n_xi: int, n_eta: int) -> 'MeshExporter':
        """
        Export simple quad connectivity for control point grid.

        Parameters:
            n_xi: Number of control points in xi direction
            n_eta: Number of control points in eta direction

        Returns:
            Self for chaining
        """
        cps_metadata = export_cps_grid(n_xi, n_eta, self.output_dir)
        self.metadata.update(cps_metadata)
        return self

    def export_field(
        self,
        name: str,
        values: np.ndarray,
    ) -> 'MeshExporter':
        """
        Export a scalar or vector field.

        Parameters:
            name: Field name (will be used as filename)
            values: Field values, shape (nn,) for scalar or (nn, n_components) for vector

        Returns:
            Self for chaining
        """
        values = np.asarray(values)

        # Save field data as float64 big-endian
        values.astype('>f8').tofile(self.output_dir / name)

        # Update metadata
        if "fields" not in self.metadata:
            self.metadata["fields"] = []

        field_info = {
            "name": name,
            "shape": list(values.shape),
        }
        self.metadata["fields"].append(field_info)

        # Update metadata file
        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(self.metadata, f, indent=2)

        return self

    def get_metadata(self) -> Dict[str, Any]:
        """Get the metadata dictionary."""
        return self.metadata

    @property
    def mesh(self) -> Optional['Mesh']:
        """Get the stored mesh."""
        return self._mesh

    @property
    def geometry(self):
        """Get the stored geometry."""
        return self._geometry


def export_iga_elements(
    surface,
    mesh: 'Mesh',
    output_dir: str | Path,
) -> Dict[str, Any]:
    """
    Export IGA elements as quads for ParaView visualization.

    Each IGA element is a rectangle in parametric space [xi_i, xi_{i+1}] × [eta_j, eta_{j+1}].
    We evaluate the NURBS/THB surface at the 4 corners to get physical coordinates.

    This creates element boundary visualization (different from control point grid).

    Parameters:
        surface: NURBS or THB surface with eval_point method
        mesh: IGA mesh
        output_dir: Output directory for binary files

    Returns:
        Metadata dictionary with element visualization info
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    elements = mesh.get_active_elements_list()
    n_elements = len(elements)

    # Collect element corner vertices
    # Each element has 4 corners in physical space
    vertices = []
    connectivity = []

    for elem in elements:
        xi_min, xi_max = elem.parametric_bounds[0]
        eta_min, eta_max = elem.parametric_bounds[1]

        # Evaluate surface at 4 corners (counter-clockwise)
        corners_param = [
            (xi_min, eta_min),   # bottom-left
            (xi_max, eta_min),   # bottom-right
            (xi_max, eta_max),   # top-right
            (xi_min, eta_max),   # top-left
        ]

        corner_indices = []
        for xi, eta in corners_param:
            pt = surface.eval_point((xi, eta))
            vertex_idx = len(vertices)
            vertices.append(pt)
            corner_indices.append(vertex_idx)

        connectivity.append(corner_indices)

    # Convert to arrays
    vertices = np.array(vertices, dtype=np.float64)
    connectivity = np.array(connectivity, dtype=np.int32)

    n_vertices = len(vertices)
    nsd = vertices.shape[1]

    # Pad to 3D for ParaView
    if nsd < 3:
        xyz_3d = np.zeros((n_vertices, 3), dtype=np.float64)
        xyz_3d[:, :nsd] = vertices
    else:
        xyz_3d = vertices

    # Export binary files
    xyz_3d.astype('>f8').tofile(output_dir / "elem_xyz")
    connectivity.astype('>i4').tofile(output_dir / "elem_ien")

    # Export element level as field (for THB visualization)
    elem_level = np.array([elem.level for elem in elements], dtype=np.float64)
    elem_level.astype('>f8').tofile(output_dir / "elem_level")

    # Build metadata
    elem_metadata = {
        "n_elem_vertices": int(n_vertices),
        "n_elements": int(n_elements),
        "nen_elem": 4,  # 4 corners per element
    }

    # Update existing metadata file if present
    metadata_file = output_dir / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        metadata.update(elem_metadata)
    else:
        metadata = elem_metadata

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    return elem_metadata


# Keep old functions for backward compatibility
def export_control_points(
    control_points: np.ndarray,
    weights: np.ndarray,
    output_dir: str | Path,
    name: str = "control_points"
) -> Dict[str, Any]:
    """
    Export control points and weights to binary files (backward compatibility).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nn, nsd = control_points.shape

    # Save as big-endian
    control_points.astype('>f8').tofile(output_dir / "xyz")
    weights.astype('>f8').tofile(output_dir / "weights")

    return {
        "nn": int(nn),
        "nsd": int(nsd),
    }

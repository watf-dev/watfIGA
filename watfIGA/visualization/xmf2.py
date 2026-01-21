"""
Generate XMF2 (XDMF) files from exported mesh data.

XMF2 is an XML-based format that describes data stored in binary files.
It can be opened directly in ParaView for visualization.

Modes:
    - "cps": Control point grid (Quadrilateral topology with ien_cps)
            In ParaView, you can switch between Points or Surface with Edges
    - "mesh": IGA mesh with connectivity (Quadrilateral topology with ien)

Usage:
    from watfIGA.visualization import generate_xmf
    generate_xmf("MESH", mode="cps")   # Control point grid
    generate_xmf("MESH", mode="mesh")  # IGA mesh with connectivity
"""

import json
import sys
from enum import Enum
from pathlib import Path
from typing import Optional
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom


class NumberType(Enum):
    Float = 0
    Int = 1


class TopologyType(Enum):
    PolyVertex = 1
    Polyline = 2
    Quadrilateral = 5


class GeometryType(Enum):
    XYZ = 0
    XY = 1


def _dims_to_string(dims):
    """Convert dimensions to string format."""
    if isinstance(dims, (list, tuple)):
        return " ".join(str(x) for x in dims)
    return str(dims)


def _create_dataitem(name, dimensions, number_type, precision, filepath):
    """Create a DataItem element."""
    item = Element("DataItem")
    item.set("Name", name)
    item.set("Dimensions", _dims_to_string(dimensions))
    item.set("Endian", "Big")
    item.set("Format", "Binary")
    item.set("Precision", str(precision))
    if number_type != NumberType.Float:
        item.set("NumberType", "Int")
    item.text = filepath
    return item


def _create_dataitem_reference(xpath):
    """Create a DataItem reference element."""
    item = Element("DataItem")
    item.set("Reference", "XML")
    item.text = xpath
    return item


def _generate_cps_xmf(metadata, mesh_dir, output_file):
    """
    Generate XMF2 file for control point grid visualization (Quadrilateral).

    Uses ien_cps connectivity (4-node quads connecting neighboring control points).
    In ParaView, you can switch representation between Points or Surface with Edges.

    Structure:
    <Grid>
      <Topology TopologyType="Quadrilateral" ...>
        <DataItem ...>path/to/ien_cps</DataItem>
      </Topology>
      <DataItem Name="xyz" ...>path/to/xyz</DataItem>
      <Geometry GeometryType="XYZ">
        <DataItem Reference="XML">/Xdmf/Domain/Grid/DataItem[@Name="xyz"]</DataItem>
      </Geometry>
    </Grid>
    """
    nn = metadata["nn"]
    nsd = metadata["nsd"]
    ne_cps = metadata["ne_cps"]
    nen_cps = metadata["nen_cps"]

    # Pad to 3D if needed
    if nsd == 2:
        gdim = [nn, 3]
    else:
        gdim = [nn, nsd]

    tdim = [ne_cps, nen_cps]

    # Build XMF structure
    xdmf = Element("Xdmf")
    xdmf.set("Version", "2.0")

    domain = SubElement(xdmf, "Domain")

    # Grid
    grid_name = Path(output_file).stem
    grid = SubElement(domain, "Grid")
    grid.set("Name", grid_name)
    grid.set("GridType", "Uniform")

    # Topology (Quadrilateral for quad mesh)
    topology = SubElement(grid, "Topology")
    topology.set("NumberOfElements", str(ne_cps))
    topology.set("TopologyType", "Quadrilateral")

    # DataItem for connectivity (inside Topology)
    mesh_name = Path(mesh_dir).name
    ien_cps_path = f"{mesh_name}/ien_cps"
    ien_cps_item = _create_dataitem("ien_cps", tdim, NumberType.Int, 4, ien_cps_path)
    topology.append(ien_cps_item)

    # DataItem for coordinates (comes before Geometry)
    xyz_path = f"{mesh_name}/xyz"
    xyz_item = _create_dataitem("xyz", gdim, NumberType.Float, 8, xyz_path)
    grid.append(xyz_item)

    # Geometry with reference
    geometry = SubElement(grid, "Geometry")
    geometry.set("GeometryType", "XYZ")
    xyz_ref = _create_dataitem_reference('/Xdmf/Domain/Grid/DataItem[@Name="xyz"]')
    geometry.append(xyz_ref)

    # Write output
    _write_xmf(xdmf, output_file)


def _generate_mesh_xmf(metadata, mesh_dir, output_file):
    """
    Generate XMF2 file for mesh visualization with connectivity.
    """
    nn = metadata["nn"]
    ne = metadata["ne"]
    nen = metadata["nen"]
    nsd = metadata["nsd"]
    npd = metadata["npd"]

    # Pad to 3D if needed
    if nsd == 2:
        gdim = [nn, 3]
    else:
        gdim = [nn, nsd]

    tdim = [ne, nen]

    # Determine topology type
    if npd == 1:
        topo_type = "Polyline"
    elif npd == 2:
        topo_type = "Quadrilateral"
    else:
        topo_type = "Hexahedron"

    # Build XMF structure
    xdmf = Element("Xdmf")
    xdmf.set("Version", "2.0")

    domain = SubElement(xdmf, "Domain")

    # Grid
    grid_name = Path(output_file).stem
    grid = SubElement(domain, "Grid")
    grid.set("Name", grid_name)
    grid.set("GridType", "Uniform")

    # Topology
    topology = SubElement(grid, "Topology")
    topology.set("NumberOfElements", str(ne))
    topology.set("TopologyType", topo_type)

    # DataItem for connectivity (inside Topology)
    mesh_name = Path(mesh_dir).name
    ien_path = f"{mesh_name}/ien"
    ien_item = _create_dataitem("ien", tdim, NumberType.Int, 4, ien_path)
    topology.append(ien_item)

    # DataItem for coordinates
    xyz_path = f"{mesh_name}/xyz"
    xyz_item = _create_dataitem("xyz", gdim, NumberType.Float, 8, xyz_path)
    grid.append(xyz_item)

    # Geometry with reference
    geometry = SubElement(grid, "Geometry")
    geometry.set("GeometryType", "XYZ")
    xyz_ref = _create_dataitem_reference('/Xdmf/Domain/Grid/DataItem[@Name="xyz"]')
    geometry.append(xyz_ref)

    # Write output
    _write_xmf(xdmf, output_file)


def _write_xmf(xdmf, filename):
    """Write XMF element to file with pretty printing."""
    rough_string = tostring(xdmf, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    with open(filename, "w") as f:
        f.write(reparsed.toprettyxml(indent="  "))
    print(f"Wrote to: {filename}")


def generate_xmf(
    input_dir,
    output_file: Optional[str] = None,
    mode: str = "cps",
) -> bool:
    """
    Generate XMF2 file from exported mesh data.

    Parameters:
        input_dir: Directory containing metadata.json and binary files (e.g., MESH)
        output_file: Output XMF file path (default: auto-generated based on mode)
        mode: "cps" for control points only, "mesh" for mesh with connectivity

    Returns:
        True if successful, False otherwise
    """
    input_dir = Path(input_dir)
    metadata_file = input_dir / "metadata.json"

    if not metadata_file.exists():
        print(f"Error: {metadata_file} not found", file=sys.stderr)
        return False

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    # Default output file is next to MESH folder
    if output_file is None:
        if mode == "cps":
            output_file = input_dir.parent / f"{input_dir.name.lower()}_cps.xmf2"
        elif mode == "elements":
            output_file = input_dir.parent / f"{input_dir.name.lower()}_elements.xmf2"
        else:
            output_file = input_dir.parent / f"{input_dir.name.lower()}_mesh.xmf2"
    else:
        output_file = Path(output_file)

    # Generate based on mode
    if mode == "cps":
        _generate_cps_xmf(metadata, str(input_dir), str(output_file))
    elif mode == "mesh":
        _generate_mesh_xmf(metadata, str(input_dir), str(output_file))
    elif mode == "elements":
        _generate_elements_xmf(metadata, str(input_dir), str(output_file))
    else:
        print(f"Error: Unknown mode '{mode}'. Use 'cps', 'mesh', or 'elements'.", file=sys.stderr)
        return False

    return True


def _generate_elements_xmf(metadata, mesh_dir, output_file):
    """
    Generate XMF2 file for IGA element visualization.

    Shows the actual element boundaries (quads in physical space),
    colored by refinement level for THB meshes.

    Parameters:
        metadata: Dictionary with n_elem_vertices, n_elements
        mesh_dir: Directory containing elem_xyz, elem_ien, elem_level
        output_file: Output XMF file path
    """
    n_vertices = metadata["n_elem_vertices"]
    n_elements = metadata["n_elements"]

    # Create root element
    xdmf = Element("Xdmf")
    xdmf.set("Version", "2.0")

    domain = SubElement(xdmf, "Domain")
    grid = SubElement(domain, "Grid")
    grid.set("Name", "iga_elements")
    grid.set("GridType", "Uniform")

    # Topology (Quadrilateral elements)
    topology = SubElement(grid, "Topology")
    topology.set("NumberOfElements", str(n_elements))
    topology.set("TopologyType", "Quadrilateral")

    topo_data = _create_dataitem(
        "elem_ien",
        (n_elements, 4),
        NumberType.Int,
        4,
        f"{mesh_dir}/elem_ien"
    )
    topology.append(topo_data)

    # Geometry (XYZ coordinates)
    xyz_data = _create_dataitem(
        "elem_xyz",
        (n_vertices, 3),
        NumberType.Float,
        8,
        f"{mesh_dir}/elem_xyz"
    )
    grid.append(xyz_data)

    geometry = SubElement(grid, "Geometry")
    geometry.set("GeometryType", "XYZ")
    geometry.append(_create_dataitem_reference(
        '/Xdmf/Domain/Grid/DataItem[@Name="elem_xyz"]'
    ))

    # Add level attribute (cell-centered for element coloring)
    attribute = SubElement(grid, "Attribute")
    attribute.set("Name", "level")
    attribute.set("AttributeType", "Scalar")
    attribute.set("Center", "Cell")

    level_data = _create_dataitem(
        "elem_level",
        (n_elements,),
        NumberType.Float,
        8,
        f"{mesh_dir}/elem_level"
    )
    attribute.append(level_data)

    # Write output
    _write_xmf(xdmf, output_file)


def generate_thb_points_xmf(mesh_dir: str, output_file: Optional[str] = None) -> bool:
    """
    Generate XMF2 file for THB mesh control points as a point cloud.

    Creates an XMF file with PolyVertex topology (points only) that can be
    opened in ParaView. The 'level' field allows coloring points by their
    refinement level.

    Parameters:
        mesh_dir: Directory containing metadata.json, xyz, and level files
        output_file: Output XMF file path (default: thb_points.xmf2 in parent dir)

    Returns:
        True if successful, False otherwise

    Example:
        >>> from watfIGA.visualization import generate_thb_points_xmf
        >>> generate_thb_points_xmf("MESH_THB")
        # Creates thb_points.xmf2 for visualization in ParaView
    """
    mesh_dir = Path(mesh_dir)
    metadata_file = mesh_dir / "metadata.json"

    if not metadata_file.exists():
        print(f"Error: {metadata_file} not found", file=sys.stderr)
        return False

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    nn = metadata["nn"]

    # Default output file
    if output_file is None:
        output_file = mesh_dir.parent / "thb_points.xmf2"
    else:
        output_file = Path(output_file)

    mesh_name = mesh_dir.name

    # Create XMF content
    xdmf = Element("Xdmf")
    xdmf.set("Version", "2.0")

    domain = SubElement(xdmf, "Domain")

    grid = SubElement(domain, "Grid")
    grid.set("Name", "thb_control_points")
    grid.set("GridType", "Uniform")

    # Topology (PolyVertex for point cloud)
    topology = SubElement(grid, "Topology")
    topology.set("NumberOfElements", str(nn))
    topology.set("TopologyType", "Polyvertex")

    # Geometry (XYZ coordinates)
    xyz_data = _create_dataitem(
        "xyz",
        (nn, 3),
        NumberType.Float,
        8,
        f"{mesh_name}/xyz"
    )
    grid.append(xyz_data)

    geometry = SubElement(grid, "Geometry")
    geometry.set("GeometryType", "XYZ")
    geometry.append(_create_dataitem_reference(
        '/Xdmf/Domain/Grid/DataItem[@Name="xyz"]'
    ))

    # Add level attribute (node-centered for point coloring)
    attribute = SubElement(grid, "Attribute")
    attribute.set("Name", "level")
    attribute.set("AttributeType", "Scalar")
    attribute.set("Center", "Node")

    level_data = _create_dataitem(
        "level",
        (nn,),
        NumberType.Float,
        8,
        f"{mesh_name}/level"
    )
    attribute.append(level_data)

    # Write output
    _write_xmf(xdmf, str(output_file))
    return True

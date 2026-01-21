"""
Primitive geometry factory functions.

This module provides factory functions for creating common NURBS geometries
used in IGA analysis, such as:
- Unit square and rectangles
- Circles and arcs
- Annular (ring) domains

These are the building blocks for more complex geometries.
"""

import numpy as np
from typing import Tuple

from .nurbs import NURBSCurve, NURBSSurface
from ..discretization.knot_vector import KnotVector, make_open_knot_vector


def make_nurbs_unit_square(p: int = 2, n_elem_xi: int = 4, n_elem_eta: int = 4,
                            physical_dim: int = 2) -> NURBSSurface:
    """
    Create a NURBS surface representing the unit square [0,1]².

    This is the simplest test geometry for IGA: identity mapping
    where parametric coordinates equal physical coordinates.

    Parameters:
        p: Polynomial degree in both directions
        n_elem_xi: Number of elements in xi direction
        n_elem_eta: Number of elements in eta direction
        physical_dim: 2 for 2D domain, 3 for surface in 3D (z=0)

    Returns:
        NURBSSurface representing the unit square
    """
    n_basis_xi = n_elem_xi + p
    n_basis_eta = n_elem_eta + p

    kv_xi = make_open_knot_vector(n_basis_xi, p, domain=(0.0, 1.0))
    kv_eta = make_open_knot_vector(n_basis_eta, p, domain=(0.0, 1.0))

    greville_xi = kv_xi.greville_abscissae()
    greville_eta = kv_eta.greville_abscissae()

    control_points = np.zeros((n_basis_xi * n_basis_eta, physical_dim))

    idx = 0
    for j in range(n_basis_eta):
        for i in range(n_basis_xi):
            control_points[idx, 0] = greville_xi[i]
            control_points[idx, 1] = greville_eta[j]
            if physical_dim == 3:
                control_points[idx, 2] = 0.0
            idx += 1

    weights = np.ones(n_basis_xi * n_basis_eta)

    return NURBSSurface(kv_xi, kv_eta, control_points, weights)


def make_nurbs_rectangle(x_range: Tuple[float, float] = (0.0, 1.0),
                          y_range: Tuple[float, float] = (0.0, 1.0),
                          p: int = 2,
                          n_elem_xi: int = 4,
                          n_elem_eta: int = 4) -> NURBSSurface:
    """
    Create a NURBS surface representing a rectangle.

    Parameters:
        x_range: (x_min, x_max)
        y_range: (y_min, y_max)
        p: Polynomial degree
        n_elem_xi: Number of elements in xi direction
        n_elem_eta: Number of elements in eta direction

    Returns:
        NURBSSurface representing the rectangle
    """
    surface = make_nurbs_unit_square(p, n_elem_xi, n_elem_eta, physical_dim=2)

    x_min, x_max = x_range
    y_min, y_max = y_range

    control_points = surface.control_points
    control_points[:, 0] = x_min + (x_max - x_min) * control_points[:, 0]
    control_points[:, 1] = y_min + (y_max - y_min) * control_points[:, 1]

    return NURBSSurface(
        surface.knot_vectors[0],
        surface.knot_vectors[1],
        control_points,
        surface.weights
    )


def make_nurbs_circle(radius: float = 1.0,
                       center: Tuple[float, float] = (0.0, 0.0)) -> NURBSCurve:
    """
    Create a NURBS curve representing a full circle.

    Uses the standard 9-control-point representation with degree 2.
    The circle is parameterized from 0 to 1, going counterclockwise
    starting from the positive x-axis.

    Parameters:
        radius: Circle radius
        center: Center coordinates (x, y)

    Returns:
        NURBSCurve representing the circle
    """
    # Degree 2, 9 control points for a full circle
    p = 2
    n_basis = 9

    # Knot vector with repeated knots at quarter points
    knots = np.array([0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1])
    kv = KnotVector(knots, p)

    # Control points at 0, 45, 90, 135, 180, 225, 270, 315, 360 degrees
    w = 1.0 / np.sqrt(2.0)  # Weight for 45-degree points

    angles = np.array([0, 45, 90, 135, 180, 225, 270, 315, 360]) * np.pi / 180
    control_points = np.zeros((n_basis, 2))
    weights = np.array([1, w, 1, w, 1, w, 1, w, 1])

    for i, angle in enumerate(angles):
        control_points[i, 0] = center[0] + radius * np.cos(angle)
        control_points[i, 1] = center[1] + radius * np.sin(angle)

    # Adjust control points at 45-degree positions (they lie outside the circle)
    # For a circle, the 45-degree control points are at radius/cos(45) = radius*sqrt(2)
    for i in [1, 3, 5, 7]:
        angle = angles[i]
        control_points[i, 0] = center[0] + radius * np.sqrt(2) * np.cos(angle)
        control_points[i, 1] = center[1] + radius * np.sqrt(2) * np.sin(angle)

    return NURBSCurve(kv, control_points, weights)


def make_nurbs_arc(radius: float = 1.0,
                    center: Tuple[float, float] = (0.0, 0.0),
                    start_angle: float = 0.0,
                    end_angle: float = np.pi / 2) -> NURBSCurve:
    """
    Create a NURBS curve representing a circular arc.

    Uses degree 2 with 3 control points for arcs up to 90 degrees.

    Parameters:
        radius: Arc radius
        center: Center coordinates (x, y)
        start_angle: Starting angle in radians
        end_angle: Ending angle in radians (must be within 90 degrees of start)

    Returns:
        NURBSCurve representing the arc
    """
    sweep = end_angle - start_angle
    if abs(sweep) > np.pi / 2 + 1e-10:
        raise ValueError("Arc sweep must be <= 90 degrees. Use make_nurbs_circle for larger arcs.")

    p = 2
    n_basis = 3

    knots = np.array([0, 0, 0, 1, 1, 1])
    kv = KnotVector(knots, p)

    # Weight for middle control point
    w = np.cos(sweep / 2)

    # Control points
    mid_angle = (start_angle + end_angle) / 2
    control_points = np.zeros((n_basis, 2))
    weights = np.array([1, w, 1])

    # Start point
    control_points[0, 0] = center[0] + radius * np.cos(start_angle)
    control_points[0, 1] = center[1] + radius * np.sin(start_angle)

    # End point
    control_points[2, 0] = center[0] + radius * np.cos(end_angle)
    control_points[2, 1] = center[1] + radius * np.sin(end_angle)

    # Middle control point (intersection of tangent lines)
    # Distance from center to middle control point
    d = radius / np.cos(sweep / 2)
    control_points[1, 0] = center[0] + d * np.cos(mid_angle)
    control_points[1, 1] = center[1] + d * np.sin(mid_angle)

    return NURBSCurve(kv, control_points, weights)


def make_nurbs_disk(radius: float = 1.0,
                     center: Tuple[float, float] = (0.0, 0.0),
                     p: int = 2,
                     n_elem_radial: int = 2,
                     n_elem_angular: int = 4) -> NURBSSurface:
    """
    Create a NURBS surface representing a circular disk.

    The disk is parameterized with:
    - xi (radial): 0 at center, 1 at boundary
    - eta (angular): 0 to 1 around the circle

    Parameters:
        radius: Disk radius
        center: Center coordinates (x, y)
        p: Polynomial degree
        n_elem_radial: Number of elements in radial direction
        n_elem_angular: Number of elements in angular direction (must be multiple of 4)

    Returns:
        NURBSSurface representing the disk
    """
    if n_elem_angular % 4 != 0:
        raise ValueError("n_elem_angular must be a multiple of 4")

    # Angular direction uses the circle representation
    n_basis_eta = n_elem_angular + p  # For periodic-like structure

    # For simplicity, use the quarter-circle structure repeated
    # This is a simplified version - a proper implementation would handle
    # the center singularity more carefully

    # Radial direction
    n_basis_xi = n_elem_radial + p
    kv_xi = make_open_knot_vector(n_basis_xi, p, domain=(0.0, 1.0))

    # Angular direction (periodic structure with repeated knots at quarters)
    n_quarters = n_elem_angular // 4
    n_basis_eta = 2 * n_elem_angular + 1

    # Build angular knot vector
    knots_eta = [0, 0, 0]
    for q in range(1, n_elem_angular):
        t = q / n_elem_angular
        knots_eta.extend([t, t])
    knots_eta.extend([1, 1, 1])
    knots_eta = np.array(knots_eta)
    kv_eta = KnotVector(knots_eta, p)

    n_basis_eta = kv_eta.n_basis

    # Create control points
    n_total = n_basis_xi * n_basis_eta
    control_points = np.zeros((n_total, 2))
    weights = np.ones(n_total)

    greville_xi = kv_xi.greville_abscissae()

    # Angular positions for control points
    w_corner = 1.0 / np.sqrt(2.0)

    idx = 0
    for j in range(n_basis_eta):
        # Angular position
        t_eta = j / (n_basis_eta - 1)
        angle = 2 * np.pi * t_eta

        # Determine if this is a corner (45, 135, 225, 315 degrees)
        is_diagonal = (j % 2 == 1) and (j < n_basis_eta - 1)

        for i in range(n_basis_xi):
            r = greville_xi[i] * radius

            if is_diagonal:
                # Diagonal control points need adjustment
                r_adj = r * np.sqrt(2)
                control_points[idx, 0] = center[0] + r_adj * np.cos(angle)
                control_points[idx, 1] = center[1] + r_adj * np.sin(angle)
                weights[idx] = w_corner
            else:
                control_points[idx, 0] = center[0] + r * np.cos(angle)
                control_points[idx, 1] = center[1] + r * np.sin(angle)
                weights[idx] = 1.0

            idx += 1

    return NURBSSurface(kv_xi, kv_eta, control_points, weights)


def make_nurbs_annulus(inner_radius: float = 0.5,
                        outer_radius: float = 1.0,
                        center: Tuple[float, float] = (0.0, 0.0),
                        p: int = 2,
                        n_elem_radial: int = 2,
                        n_elem_angular: int = 4) -> NURBSSurface:
    """
    Create a NURBS surface representing an annular (ring) domain.

    The annulus is parameterized with:
    - xi (radial): 0 at inner radius, 1 at outer radius
    - eta (angular): 0 to 1 around the ring (counterclockwise)

    This is useful for testing because it has curved boundaries
    but no singularity at the center.

    Parameters:
        inner_radius: Inner radius
        outer_radius: Outer radius
        center: Center coordinates (x, y)
        p: Polynomial degree
        n_elem_radial: Number of elements in radial direction
        n_elem_angular: Number of elements in angular direction

    Returns:
        NURBSSurface representing the annulus
    """
    # Radial direction
    n_basis_xi = n_elem_radial + p
    kv_xi = make_open_knot_vector(n_basis_xi, p, domain=(0.0, 1.0))

    # Angular direction - use circle-like structure
    # 9 control points per radial layer for full circle
    n_angular_cp = 9
    knots_eta = np.array([0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1])
    kv_eta = KnotVector(knots_eta, 2)  # Always degree 2 for circle

    n_basis_eta = kv_eta.n_basis

    # Create control points
    n_total = n_basis_xi * n_basis_eta
    control_points = np.zeros((n_total, 2))
    weights = np.ones(n_total)

    greville_xi = kv_xi.greville_abscissae()
    w_diag = 1.0 / np.sqrt(2.0)

    # Angles for 9 control points (0, 45, 90, ..., 360 degrees)
    angles = np.array([0, 45, 90, 135, 180, 225, 270, 315, 360]) * np.pi / 180

    idx = 0
    for j in range(n_basis_eta):
        angle = angles[j]
        is_diagonal = (j % 2 == 1)

        for i in range(n_basis_xi):
            # Interpolate radius
            t = greville_xi[i]
            r = inner_radius + t * (outer_radius - inner_radius)

            if is_diagonal:
                # Diagonal control points
                r_adj = r * np.sqrt(2)
                control_points[idx, 0] = center[0] + r_adj * np.cos(angle)
                control_points[idx, 1] = center[1] + r_adj * np.sin(angle)
                weights[idx] = w_diag
            else:
                control_points[idx, 0] = center[0] + r * np.cos(angle)
                control_points[idx, 1] = center[1] + r * np.sin(angle)
                weights[idx] = 1.0

            idx += 1

    return NURBSSurface(kv_xi, kv_eta, control_points, weights)

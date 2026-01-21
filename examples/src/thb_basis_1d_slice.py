#!/usr/bin/env python3
"""
Visualize THB basis functions as 1D slices.

This script cuts through the 2D parametric domain at a fixed eta value
to show how basis functions transition across the refined region boundary.

Created: 2025-01-21
Author: Wataru Fukuda
"""

import sys
import os
import argparse
import numpy as np

# Use Agg backend if --save is specified
if '--save' in sys.argv:
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from watfIGA.geometry.primitives import make_nurbs_unit_square
from watfIGA.geometry.thb import THBSurface
from watfIGA.visualization.basis import evaluate_thb_basis


def plot_1d_slice(thb, eta_value=0.0, n_points=200, save_path=None):
    """
    Plot basis functions along a 1D slice at fixed eta.

    Parameters:
        thb: THBSurface with refinement
        eta_value: Fixed eta value for the slice
        n_points: Number of evaluation points in xi
        save_path: If provided, save figure to this path
    """
    xi_grid = np.linspace(0, 1, n_points)
    # For 1D slice, we use a single eta value
    eta_grid = np.array([eta_value])

    # Get basis function categories
    truncated_l0 = thb.get_truncated_basis(0)
    active_l0 = thb.get_active_control_points(0)
    active_l1 = thb.get_active_control_points(1) if thb.n_levels > 1 else set()

    # Evaluate all basis functions
    l0_untrunc_data = []
    l0_trunc_data = []
    l1_data = []

    # Level 0 basis functions
    for idx in sorted(active_l0):
        N = evaluate_thb_basis(thb, 0, idx, xi_grid, eta_grid)
        N_1d = N[0, :]  # Extract 1D slice

        if idx in truncated_l0:
            l0_trunc_data.append((idx, N_1d))
        else:
            l0_untrunc_data.append((idx, N_1d))

    # Level 1 basis functions
    for idx in sorted(active_l1):
        N = evaluate_thb_basis(thb, 1, idx, xi_grid, eta_grid)
        N_1d = N[0, :]
        l1_data.append((idx, N_1d))

    # Compute sum
    basis_sum = np.zeros(n_points)
    for _, N in l0_untrunc_data + l0_trunc_data + l1_data:
        basis_sum += N

    # Create figure with subplots
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # Refined region boundary (for 4x4 mesh, elements 0,1 are refined = xi < 0.5)
    refined_boundary = 0.5

    # Plot 1: L0 Untruncated
    ax = axes[0]
    for idx, N in l0_untrunc_data:
        ax.plot(xi_grid, N, label=f'L0 #{idx}', alpha=0.7)
    ax.axvline(x=refined_boundary, color='red', linestyle='--', linewidth=2,
               label='Refined boundary')
    ax.axvspan(0, refined_boundary, alpha=0.1, color='red', label='Refined region')
    ax.set_ylabel('Basis value')
    ax.set_title(f'L0 Untruncated Basis Functions (eta={eta_value})')
    ax.legend(loc='upper right', fontsize=8, ncol=3)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)

    # Plot 2: L0 Truncated
    ax = axes[1]
    for idx, N in l0_trunc_data:
        ax.plot(xi_grid, N, label=f'L0 TRUNC #{idx}', linewidth=2)
    ax.axvline(x=refined_boundary, color='red', linestyle='--', linewidth=2)
    ax.axvspan(0, refined_boundary, alpha=0.1, color='red')
    ax.set_ylabel('Basis value')
    ax.set_title(f'L0 Truncated Basis Functions - ZERO inside refined region (eta={eta_value})')
    ax.legend(loc='upper right', fontsize=8, ncol=3)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)

    # Plot 3: L1 Active
    ax = axes[2]
    for idx, N in l1_data:
        ax.plot(xi_grid, N, label=f'L1 #{idx}', alpha=0.7)
    ax.axvline(x=refined_boundary, color='red', linestyle='--', linewidth=2)
    ax.axvspan(0, refined_boundary, alpha=0.1, color='red')
    ax.set_ylabel('Basis value')
    ax.set_title(f'L1 Active Basis Functions - only inside refined region (eta={eta_value})')
    ax.legend(loc='upper right', fontsize=8, ncol=4)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)

    # Plot 4: Sum (Partition of Unity)
    ax = axes[3]
    ax.plot(xi_grid, basis_sum, 'k-', linewidth=2, label='Sum of all basis')
    ax.axvline(x=refined_boundary, color='red', linestyle='--', linewidth=2)
    ax.axvspan(0, refined_boundary, alpha=0.1, color='red')
    ax.set_xlabel('xi')
    ax.set_ylabel('Sum')
    ax.set_title(f'Partition of Unity: sum = {basis_sum.min():.6f} to {basis_sum.max():.6f}')
    ax.set_ylim(0.9, 1.1)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_selected_basis(thb, basis_indices, eta_value=0.0, n_points=200, save_path=None):
    """
    Plot selected basis functions to highlight truncation effect.

    Parameters:
        thb: THBSurface
        basis_indices: List of (level, idx) tuples to plot
        eta_value: Fixed eta value
        n_points: Number of points
        save_path: Save path
    """
    xi_grid = np.linspace(0, 1, n_points)
    eta_grid = np.array([eta_value])

    truncated_l0 = thb.get_truncated_basis(0)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Refined boundary
    refined_boundary = 0.5
    ax.axvline(x=refined_boundary, color='red', linestyle='--', linewidth=2,
               label='Refined boundary (xi=0.5)')
    ax.axvspan(0, refined_boundary, alpha=0.15, color='red', label='Refined region')

    colors = plt.cm.tab10(np.linspace(0, 1, len(basis_indices)))

    for (level, idx), color in zip(basis_indices, colors):
        N = evaluate_thb_basis(thb, level, idx, xi_grid, eta_grid)
        N_1d = N[0, :]

        if level == 0:
            if idx in truncated_l0:
                label = f'L0 TRUNC #{idx}'
                linestyle = '-'
                linewidth = 2.5
            else:
                label = f'L0 #{idx}'
                linestyle = '--'
                linewidth = 1.5
        else:
            label = f'L1 #{idx}'
            linestyle = ':'
            linewidth = 1.5

        ax.plot(xi_grid, N_1d, label=label, color=color,
                linestyle=linestyle, linewidth=linewidth)

    ax.set_xlabel('xi', fontsize=12)
    ax.set_ylabel('Basis function value', fontsize=12)
    ax.set_title(f'THB Basis Functions at eta={eta_value}\n'
                 f'Truncated L0 functions are ZERO inside refined region', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)

    # Add annotations
    ax.annotate('Truncated L0 = 0\nhere', xy=(0.2, 0.05), fontsize=10,
                ha='center', color='darkred')
    ax.annotate('L1 functions\nactive here', xy=(0.25, 0.7), fontsize=10,
                ha='center', color='darkblue')
    ax.annotate('L0 functions\nactive here', xy=(0.75, 0.7), fontsize=10,
                ha='center', color='darkgreen')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(description="1D slice of THB basis functions")
    parser.add_argument("--eta", type=float, default=0.125,
                        help="Eta value for the slice (default: 0.125)")
    parser.add_argument("--n-points", type=int, default=200,
                        help="Number of evaluation points")
    parser.add_argument("--save", action="store_true",
                        help="Save figures to files")
    args = parser.parse_args()

    print("=" * 60)
    print("THB Basis Function 1D Slice Visualization")
    print("=" * 60)

    # Create THB surface
    print("\n1. Creating THB surface (4x4 elements, p=2)...")
    surface = make_nurbs_unit_square(p=2, n_elem_xi=4, n_elem_eta=4)
    thb = THBSurface.from_nurbs_surface(surface)

    # Refine corner region
    print("\n2. Refining elements (0,0), (1,0), (0,1)...")
    for ei, ej in [(0, 0), (1, 0), (0, 1)]:
        thb.refine_element(0, ei, ej)
    thb.finalize_refinement()

    print(f"   Active L0: {len(thb.get_active_control_points(0))}")
    print(f"   Truncated L0: {len(thb.get_truncated_basis(0))}")
    print(f"   Active L1: {len(thb.get_active_control_points(1))}")
    print(f"   Truncated indices: {sorted(thb.get_truncated_basis(0))}")

    # Plot all basis functions
    print(f"\n3. Plotting 1D slice at eta={args.eta}...")
    save_all = "thb_basis_1d_all.png" if args.save else None
    plot_1d_slice(thb, eta_value=args.eta, n_points=args.n_points, save_path=save_all)

    # Plot selected basis functions for clarity
    print("\n4. Plotting selected basis functions...")
    # Select a few interesting basis functions:
    # - Some truncated L0 (indices 2, 3, 8, 9 based on typical layout)
    # - Some L1 functions
    truncated = sorted(thb.get_truncated_basis(0))[:4]
    l1_active = sorted(thb.get_active_control_points(1))[:4]

    selected = [(0, idx) for idx in truncated[:3]] + [(1, idx) for idx in l1_active[:3]]

    save_selected = "thb_basis_1d_selected.png" if args.save else None
    plot_selected_basis(thb, selected, eta_value=args.eta,
                        n_points=args.n_points, save_path=save_selected)

    print("\n" + "=" * 60)

    if not args.save:
        plt.show()


if __name__ == "__main__":
    main()
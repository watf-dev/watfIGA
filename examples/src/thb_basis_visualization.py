#!/usr/bin/env python3
"""
Visualize THB basis functions over the parametric domain.

This script demonstrates how to use THBBasisVisualizer to verify:
1. Untruncated L0 basis: Full B-spline shape outside refined region
2. Truncated L0 basis: Zero inside refined region, nonzero outside
3. L1 basis: Active only inside refined region
4. Partition of unity: Sum of all basis functions = 1 everywhere

Created: 2025-01-20
Author: Wataru Fukuda
"""

import sys
import argparse
import os

# Use Agg backend if --save is specified (non-interactive)
if '--save' in sys.argv:
    import matplotlib
    matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from watfIGA.geometry.primitives import make_nurbs_unit_square
from watfIGA.geometry.thb import THBSurface
from watfIGA.visualization import THBBasisVisualizer


def main():
    parser = argparse.ArgumentParser(description="Visualize THB basis functions")
    parser.add_argument("--n-points", type=int, default=100,
                        help="Evaluation points per direction")
    parser.add_argument("--save", action="store_true",
                        help="Save figures to files")
    parser.add_argument("--all", action="store_true",
                        help="Plot ALL basis functions (not just representative)")
    parser.add_argument("--export-data", type=str, default=None,
                        help="Export basis data to specified directory")
    args = parser.parse_args()

    print("=" * 60)
    print("THB Basis Function Visualization")
    print("=" * 60)

    # Create THB surface with refinement
    print("\n1. Creating THB surface...")
    surface = make_nurbs_unit_square(p=2, n_elem_xi=4, n_elem_eta=4)
    thb = THBSurface.from_nurbs_surface(surface)

    # Refine corner region
    print("\n2. Refining elements (0,0), (1,0), (0,1)...")
    for ei, ej in [(0, 0), (1, 0), (0, 1)]:
        thb.refine_element(0, ei, ej)
    thb.finalize_refinement()

    print(f"   Active L0 basis: {len(thb.get_active_control_points(0))}")
    print(f"   Truncated L0:    {len(thb.get_truncated_basis(0))}")
    print(f"   Active L1 basis: {len(thb.get_active_control_points(1))}")

    # Create visualizer
    print(f"\n3. Creating visualizer with {args.n_points}x{args.n_points} grid...")
    viz = THBBasisVisualizer(thb, n_points=args.n_points)

    # Print basis info
    info = viz.get_basis_info()
    print(f"   Active basis functions:")
    print(f"     L0 untruncated: {info['l0_untruncated']}")
    print(f"     L0 truncated:   {info['l0_truncated']}")
    print(f"     L1 active:      {info['l1_active']}")
    print(f"     Total:          {info['total']}")

    # Check partition of unity
    print("\n4. Partition of unity check:")
    stats = viz.get_partition_of_unity_stats()
    print(f"   Min sum:  {stats['min']:.6f}")
    print(f"   Max sum:  {stats['max']:.6f}")
    print(f"   Mean sum: {stats['mean']:.6f}")

    if viz.check_partition_of_unity():
        print("   PASSED!")
    else:
        print("   FAILED!")

    # Export data if requested
    if args.export_data:
        print(f"\n5. Exporting basis data to {args.export_data}...")
        viz.export_basis_data(args.export_data)
        print("   Done.")

    # Plot 3D surfaces
    print("\n6. Creating 3D visualization...")
    save_3d = "thb_basis_3d.png" if args.save else None
    viz.plot_3d(plot_all=args.all, save_path=save_3d, show=not args.save)
    if save_3d:
        print(f"   Saved: {save_3d}")

    # Plot heatmaps of truncated basis
    print("\n7. Creating heatmap visualization of truncated basis...")
    save_heatmap = "thb_basis_heatmaps.png" if args.save else None
    # viz.plot_heatmaps(save_path=save_heatmap, show=not args.save)
    viz.plot_heatmaps(plot_all=args.all, save_path=save_heatmap, show=not args.save)
    if save_heatmap:
        print(f"   Saved: {save_heatmap}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

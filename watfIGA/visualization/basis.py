"""
THB basis function evaluation and visualization.

This module provides functions to:
1. Evaluate B-spline and THB basis functions over parametric grids
2. Visualize basis functions as 3D surfaces or 2D heatmaps
3. Check partition of unity

The evaluation functions have no matplotlib dependency and can be used
independently for numerical analysis.

Example:
    from watfIGA.visualization.basis import THBBasisVisualizer

    # Create visualizer
    viz = THBBasisVisualizer(thb_surface, n_points=100)

    # Evaluate and check partition of unity
    basis_sum = viz.compute_basis_sum()
    assert viz.check_partition_of_unity()

    # Plot and save
    viz.plot_3d(save_path="thb_basis_3d.png")
    viz.plot_heatmaps(save_path="thb_basis_heatmaps.png")
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
import numpy as np

if TYPE_CHECKING:
    from ..geometry.thb import THBSurface
    from ..discretization.knot_vector import KnotVector

__all__ = [
    'evaluate_bspline_basis_2d',
    'evaluate_thb_basis',
    'THBBasisVisualizer',
]


def evaluate_bspline_basis_2d(
    kv_xi: 'KnotVector',
    kv_eta: 'KnotVector',
    i: int,
    j: int,
    xi_grid: np.ndarray,
    eta_grid: np.ndarray
) -> np.ndarray:
    """
    Evaluate a 2D B-spline basis function N_{i,j} over a grid.

    The 2D basis function is the tensor product: N_{i,j}(xi, eta) = N_i(xi) * N_j(eta)

    Parameters:
        kv_xi: Knot vector in xi direction
        kv_eta: Knot vector in eta direction
        i: Basis function index in xi direction
        j: Basis function index in eta direction
        xi_grid: 1D array of evaluation points in xi
        eta_grid: 1D array of evaluation points in eta

    Returns:
        2D array of basis values, shape (len(eta_grid), len(xi_grid))
    """
    from ..geometry.bspline import eval_basis_ders_1d

    p_xi = kv_xi.degree
    p_eta = kv_eta.degree

    # Evaluate 1D basis functions
    N_xi = np.zeros(len(xi_grid))
    N_eta = np.zeros(len(eta_grid))

    for k, xi in enumerate(xi_grid):
        # Clamp to domain
        xi = max(kv_xi.domain[0], min(xi, kv_xi.domain[1] - 1e-10))
        span = kv_xi.find_span(xi)

        # Check if basis i is active at this span
        if span - p_xi <= i <= span:
            N_all = eval_basis_ders_1d(kv_xi, xi, 0, span)[0, :]
            local_idx = i - (span - p_xi)
            N_xi[k] = N_all[local_idx]

    for k, eta in enumerate(eta_grid):
        eta = max(kv_eta.domain[0], min(eta, kv_eta.domain[1] - 1e-10))
        span = kv_eta.find_span(eta)

        if span - p_eta <= j <= span:
            N_all = eval_basis_ders_1d(kv_eta, eta, 0, span)[0, :]
            local_idx = j - (span - p_eta)
            N_eta[k] = N_all[local_idx]

    # Tensor product
    return np.outer(N_eta, N_xi)


def evaluate_thb_basis(
    thb: 'THBSurface',
    level: int,
    local_idx: int,
    xi_grid: np.ndarray,
    eta_grid: np.ndarray
) -> np.ndarray:
    """
    Evaluate a THB basis function over a grid.

    For truncated basis functions, applies the truncation formula:
        trunc(N_i^l) = N_i^l - sum_j c_ij * N_j^{l+1}

    where c_ij are the truncation coefficients.

    Parameters:
        thb: THBSurface with refinement applied
        level: Level of the basis function
        local_idx: Local index at that level
        xi_grid: 1D array of evaluation points in xi
        eta_grid: 1D array of evaluation points in eta

    Returns:
        2D array of basis values, shape (len(eta_grid), len(xi_grid))
    """
    kv_xi, kv_eta = thb.get_knot_vectors(level)
    n_xi = kv_xi.n_basis

    i = local_idx % n_xi
    j = local_idx // n_xi

    # Evaluate the base B-spline
    N_base = evaluate_bspline_basis_2d(kv_xi, kv_eta, i, j, xi_grid, eta_grid)

    # Check if truncated
    if not thb.is_basis_truncated(level, local_idx):
        return N_base

    # Apply truncation: subtract fine-level contributions
    trunc_coeffs = thb.get_truncation_coefficients(level, local_idx)

    N_truncated = N_base.copy()
    for (fine_level, fine_idx), coeff in trunc_coeffs.items():
        kv_xi_fine, kv_eta_fine = thb.get_knot_vectors(fine_level)
        n_xi_fine = kv_xi_fine.n_basis

        i_fine = fine_idx % n_xi_fine
        j_fine = fine_idx // n_xi_fine

        N_fine = evaluate_bspline_basis_2d(
            kv_xi_fine, kv_eta_fine, i_fine, j_fine, xi_grid, eta_grid
        )
        N_truncated -= coeff * N_fine

    return N_truncated


class THBBasisVisualizer:
    """
    Visualizer for THB basis functions.

    Provides methods to evaluate, analyze, and plot THB basis functions
    over the parametric domain.

    Example:
        thb = THBSurface.from_nurbs_surface(surface)
        thb.refine_element(0, 0, 0)
        thb.finalize_refinement()

        viz = THBBasisVisualizer(thb, n_points=100)

        # Check partition of unity
        if viz.check_partition_of_unity():
            print("Partition of unity verified!")

        # Plot basis functions
        viz.plot_3d(save_path="basis_3d.png")
    """

    def __init__(self, thb: 'THBSurface', n_points: int = 100):
        """
        Initialize the visualizer.

        Parameters:
            thb: THBSurface with refinement applied
            n_points: Number of evaluation points per direction
        """
        self.thb = thb
        self.n_points = n_points

        # Create evaluation grid
        self.xi_grid = np.linspace(0, 1, n_points)
        self.eta_grid = np.linspace(0, 1, n_points)
        self.XI, self.ETA = np.meshgrid(self.xi_grid, self.eta_grid)

        # Categorize basis functions
        self._categorize_basis_functions()

        # Cache for evaluated basis functions
        self._basis_values: Optional[List[Tuple[int, int, np.ndarray]]] = None
        self._basis_sum: Optional[np.ndarray] = None

    def _categorize_basis_functions(self) -> None:
        """Categorize basis functions by level and truncation status."""
        self.l0_untruncated: List[Tuple[int, int]] = []
        self.l0_truncated: List[Tuple[int, int]] = []
        self.l1_active: List[Tuple[int, int]] = []

        # Level 0
        truncated_l0 = self.thb.get_truncated_basis(0)
        for idx in sorted(self.thb.get_active_control_points(0)):
            if idx in truncated_l0:
                self.l0_truncated.append((0, idx))
            else:
                self.l0_untruncated.append((0, idx))

        # Level 1 (if exists)
        if self.thb.n_levels > 1:
            for idx in sorted(self.thb.get_active_control_points(1)):
                self.l1_active.append((1, idx))

    @property
    def n_basis_functions(self) -> int:
        """Total number of active basis functions."""
        return len(self.l0_untruncated) + len(self.l0_truncated) + len(self.l1_active)

    def get_basis_info(self) -> Dict[str, int]:
        """Get summary of basis function counts."""
        return {
            'l0_untruncated': len(self.l0_untruncated),
            'l0_truncated': len(self.l0_truncated),
            'l1_active': len(self.l1_active),
            'total': self.n_basis_functions,
        }

    def evaluate_all(self) -> List[Tuple[int, int, np.ndarray]]:
        """
        Evaluate all active basis functions.

        Returns:
            List of (level, local_idx, values) tuples
        """
        if self._basis_values is not None:
            return self._basis_values

        all_basis = self.l0_untruncated + self.l0_truncated + self.l1_active
        self._basis_values = []

        for level, idx in all_basis:
            N = evaluate_thb_basis(self.thb, level, idx, self.xi_grid, self.eta_grid)
            self._basis_values.append((level, idx, N))

        return self._basis_values

    def compute_basis_sum(self) -> np.ndarray:
        """
        Compute sum of all basis functions (for partition of unity check).

        Returns:
            2D array of summed values
        """
        if self._basis_sum is not None:
            return self._basis_sum

        basis_values = self.evaluate_all()
        self._basis_sum = np.zeros((self.n_points, self.n_points))

        for level, idx, N in basis_values:
            self._basis_sum += N

        return self._basis_sum

    def check_partition_of_unity(self, atol: float = 1e-10) -> bool:
        """
        Check if partition of unity holds (sum of all basis = 1).

        Parameters:
            atol: Absolute tolerance for comparison

        Returns:
            True if partition of unity holds
        """
        basis_sum = self.compute_basis_sum()
        return np.allclose(basis_sum, 1.0, atol=atol)

    def get_partition_of_unity_stats(self) -> Dict[str, float]:
        """Get statistics about the partition of unity."""
        basis_sum = self.compute_basis_sum()
        return {
            'min': float(basis_sum.min()),
            'max': float(basis_sum.max()),
            'mean': float(basis_sum.mean()),
        }

    def plot_3d(
        self,
        plot_all: bool = True,
        plot_sum: bool = True,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Plot basis functions as 3D surfaces.

        Parameters:
            plot_all: If True, plot all basis functions; otherwise representative subset
            plot_sum: Whether to include partition of unity sum plot
            save_path: If provided, save figure to this path
            show: Whether to call plt.show()

        Returns:
            matplotlib Figure object
        """
        import matplotlib.pyplot as plt

        basis_values = self.evaluate_all()
        basis_sum = self.compute_basis_sum()
        truncated_l0_indices = {idx for _, idx in self.l0_truncated}

        # Decide what to plot
        if plot_all:
            to_plot = [(level, idx) for level, idx, _ in basis_values]
        else:
            to_plot = []
            to_plot.extend(self.l0_untruncated[:3])
            to_plot.extend(self.l0_truncated[:4])
            to_plot.extend(self.l1_active[:5])

        # Create figure
        n_cols = 6 if plot_all else 4
        n_basis_to_plot = len(to_plot)
        n_rows = (n_basis_to_plot + n_cols) // n_cols

        fig = plt.figure(figsize=(3 * n_cols, 3 * n_rows))

        plot_idx = 1
        for level, idx, N in basis_values:
            if (level, idx) not in to_plot:
                continue

            ax = fig.add_subplot(n_rows, n_cols, plot_idx, projection='3d')

            # Determine title
            if level == 0:
                if idx in truncated_l0_indices:
                    title = f"L0 TRUNC #{idx}"
                else:
                    title = f"L0 #{idx}"
            else:
                title = f"L1 #{idx}"

            ax.plot_surface(self.XI, self.ETA, N, cmap='viridis', alpha=0.8)
            ax.set_xlabel('xi', fontsize=8)
            ax.set_ylabel('eta', fontsize=8)
            ax.set_title(title, fontsize=9)
            ax.set_zlim(-0.1, 1.1)
            ax.tick_params(labelsize=6)

            plot_idx += 1

        # Plot sum
        if plot_sum and plot_idx <= n_rows * n_cols:
            ax = fig.add_subplot(n_rows, n_cols, plot_idx, projection='3d')
            ax.plot_surface(self.XI, self.ETA, basis_sum, cmap='coolwarm', alpha=0.8)
            ax.set_xlabel('xi', fontsize=8)
            ax.set_ylabel('eta', fontsize=8)
            ax.set_title(
                f'SUM\nmin={basis_sum.min():.4f}\nmax={basis_sum.max():.4f}',
                fontsize=9
            )
            ax.set_zlim(0.8, 1.2)
            ax.tick_params(labelsize=6)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()

        return fig

    def plot_heatmaps(
        self,
        plot_all: bool = True,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Plot basis functions as 2D heatmaps.

        Parameters:
            plot_all: If True, plot all basis functions;
                      otherwise plot truncated L0 basis only
            save_path: If provided, save figure to this path
            show: Whether to call plt.show()

        Returns:
            matplotlib Figure object, or None if no basis to plot
        """
        import matplotlib.pyplot as plt
        import numpy as np

        basis_values = self.evaluate_all()
        truncated_l0_indices = {idx for _, idx in self.l0_truncated}

        # Decide what to plot
        if plot_all:
            to_plot = [(f"L{level}#{idx}", N) for level, idx, N in basis_values]
            title = 'All Active Basis Functions'
        else:
            to_plot = [
                (idx, N)
                for level, idx, N in basis_values
                if level == 0 and idx in truncated_l0_indices
            ]
            title = 'Truncated L0 Basis Functions'

        if not to_plot:
            return None

        # Layout
        n_cols = 4
        n_rows = (len(to_plot) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        axes = np.atleast_2d(axes)

        for i, (label, N) in enumerate(to_plot):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col]

            im = ax.imshow(
                N,
                extent=[0, 1, 0, 1],
                origin='lower',
                # cmap='RdYlBu_r',
                cmap='inferno',
                vmin=-0.1,
                vmax=1.0,
                aspect='equal'
            )
            ax.set_xlabel('xi')
            ax.set_ylabel('eta')

            if plot_all:
                ax.set_title(str(label))
            else:
                ax.set_title(f'Truncated L0 #{label}')

            plt.colorbar(im, ax=ax, shrink=0.8)

        # Hide unused axes
        for i in range(len(to_plot), n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].set_visible(False)

        fig.suptitle(title, fontsize=14)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()

        return fig

    def export_basis_data(self, output_dir: str) -> None:
        """
        Export basis function data to numpy files.

        Creates:
            - basis_values.npz: All basis function values
            - basis_sum.npy: Partition of unity sum
            - basis_info.json: Metadata

        Parameters:
            output_dir: Directory to save files
        """
        import json
        from pathlib import Path

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Evaluate all basis functions
        basis_values = self.evaluate_all()
        basis_sum = self.compute_basis_sum()

        # Save basis values
        data = {
            f"L{level}_{idx}": N
            for level, idx, N in basis_values
        }
        np.savez(output_dir / "basis_values.npz", **data)

        # Save sum
        np.save(output_dir / "basis_sum.npy", basis_sum)

        # Save grid
        np.savez(output_dir / "grid.npz", xi=self.xi_grid, eta=self.eta_grid)

        # Save metadata
        info = self.get_basis_info()
        info['n_points'] = self.n_points
        info['partition_of_unity'] = self.get_partition_of_unity_stats()

        with open(output_dir / "basis_info.json", 'w') as f:
            json.dump(info, f, indent=2)

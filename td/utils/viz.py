"""Energy landscape visualization for MHN debugging.

Optional — requires matplotlib. Only used for development/debugging.
"""

from __future__ import annotations

import numpy as np


def plot_energy_landscape(mhn, query: np.ndarray | None = None,
                          save_path: str | None = None):
    """Visualize the MHN energy landscape.

    Shows pairwise similarities between stored patterns and optionally
    the query position.

    Args:
        mhn: ModernHopfieldNetwork instance.
        query: Optional query vector to highlight.
        save_path: If provided, save figure to this path.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping visualization")
        return

    if not mhn.patterns:
        print("No patterns stored — nothing to visualize")
        return

    active = [p for p in mhn.patterns if p.active]
    n = len(active)
    if n == 0:
        print("No active patterns — nothing to visualize")
        return

    # Similarity matrix
    composites = np.stack([p.composite.astype(np.float32) for p in active])
    sim_matrix = composites @ composites.T / mhn.config.dim

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Heatmap
    im = ax1.imshow(sim_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax1.set_title("Pattern Similarity Matrix")
    ax1.set_xlabel("Pattern Index")
    ax1.set_ylabel("Pattern Index")
    plt.colorbar(im, ax=ax1, label="Similarity")

    # Query similarities
    if query is not None:
        query_f = query.astype(np.float32)
        sims = composites @ query_f / mhn.config.dim
        colors = ["green" if s > 0.3 else "gray" for s in sims]
        ax2.barh(range(n), sims, color=colors)
        ax2.set_xlabel("Similarity to Query")
        ax2.set_ylabel("Pattern Index")
        ax2.set_title("Query → Pattern Similarities")
        ax2.axvline(x=0.3, color="red", linestyle="--", label="Threshold")
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "No query provided",
                ha="center", va="center", transform=ax2.transAxes)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()

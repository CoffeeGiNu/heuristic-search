from typing import Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib import patches

from ..problems.grid_path_finding import GridAction, GridPathFinding, GridState


# ----------------------------------------------------------------------
# Utility: high‑quality static image of the path
# ----------------------------------------------------------------------
def visualize_grid_path(
    grid_path_finding: GridPathFinding,
    path_with_actions: Sequence[tuple[GridState, Optional[GridAction]]],
    filename: str = "path.png",
    *,
    figsize: tuple[float, float] = (6, 6),
) -> str:
    """
    Render the path on a grid and save to an image file.

    Parameters
    ----------
    grid_path_finding : GridPathFinding
        The problem instance (provides shape / start / goal).
    path_with_actions : Sequence[tuple[GridState, Optional[GridAction]]]
        A sequence of (state, action) pairs describing the path.
    filename : str, default "path.png"
        Output image filename (PNG).
    figsize : (float, float), optional
        Matplotlib figure size in inches.

    Returns
    -------
    str
        The filename that was written.
    """
    # --- setup figure --------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    width, height = grid_path_finding.shape

    # invert y axis so (0,0) is top‑left like array indexing
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)

    # --- draw grid -----------------------------------------------------
    for x in range(width + 1):
        ax.axvline(x - 0.5, color="lightgray", linewidth=0.8, zorder=0)
    for y in range(height + 1):
        ax.axhline(y - 0.5, color="lightgray", linewidth=0.8, zorder=0)

    # --- draw start & goal --------------------------------------------
    sx, sy = grid_path_finding.initial_position
    gx, gy = grid_path_finding.goal_position
    ax.add_patch(
        patches.Rectangle(
            (sx - 0.5, sy - 0.5),
            1,
            1,
            facecolor="#3cb44b",
            alpha=0.4,
            edgecolor="black",
            linewidth=1.0,
            label="Start",
            zorder=2,
        )
    )
    ax.add_patch(
        patches.Rectangle(
            (gx - 0.5, gy - 0.5),
            1,
            1,
            facecolor="#e6194B",
            alpha=0.4,
            edgecolor="black",
            linewidth=1.0,
            label="Goal",
            zorder=2,
        )
    )

    # --- extract coordinates ------------------------------------------
    states = [s for s, _ in path_with_actions]
    xs = [s.position[0] for s in states]
    ys = [s.position[1] for s in states]

    # --- plot path line & markers -------------------------------------
    if len(xs) > 1:
        ax.plot(xs, ys, color="#4363d8", linewidth=2.0, zorder=3, label="Path")
    ax.scatter(xs, ys, s=60, color="#4363d8", zorder=4)

    # step numbers
    for idx, (x, y) in enumerate(zip(xs, ys)):
        ax.text(
            x,
            y,
            str(idx),
            color="white",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            zorder=5,
        )

    # --- draw movement arrows -----------------------------------------
    for (cur, _), (nxt, _) in zip(path_with_actions[:-1], path_with_actions[1:]):
        dx = nxt.position[0] - cur.position[0]
        dy = nxt.position[1] - cur.position[1]
        mid_x = (cur.position[0] + nxt.position[0]) / 2
        mid_y = (cur.position[1] + nxt.position[1]) / 2
        ax.arrow(
            mid_x - 0.25 * dx,
            mid_y - 0.25 * dy,
            0.5 * dx,
            0.5 * dy,
            head_width=0.2,
            head_length=0.25,
            fc="#4363d8",
            ec="#4363d8",
            linewidth=0.8,
            zorder=4,
            length_includes_head=True,
        )

    # --- labels & legend ----------------------------------------------
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Grid path solution")
    # build legend only once
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.show()

    return filename

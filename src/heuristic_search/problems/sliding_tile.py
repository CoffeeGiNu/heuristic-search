import copy
import os
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    final,
)

import matplotlib
import matplotlib.pyplot as plt
from IPython.display import HTML

# --- Add new imports for animation ---
from matplotlib import animation, patches
from matplotlib.axes import Axes
from overrides import override

from heuristic_search.base import Action, Cost, State
from heuristic_search.base.base import StateSpaceProblem

# ------------------------------------


@final
@dataclass(slots=True)
class TileState(State):
    tiles: list[list[int]]
    blank_position: tuple[int, ...]

    def __init__(self, tiles: list[list[int]]) -> None:
        self.tiles: list[list[int]] = tiles
        self.blank_position: tuple[int, ...] = self._get_blank_position()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TileState):
            return False
        return self.tiles == other.tiles

    def __hash__(self) -> int:
        return hash(tuple(tuple(row) for row in self.tiles))

    def __str__(self) -> str:
        return self.tiles.__str__()

    def _get_blank_position(self) -> tuple[int, ...]:
        for i, row in enumerate(self.tiles):
            if 0 in row:
                return (i, row.index(0))
        raise ValueError("No blank position found")


@final
@dataclass(slots=True)
class SlidingAction(Action):
    move_direction: tuple[int, ...]
    name: Optional[str]
    cost: Cost

    def __init__(
        self,
        move_direction: tuple[int, ...],
        name: Optional[str] = None,
        cost: Cost = 1,
    ) -> None:
        self.move_direction: tuple[int, ...] = move_direction
        self.name: Optional[str] = name
        self.cost: Cost = cost

    def __hash__(self) -> int:
        return hash(self.move_direction)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SlidingAction):
            return False
        return all(
            self.move_direction[i] == other.move_direction[i]
            for i in range(len(self.move_direction))
        )

    def __str__(self) -> str:
        return "move:(" + ", ".join(map(str, self.move_direction)) + ")"

    @override
    def get_action_name(self) -> str:
        return self.name if self.name else str(self)

    @override
    def get_action_cost(self) -> Cost:
        return self.cost


class NeighborStrategy(Protocol):
    def __call__(self, state: TileState) -> Iterable[Action]:
        """
        Returns a list of actions for the given state.
        """
        ...


def four_directions(state: TileState) -> Iterable[Action]:
    return (
        SlidingAction(move_direction=(0, 1), name="up"),
        SlidingAction(move_direction=(1, 0), name="right"),
        SlidingAction(move_direction=(0, -1), name="down"),
        SlidingAction(move_direction=(-1, 0), name="left"),
    )


# TODO: re-implement with numpy?
class TileMap(object):
    """
    A class representing a map for sliding tile.
    """

    shape: tuple[int, ...]
    tiles: Optional[list[list[int]]]

    def __init__(
        self,
        shape: tuple[int, ...],
        tiles: Optional[list[list[int]]] = None,
    ) -> None:
        self.shape = shape
        self.tiles = tiles

    def in_bounds(self, position: tuple[int, ...]) -> bool:
        return all(0 <= position[i] < self.shape[i] for i in range(len(self.shape)))

    # Deprecated conversion helpers removed – internal & external representation
    # are now both `tiles`, so no orientation transformation is necessary.


def manhattan_distance(state: TileState, goal_state: TileState) -> Cost:
    return sum(
        abs(state.tiles[i][j] - goal_state.tiles[i][j])
        for i in range(len(state.tiles))
        for j in range(len(state.tiles[i]))
    )


class SlidingTile(StateSpaceProblem):
    tile_map: TileMap
    initial_tiles: list[list[int]]
    goal_state: TileState
    neighbor_strategy: NeighborStrategy
    heuristic_function: Callable[..., Cost]

    def __init__(
        self,
        tile_map: TileMap,
        initial_tiles: list[list[int]],
        goal_tiles: list[list[int]],
        neighbor_strategy: NeighborStrategy,
        heuristic_function: Callable[..., Cost],
    ) -> None:
        self.tile_map = tile_map
        self.initial_tiles = initial_tiles
        self.goal_state = TileState(tiles=goal_tiles)
        self.neighbor_strategy: NeighborStrategy = neighbor_strategy
        self.heuristic_function: Callable[..., Cost] = heuristic_function

    @override
    def get_initial_state(self) -> State:
        return TileState(tiles=self.initial_tiles)

    @override
    def get_goal_state(self) -> State:
        return self.goal_state

    @override
    def is_goal_state(self, state: State) -> bool:
        if not isinstance(state, TileState):
            raise TypeError("State must be a TileState")
        return state == self.goal_state

    @override
    def get_available_actions(self, state: State) -> list[Action]:
        if not isinstance(state, TileState):
            raise TypeError("State must be a TileState")

        valid_actions: list[Action] = []
        for action in self.neighbor_strategy(state):
            if not isinstance(action, SlidingAction):
                raise TypeError("Action must be a SlidingAction")
            if self._is_valid_position(state, action):
                valid_actions.append(action)
        return valid_actions

    def _is_valid_position(self, state: TileState, action: SlidingAction) -> bool:
        next_position: tuple[int, ...] = self._get_next_position(state, action)
        return self.tile_map.in_bounds(next_position)

    def _get_next_position(
        self, state: TileState, action: SlidingAction
    ) -> tuple[int, ...]:
        return tuple(
            state.blank_position[i] + action.move_direction[i]
            for i in range(len(state.blank_position))
        )

    # TODO: do swap tiles algorithm to be more efficient
    @override
    def get_next_state(self, state: State, action: Action) -> State:
        if not isinstance(state, TileState):
            raise TypeError("State must be a TileState")
        if not isinstance(action, SlidingAction):
            raise TypeError("Action must be a SlidingAction")

        next_position: tuple[int, ...] = self._get_next_position(
            state=state, action=action
        )
        next_tiles: list[list[int]] = copy.deepcopy(state.tiles)
        next_tiles[state.blank_position[0]][state.blank_position[1]] = state.tiles[
            next_position[0]
        ][next_position[1]]
        next_tiles[next_position[0]][next_position[1]] = 0

        return TileState(tiles=next_tiles)

    @override
    def get_action_cost(self, state: State, action: Action) -> Cost:
        if not isinstance(state, TileState):
            raise TypeError("State must be a TileState")
        if not isinstance(action, SlidingAction):
            raise TypeError("Action must be a SlidingAction")
        return action.cost

    @override
    def heuristic(self, state: State) -> Cost:
        if not isinstance(state, TileState):
            raise TypeError("State must be a TileState")
        return self.heuristic_function(state=state, goal_state=self.goal_state)

    def state_to_cell(self, state: State) -> list[list[int]]:
        if not isinstance(state, TileState):
            raise TypeError("State must be a TileState")
        return [
            [state.tiles[i][j] for j in range(len(state.tiles[i]))]
            for i in range(len(state.tiles))
        ]


# -----------------------------------------------------------------------------
# Visualization Utilities (no modifications to existing classes)
# -----------------------------------------------------------------------------


def _draw_tile_board(ax: "Axes", tiles: List[List[int]]) -> None:  # noqa: N802
    """Draw a single sliding-tile board on the given Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes to draw on.
    tiles : list[list[int]]
        2-D list representing the board; 0 denotes blank.
    """
    n_rows = len(tiles)
    n_cols = len(tiles[0]) if n_rows > 0 else 0

    # Draw grid background
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.axis("off")

    for i in range(n_rows):
        for j in range(n_cols):
            value = tiles[i][j]
            # Create square patch
            facecolor = "white" if value == 0 else "#cccccc"
            edgecolor = "black"
            rect = patches.Rectangle(
                (j - 0.5, i - 0.5), 1, 1, facecolor=facecolor, edgecolor=edgecolor
            )
            ax.add_patch(rect)
            # Draw text for non-blank tiles
            if value != 0:
                ax.text(
                    j,
                    i,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                )


# Public API ------------------------------------------------------------------


def visualize_sliding_tile_path(
    problem: "StateSpaceProblem",
    path_with_actions: Sequence[Tuple["State", Optional["Action"]]],
    filename: Optional[str] = "tile_path.png",
    *,
    figsize: Optional[Tuple[float, float]] = None,
    show: bool = False,
    close: bool = True,
    **subplot_kwargs: Any,
) -> Union[str, Any]:
    """Visualize sliding-tile solution path.

    Parameters
    ----------
    problem : StateSpaceProblem
        Problem instance (must be SlidingTile).
    path_with_actions : Sequence[tuple[State, Optional[Action]]]
        Sequence of (state, action) pairs forming a solution path. The first
        element should be the initial state.
    filename : str, default "tile_path.png"
        Where to save the PNG output.
    figsize : tuple[float, float], optional
        Figure size passed to Matplotlib.
    show : bool, default False
        Whether to show the plot.
    close : bool, default True
        Whether to close the plot after saving.
    **subplot_kwargs : Any
        Additional keyword arguments for plt.subplots.

    Returns
    -------
    Union[str, plt.Figure]
        Path to the saved PNG file or the Matplotlib figure.
    """
    # Defensive import to avoid circular issues
    from heuristic_search.problems.sliding_tile import SlidingTile  # type: ignore

    if not isinstance(problem, SlidingTile):
        raise TypeError("visualize_sliding_tile_path expects a SlidingTile problem")

    # Extract just the states (ignore actions here)
    states: List["State"] = [s for s, _ in path_with_actions]

    n_steps = len(states)
    if n_steps == 0:
        raise ValueError("path_with_actions must contain at least one state")

    # Determine board dimension from first state
    first_state = states[0]
    if not hasattr(first_state, "tiles"):
        raise TypeError("State must be TileState with a 'tiles' attribute")

    rows = len(first_state.tiles)  # type: ignore[attr-defined]
    cols = len(first_state.tiles[0])  # type: ignore[attr-defined]

    # Layout: one row of subplots
    if figsize is None:
        figsize = (cols * n_steps * 1.2, rows * 1.2)

    fig, axes = plt.subplots(
        1, n_steps, figsize=figsize, squeeze=False, **subplot_kwargs
    )

    for idx, (state, ax) in enumerate(zip(states, axes.flatten())):
        if not hasattr(state, "tiles"):
            raise TypeError("All states must be TileState instances")
        _draw_tile_board(ax, state.tiles)  # type: ignore[attr-defined]
        ax.set_title(f"Step {idx}")

    plt.tight_layout()

    saved_path: Optional[str] = None
    if filename is not None:
        fig.savefig(filename, dpi=300)
        saved_path = os.path.abspath(filename)

    if show:
        plt.show()

    if close:
        plt.close(fig)

    # Return based on what the caller likely wants
    if saved_path is not None and not show:
        # Typical non-interactive use – return file path
        return saved_path

    return fig  # type: ignore[return-value]


# -----------------------------------------------------------------------------
# Interactive Visualization (Jupyter-friendly)
# -----------------------------------------------------------------------------

# ipywidgets is optional – fall back to a static Figure if unavailable
try:
    import ipywidgets as widgets
    from IPython.display import display
except ImportError:  # pragma: no cover
    widgets = None  # type: ignore

from matplotlib.widgets import Slider as _MplSlider  # fallback slider


def interactive_sliding_tile_path(
    problem: "StateSpaceProblem",
    path_with_actions: Sequence[Tuple["State", Optional["Action"]]],
    *,
    tile_size: int = 80,
    font_size: int = 20,
    show_play_button: bool = True,
) -> (
    Any
):  # Return type is plotly.graph_objs.Figure but keep Any to avoid hard dep in typing
    """Create an interactive sliding-tile animation with Plotly.

    Parameters
    ----------
    problem : StateSpaceProblem
        Must be a :class:`SlidingTile` instance.
    path_with_actions : Sequence[tuple[State, Optional[Action]]]
        Solution path. First element should be the initial state.
    tile_size : int, default 80
        Pixel size of each square tile.
    font_size : int, default 20
        Font size for tile numbers.
    show_play_button : bool, default True
        If True, include a ▶ play button in the figure controls.

    Returns
    -------
    plotly.graph_objs.Figure
        A Plotly figure containing the animation.
    """

    # Deferred import to avoid mandatory dependency at import time
    try:
        import plotly.graph_objs as go  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "interactive_sliding_tile_plotly requires plotly; install via `pip install plotly`."
        ) from exc

    # Validate inputs -------------------------------------------------------
    from heuristic_search.problems.sliding_tile import SlidingTile  # type: ignore

    if not isinstance(problem, SlidingTile):
        raise TypeError("interactive_sliding_tile_plotly expects a SlidingTile problem")

    states: list["State"] = [s for s, _ in path_with_actions]
    n_steps: int = len(states)
    if n_steps == 0:
        raise ValueError("path_with_actions must contain at least one state")

    first_state = states[0]
    if not hasattr(first_state, "tiles"):
        raise TypeError("State must have a 'tiles' attribute (TileState)")

    rows = len(first_state.tiles)  # type: ignore[attr-defined]
    cols = len(first_state.tiles[0])  # type: ignore[attr-defined]

    # Helper to build shapes & annotations for a given layout --------------
    def _make_shapes_annos(tile_layout: list[list[int]]):
        shapes: list[dict[str, Any]] = []
        annos: list[dict[str, Any]] = []

        for i in range(rows):
            for j in range(cols):
                value = tile_layout[i][j]
                y_center = rows - 1 - i
                shapes.append(
                    dict(
                        type="rect",
                        x0=j - 0.5,
                        y0=y_center - 0.5,
                        x1=j + 0.5,
                        y1=y_center + 0.5,
                        line=dict(color="black", width=1),
                        fillcolor="#cccccc" if value != 0 else "white",
                    )
                )
                if value != 0:
                    annos.append(
                        dict(
                            x=j,
                            y=y_center,
                            text=str(value),
                            showarrow=False,
                            font=dict(size=font_size, color="black"),
                            xanchor="center",
                            yanchor="middle",
                        )
                    )

        return shapes, annos

    # Build frames ---------------------------------------------------------
    frames: list[Any] = []
    for step_idx, state in enumerate(states):
        shapes, annos = _make_shapes_annos(state.tiles)  # type: ignore[arg-type]
        frames.append(
            go.Frame(name=str(step_idx), layout=dict(shapes=shapes, annotations=annos))
        )

    # Base figure from first frame layout
    base_shapes, base_annos = _make_shapes_annos(states[0].tiles)  # type: ignore[arg-type]
    fig = go.Figure(frames=frames)

    fig.update_layout(shapes=base_shapes, annotations=base_annos)

    # Axes formatting ------------------------------------------------------
    fig.update_xaxes(
        visible=False,
        range=[-0.5, cols - 0.5],
        constrain="domain",
    )
    fig.update_yaxes(
        visible=False,
        range=[-0.5, rows - 0.5],
        scaleanchor="x",
        scaleratio=1,
    )

    fig.update_layout(
        width=cols * tile_size,
        height=rows * tile_size,
        margin=dict(l=0, r=0, t=40, b=0),
        title=dict(text="Sliding-Tile Path", x=0.5, xanchor="center"),
        showlegend=False,
        updatemenus=[],  # will fill below
        sliders=[],
    )

    # Slider ----------------------------------------------------------------
    slider_steps = [
        dict(
            method="animate",
            args=[
                [str(k)],
                dict(
                    mode="immediate",
                    frame=dict(duration=0, redraw=True),
                    transition=dict(duration=0),
                ),
            ],
            label=f"{k}",
        )
        for k in range(n_steps)
    ]

    sliders = [
        dict(
            active=0,
            steps=slider_steps,
            transition=dict(duration=0),
            x=0.1,
            y=-0.05,
            len=0.8,
        )
    ]
    fig.update_layout(sliders=sliders)

    # Play button -----------------------------------------------------------
    if show_play_button:
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    y=-0.05,
                    x=0.5,
                    xanchor="center",
                    yanchor="top",
                    buttons=[
                        dict(
                            label="▶ Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=400, redraw=True),
                                    transition=dict(duration=0),
                                    fromcurrent=True,
                                    mode="immediate",
                                ),
                            ],
                        )
                    ],
                )
            ]
        )

    return fig

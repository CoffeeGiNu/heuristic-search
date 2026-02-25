from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Protocol, Sequence, cast, final
from typing_extensions import override

import matplotlib.pyplot as plt
from matplotlib import patches

from ..base import Action, Cost, State, StateSpaceProblem


@final
@dataclass(slots=True)
class GridState(State):
    position: tuple[int, ...]

    @override
    def __hash__(self) -> int:
        return hash(self.position)

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GridState):
            return False
        return self.position == other.position


@final
@dataclass(slots=True)
class GridAction(Action):
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

    @override
    def __hash__(self) -> int:
        return hash(self.move_direction)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GridAction):
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
    def __call__(self, state: GridState) -> Iterable[Action]:
        """
        Returns a list of actions for the given state.
        """
        ...


def four_directions_2d(state: GridState) -> Iterable[Action]:
    """
    Returns the four directions (left, right, up, down)
    from the given state for 2d grid.
    """
    return (
        GridAction(move_direction=(0, 1), name="up"),
        GridAction(move_direction=(1, 0), name="right"),
        GridAction(move_direction=(0, -1), name="down"),
        GridAction(move_direction=(-1, 0), name="left"),
    )


class GridMap(object):
    """
    A class representing a map for pathfinding.
    """

    shape: tuple[int, ...]
    walls: set[tuple[int, ...]]

    def __init__(
        self,
        shape: tuple[int, ...],
        walls: Optional[set[tuple[int, ...]]] = None,
    ) -> None:
        self.shape: tuple[int, ...] = shape
        self.walls: set[tuple[int, ...]] = walls if walls is not None else set()

    def is_wall(self, position: tuple[int, ...]) -> bool:
        return position in self.walls

    def in_bounds(self, position: tuple[int, ...]) -> bool:
        return all(0 <= position[i] < self.shape[i] for i in range(len(self.shape)))

    def map_to_cell(self) -> list[list[int]]:
        # TODO: implement.
        raise NotImplementedError("Not implemented")


def manhattan_distance(
    current_position: tuple[int, ...], goal_position: tuple[int, ...]
) -> Cost:
    return sum(
        abs(current_position[i] - goal_position[i])
        for i in range(len(current_position))
    )


# TODO: Implement for wall blocks.
@final
@dataclass(slots=True)
class GridPathFinding(StateSpaceProblem):
    grid_map: GridMap
    initial_position: tuple[int, ...]
    goal_position: tuple[int, ...]
    neighbor_strategy: NeighborStrategy
    heuristic_function: Callable[..., Cost]

    def __init__(
        self,
        grid_map: GridMap,
        initial_position: tuple[int, ...],
        goal_position: tuple[int, ...],
        actions: Optional[Iterable[Action]] = None,
        heuristic_function: Callable[..., Cost] = manhattan_distance,
        neighbor_strategy: Optional[NeighborStrategy] = None,
    ) -> None:
        self.grid_map: GridMap = grid_map
        self.initial_position: tuple[int, ...] = initial_position
        self.goal_position: tuple[int, ...] = goal_position
        self.heuristic_function: Callable[..., Cost] = heuristic_function

        if actions is None and neighbor_strategy is None:
            raise ValueError("Either actions or neighbor_strategy must be provided")

        if actions is not None and neighbor_strategy is not None:
            raise ValueError(
                "Only one of actions or neighbor_strategy must be provided"
            )

        if actions is not None and neighbor_strategy is None:
            simple_action_strategy: tuple[Action, ...] = tuple(actions)
            self.neighbor_strategy: NeighborStrategy = (
                lambda state, actions=simple_action_strategy: actions
            )

        if actions is None and neighbor_strategy is not None:
            self.neighbor_strategy: NeighborStrategy = neighbor_strategy

    @override
    def get_initial_state(self) -> State:
        return GridState(position=self.initial_position)

    @override
    def is_goal_state(self, state: State) -> bool:
        if not isinstance(state, GridState):
            raise TypeError("State must be of type GridState")

        return state.position == self.goal_position

    @override
    def get_available_actions(self, state: State) -> list[Action]:
        if not isinstance(state, GridState):
            raise TypeError("State must be of type GridState")

        valid_actions: list[Action] = []
        for action in self.neighbor_strategy(state=state):
            if not isinstance(action, GridAction):
                raise TypeError("Action must be of type GridAction")

            if self._is_valid_position(state=state, action=action):
                valid_actions.append(action)

        return valid_actions

    def _is_valid_position(self, state: GridState, action: GridAction) -> bool:
        next_position: tuple[int, ...] = self._get_next_position(
            state=state, action=action
        )
        return self.grid_map.in_bounds(next_position) and not self.grid_map.is_wall(
            next_position
        )

    def _get_next_position(
        self, state: GridState, action: GridAction
    ) -> tuple[int, ...]:
        return tuple(
            position + action.move_direction[i]
            for i, position in enumerate(state.position)
        )

    @override
    def get_next_state(self, state: State, action: Action) -> State:
        if not isinstance(state, GridState):
            raise TypeError("State must be of type GridState")
        if not isinstance(action, GridAction):
            raise TypeError("Action must be of type GridAction")

        next_position: tuple[int, ...] = self._get_next_position(
            state=state, action=action
        )

        return GridState(position=next_position)

    @override
    def get_action_cost(self, state: State, action: Action) -> Cost:
        if not isinstance(state, GridState):
            raise TypeError("State must be of type GridState")
        if not isinstance(action, GridAction):
            raise TypeError("Action must be of type GridAction")

        return action.get_action_cost()

    @override
    def heuristic(self, state: State) -> Cost:
        if not isinstance(state, GridState):
            raise TypeError("State must be of type GridState")
        return self.heuristic_function(
            current_position=state.position, goal_position=self.goal_position
        )


def visualize_grid_path_with_walls(
    grid_path_finding: StateSpaceProblem,
    path_with_actions: Sequence[tuple[State, Optional[Action]]],
    filename: str = "path.png",
    *,
    figsize: tuple[float, float] = (6, 6),
) -> str:
    """
    Render the path and walls on a grid and save to an image file.
    """

    grid_path_finding = cast(GridPathFinding, grid_path_finding)

    path_with_actions = cast(
        Sequence[tuple[GridState, Optional[GridAction]]], path_with_actions
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    width, height = grid_path_finding.grid_map.shape

    # invert y axis so (0,0) is top‑left like array indexing
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)

    # --- draw grid -----------------------------------------------------
    for x in range(width + 1):
        ax.axvline(x - 0.5, color="lightgray", linewidth=0.8, zorder=0)  # type: ignore
    for y in range(height + 1):
        ax.axhline(y - 0.5, color="lightgray", linewidth=0.8, zorder=0)

    # --- draw walls ----------------------------------------------------
    for wx, wy in grid_path_finding.grid_map.walls:
        ax.add_patch(
            patches.Rectangle(
                (wx - 0.5, wy - 0.5),
                1,
                1,
                facecolor="black",
                alpha=0.7,
                edgecolor="black",
                linewidth=1.0,
                zorder=1,
            )
        )

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
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.show()

    return filename

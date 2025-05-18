from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Protocol, final

from overrides import override

from ..base import Action, Cost, State, StateSpaceProblem


@final
@dataclass(slots=True)
class GridState(State):
    position: tuple[int, ...]

    def __hash__(self) -> int:
        return hash(self.position)

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


# TODO: Implement for wall blocks.
@final
@dataclass(slots=True)
class GridPathFinding(StateSpaceProblem):
    shape: tuple[int, ...]
    initial_position: tuple[int, ...]
    goal_position: tuple[int, ...]
    neighbor_strategy: NeighborStrategy

    def __init__(
        self,
        shape: tuple[int, ...] = (3, 3),
        initial_position: tuple[int, ...] = (0, 0),
        goal_position: tuple[int, ...] = (2, 2),
        actions: Optional[Iterable[Action]] = None,
        neighbor_strategy: Optional[NeighborStrategy] = None,
    ) -> None:
        self.shape: tuple[int, ...] = shape
        self.initial_position: tuple[int, ...] = initial_position
        self.goal_position: tuple[int, ...] = goal_position

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

        return all(
            self.shape[i] > next_position[i] >= 0 for i in range(len(self.shape))
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

        # TODO: Implement heuristic function.
        # Manhattan distance heuristic.

        return 0

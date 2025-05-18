from typing import Optional

from overrides import override

from ..base import Action, State, StateSpaceProblem


class TravelingSalesPersonState(State):
    visited: set[int]
    current_city: int

    def __init__(self, visited: set[int], current_city: int) -> None:
        self.visited: set[int] = visited
        self.current_city: int = current_city

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TravelingSalesPersonState):
            return False
        return (self.visited == other.visited) and (
            self.current_city == other.current_city
        )


class TravelingSalesPersonAction(Action):
    city: int
    name: Optional[str]
    cost: int | float

    def __init__(
        self, city: int, name: Optional[str] = None, cost: int | float = 1
    ) -> None:
        self.city: int = city
        self.name: Optional[str] = name
        self.cost: int | float = cost

    def __hash__(self) -> int:
        return hash(self.city)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TravelingSalesPersonAction):
            return False
        return self.city == other.city

    def __str__(self) -> str:
        return f"move:({self.city})"

    @override
    def get_action_name(self) -> str:
        return self.name if self.name else str(self)

    @override
    def get_action_cost(self) -> int | float:
        return self.cost


# TODO: improve implementation.
# TODO: now, distance_matrix's pre-compute is needed.
class TravelingSalesPerson(StateSpaceProblem):
    cities: list[int]
    distance_matrix: list[list[int | float]]
    initial_position: int

    def __init__(
        self,
        cities: list[int],
        distance_matrix: list[list[int | float]],
        initial_position: Optional[int] = None,
    ) -> None:
        self.cities: list[int] = cities
        self.distance_matrix: list[list[int | float]] = distance_matrix
        if initial_position is None:
            self.initial_position = cities[0]
        else:
            self.initial_position = initial_position

    @override
    def get_initial_state(self) -> TravelingSalesPersonState:
        return TravelingSalesPersonState(
            visited=set(), current_city=self.initial_position
        )

    @override
    def is_goal_state(self, state: State) -> bool:
        if not isinstance(state, TravelingSalesPersonState):
            raise TypeError(f"Expected TravelingSalesPersonState, got {type(state)}")
        return set(self.cities) == state.visited

    @override
    def get_available_actions(self, state: State) -> list[Action]:
        if not isinstance(state, TravelingSalesPersonState):
            raise TypeError(f"Expected TravelingSalesPersonState, got {type(state)}")
        return [
            TravelingSalesPersonAction(city=city)
            for city in self.cities
            if city not in state.visited
        ]

    @override
    def get_next_state(self, state: State, action: Action) -> State:
        if not isinstance(state, TravelingSalesPersonState):
            raise TypeError(f"Expected TravelingSalesPersonState, got {type(state)}")
        if not isinstance(action, TravelingSalesPersonAction):
            raise TypeError(f"Expected TravelingSalesPersonAction, got {type(action)}")
        new_visited: set[int] = state.visited.copy()
        new_visited.add(action.city)

        return TravelingSalesPersonState(visited=new_visited, current_city=action.city)

    @override
    def get_action_cost(self, state: State, action: Action) -> int | float:
        if not isinstance(state, TravelingSalesPersonState):
            raise TypeError(f"Expected TravelingSalesPersonState, got {type(state)}")
        if not isinstance(action, TravelingSalesPersonAction):
            raise TypeError(f"Expected TravelingSalesPersonAction, got {type(action)}")

        return self.distance_matrix[state.current_city][action.city]

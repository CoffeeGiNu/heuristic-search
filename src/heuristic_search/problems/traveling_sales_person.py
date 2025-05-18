from math import dist
from typing import Optional, TypeAlias

from overrides import override

from ..base import Action, Cost, State, StateSpaceProblem


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
    cost: Cost

    def __init__(self, city: int, name: Optional[str] = None, cost: Cost = 1) -> None:
        self.city: int = city
        self.name: Optional[str] = name
        self.cost: Cost = cost

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
    def get_action_cost(self) -> Cost:
        return self.cost


Distances: TypeAlias = list[list[Cost]]
CityPositions: TypeAlias = dict[int, tuple[float, ...]]
CityPair: TypeAlias = tuple[int, int]


class TravelingSalesPerson(StateSpaceProblem):
    cities: list[int]
    city_positions: CityPositions
    distances: dict[CityPair, Cost]
    initial_position: int

    def __init__(
        self,
        cities: list[int],
        distance_matrix: Optional[Distances] = None,
        city_positions: Optional[CityPositions] = None,
        initial_position: Optional[int] = None,
    ) -> None:
        self.cities: list[int] = cities
        city_count: int = len(cities)
        self.distances: dict[CityPair, Cost] = {}
        self.city_positions: CityPositions = (
            city_positions if city_positions is not None else {}
        )
        if distance_matrix is not None:
            for i in range(city_count):
                for j in range(city_count):
                    cost: Cost = distance_matrix[i][j]
                    self.distances[(cities[i], cities[j])] = cost
                    self.distances[(cities[j], cities[i])] = cost
        elif city_positions is not None:
            for i in range(city_count):
                for j in range(city_count):
                    cost: Cost = float(
                        dist(city_positions[cities[i]], city_positions[cities[j]])
                    )
                    self.distances[(cities[i], cities[j])] = cost
                    self.distances[(cities[j], cities[i])] = cost
        else:
            raise ValueError(
                "Either distance_matrix or city_positions must be provided."
            )
        if initial_position is None:
            self.initial_position: int = cities[0]
        else:
            self.initial_position: int = initial_position

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
    def get_action_cost(self, state: State, action: Action) -> Cost:
        if not isinstance(state, TravelingSalesPersonState):
            raise TypeError(f"Expected TravelingSalesPersonState, got {type(state)}")
        if not isinstance(action, TravelingSalesPersonAction):
            raise TypeError(f"Expected TravelingSalesPersonAction, got {type(action)}")

        return self.distances[(state.current_city, action.city)]

    @override
    def heuristic(self, state: State) -> Cost:
        if not isinstance(state, TravelingSalesPersonState):
            raise TypeError(f"Expected TravelingSalesPersonState, got {type(state)}")
        # TODO: Implement a heuristic function.

        return 0

from abc import ABC, abstractmethod
from os import path
from typing import Optional

from overrides import override


class State(ABC):
    """
    Interface for a state.
    """

    pass


# TODO: costはStateに従って変わる. 引数にStateを渡す？
class Action(ABC):
    """
    Interface for an action.
    """

    cost: int | float
    name: Optional[str]

    @abstractmethod
    def get_action_name(self) -> str:
        """
        Returns the name of the action.
        """

    @abstractmethod
    def get_action_cost(self) -> int | float:
        """
        Returns the cost of the action.
        """


class Node(ABC):
    """
    Interface for a search node.
    """

    state: State
    path_cost: int | float
    depth: int
    parent: Optional["Node"]

    @abstractmethod
    def set_path_cost(self, cost: int | float) -> None:
        """
        Sets the path cost of the node.
        """

    @abstractmethod
    def set_depth(self, depth: int) -> None:
        """
        Sets the depth of the node.
        """

    @abstractmethod
    def set_parent(self, parent: "Node") -> None:
        """
        Sets the parent of the node.
        """

    @abstractmethod
    def get_parent(self) -> Optional["Node"]:
        """
        Returns the parent of the node.
        """

    @abstractmethod
    def get_path(self) -> list["Node"]:
        """
        Returns the path from the root to this node.
        """


class StateSpaceProblem(ABC):
    """
    Interface for a state space problem.
    """

    @abstractmethod
    def get_initial_state(self) -> State:
        """
        Returns the initial state of the problem.
        """

    @abstractmethod
    def is_goal_state(self, state: State) -> bool:
        """
        Returns True if the state is a goal state, False otherwise.
        """

    @abstractmethod
    def get_available_actions(self, state: State) -> list[Action]:
        """
        Returns a list of available actions for the given state.
        """

    @abstractmethod
    def get_next_state(self, state: State, action: Action) -> State:
        """
        Returns the next state after applying the action to the state.
        """

    @abstractmethod
    def get_action_cost(self, state: State, action: Action) -> int | float:
        """
        Returns the cost of applying the action to the state.
        """

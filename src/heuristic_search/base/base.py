from abc import ABC, abstractmethod
from os import path
from typing import Any, Optional, TypeAlias

from overrides import override

Cost: TypeAlias = int | float


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

    cost: Cost
    name: Optional[str]

    @abstractmethod
    def get_action_name(self) -> str:
        """
        Returns the name of the action.
        """

    @abstractmethod
    def get_action_cost(self) -> Cost:
        """
        Returns the cost of the action.
        """


class Node(ABC):
    """
    Interface for a search node.
    """

    state: State
    path_cost: Cost
    depth: int
    parent: Optional["Node"]

    @abstractmethod
    def set_path_cost(self, cost: Cost) -> None:
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
    def get_goal_state(self) -> State:
        """
        Returns the goal state of the problem.
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
    def get_action_cost(self, state: State, action: Action) -> Cost:
        """
        Returns the cost of applying the action to the state.
        """

    @abstractmethod
    def heuristic(self, state: State) -> Cost:
        """
        Returns the heuristic value of the state.
        """


class OpenList(ABC):
    """
    Interface for an open list.
    """

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the number of nodes in the open list.
        """

    @abstractmethod
    def pop(self) -> Node:
        """
        Pops the node with the lowest cost from the open list.
        """

    @abstractmethod
    def push(self, node: Node, priority: Any) -> None:
        """
        Pushes a node to the open list.
        """

    @abstractmethod
    def shrink(self, beam_width: Optional[int] = None) -> None:
        """
        Shrinks the open list, for limited memory search like beam search.
        """


class ClosedList(ABC):
    """
    Interface for a closed list.
    """

    @abstractmethod
    def push(self, node: Node) -> None:
        """
        Pushes a node to the closed list.
        """

    @abstractmethod
    def is_explored(self, node: Node) -> bool:
        """
        Returns True if the node is explored, False otherwise.
        """

from typing import Callable, Optional

from overrides import override

from ..base import Action, ClosedList, Cost, Node, OpenList, State, StateSpaceProblem
from ..utils import SearchLogger


class GraphSearchNode(Node):
    def __init__(self, state: State) -> None:
        super().__init__()
        self.state: State = state
        self.path_cost: Cost = 0
        self.depth: int = 0
        self.parent: Node | None = None

    @override
    def set_path_cost(self, cost: Cost) -> None:
        self.path_cost: Cost = cost

    @override
    def set_depth(self, depth: int) -> None:
        self.depth: int = depth

    @override
    def set_parent(self, parent: Node) -> None:
        self.parent = parent

    @override
    def get_parent(self) -> Node | None:
        return self.parent

    @override
    def get_path(self) -> list[Node]:
        current_node: Optional[Node] = self
        paths: list[Node] = []

        while current_node is not None and hasattr(current_node, "parent"):
            paths.append(current_node)
            current_node = current_node.get_parent()

        return paths


class GraphSearch(object):
    open_list: list[Node]
    closed_list: set[Node]
    initial_state: State
    initial_node: Node
    problem: StateSpaceProblem
    priority_function: Callable
    logger: SearchLogger

    def __init__(
        self,
        problem: StateSpaceProblem,
        priority_function: Callable,
        logger: SearchLogger,
    ) -> None:
        self.problem: StateSpaceProblem = problem
        self.priority_function: Callable = priority_function
        self.logger: SearchLogger = logger
        self.initial_state: State = problem.get_initial_state()
        self.initial_node: Node = GraphSearchNode(state=self.initial_state)
        self.open_list: list[Node] = [self.initial_node]
        self.closed_list: set[Node] = {self.initial_node}
        self.initial_node.set_path_cost(cost=0)
        self.initial_node.set_depth(depth=0)

    def is_explored(self, node: Node) -> bool:
        for explored_node in self.closed_list:
            if (explored_node.state == node.state) and (
                explored_node.path_cost <= node.path_cost
            ):
                return True
        return False

    def solve(self) -> Optional[list[Node]]:
        self.logger.start()
        while len(self.open_list) > 0:
            self.open_list.sort(
                key=lambda node: self.priority_function(node), reverse=True
            )
            current_node: Node = self.open_list.pop()
            self.logger.expanded += 1

            if self.problem.is_goal_state(state=current_node.state):
                self.logger.stop()
                self.logger.print()
                return current_node.get_path()
            else:
                actions: list[Action] = self.problem.get_available_actions(
                    state=current_node.state
                )
                for action in actions:
                    next_state: State = self.problem.get_next_state(
                        state=current_node.state, action=action
                    )
                    next_node: Node = GraphSearchNode(state=next_state)
                    next_node.set_depth(depth=current_node.depth + 1)
                    next_node.set_path_cost(
                        cost=current_node.path_cost
                        + self.problem.get_action_cost(
                            state=current_node.state, action=action
                        )
                    )
                    if not self.is_explored(node=next_node):
                        next_node.set_parent(parent=current_node)
                        self.open_list.append(next_node)
                        self.closed_list.add(next_node)
                        self.logger.generated += 1
                    else:
                        self.logger.pruned += 1

        self.logger.stop()
        self.logger.print()
        return None


# TODO: Interface for GraphSearch Algorithms.
class OptimizedGraphSearch(object):
    """
    Optimized graph search algorithm, using open and closed lists.
    """

    open_list: OpenList
    closed_list: ClosedList
    initial_state: State
    initial_node: Node
    problem: StateSpaceProblem
    priority_function: Callable
    logger: SearchLogger

    def __init__(
        self,
        problem: StateSpaceProblem,
        priority_function: Callable,
        open_list: OpenList,
        closed_list: ClosedList,
        logger: SearchLogger,
    ) -> None:
        self.problem: StateSpaceProblem = problem
        self.priority_function: Callable = priority_function
        self.open_list: OpenList = open_list
        self.closed_list: ClosedList = closed_list
        self.logger: SearchLogger = logger
        self.initial_state: State = problem.get_initial_state()
        self.initial_node: Node = GraphSearchNode(state=self.initial_state)
        self.open_list.push(
            node=self.initial_node,
            priority=priority_function(self.initial_node),
        )
        self.closed_list.push(node=self.initial_node)

    def is_explored(self, node: Node) -> bool:
        return self.closed_list.is_explored(node=node)

    def solve(self) -> Optional[list[Node]]:
        self.logger.start()
        while len(self.open_list) > 0:
            current_node: Node = self.open_list.pop()
            self.logger.expanded += 1

            if self.problem.is_goal_state(state=current_node.state):
                self.logger.stop()
                self.logger.print()
                return current_node.get_path()
            else:
                actions: list[Action] = self.problem.get_available_actions(
                    state=current_node.state
                )
                for action in actions:
                    next_state: State = self.problem.get_next_state(
                        state=current_node.state, action=action
                    )
                    next_node: Node = GraphSearchNode(state=next_state)
                    next_node.set_depth(depth=current_node.depth + 1)
                    next_node.set_path_cost(
                        cost=current_node.path_cost
                        + self.problem.get_action_cost(
                            state=current_node.state, action=action
                        )
                    )

                    if not self.is_explored(node=next_node):
                        next_node.set_parent(parent=current_node)
                        self.open_list.push(
                            node=next_node,
                            priority=self.priority_function(next_node),
                        )
                        self.closed_list.push(node=next_node)
                        self.logger.generated += 1
                    else:
                        self.logger.pruned += 1

        self.logger.stop()
        self.logger.print()
        return None

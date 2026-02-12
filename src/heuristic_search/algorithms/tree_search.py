from typing import Callable, Optional

from overrides import override

from ..base import Action, Cost, Node, State, StateSpaceProblem
from ..utils import SearchLogger


# TODO: getter/setterについて要検討.
# (例) fieldがそれぞれvalidation必要かどうか？と公開かどうかでproperty.
# validなstateかどうか？やdepthは非負？とか.
class TreeSearchNode(Node):
    def __init__(self, state: State) -> None:
        super().__init__()
        self.state = state
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
        self.parent: Node | None = parent

    @override
    def get_parent(self) -> Optional[Node]:
        return self.parent

    @override
    def get_path(self) -> list[Node]:
        current_node: Optional[Node] = self
        paths: list[Node] = []

        while current_node is not None and hasattr(current_node, "parent"):
            paths.append(current_node)
            current_node = current_node.get_parent()

        return paths


class TreeSearch(object):
    open_list: list[Node]
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
        self.initial_node: Node = TreeSearchNode(state=self.initial_state)
        self.open_list: list[Node] = [self.initial_node]
        self.initial_node.set_path_cost(cost=0)
        self.initial_node.set_depth(depth=0)

    def solve(self) -> list[Node] | None:
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
                    next_node: Node = TreeSearchNode(state=next_state)
                    next_node.set_parent(parent=current_node)
                    next_node.set_depth(depth=current_node.depth + 1)
                    next_node.set_path_cost(
                        cost=self.problem.get_action_cost(
                            state=current_node.state, action=action
                        )
                    )
                    self.open_list.append(next_node)
                    self.logger.generated += 1

        self.logger.stop()
        self.logger.print()
        return None


class RecursiveTreeSearch(object):
    initial_state: State
    initial_node: Node
    problem: StateSpaceProblem
    logger: SearchLogger

    def __init__(
        self,
        problem: StateSpaceProblem,
        logger: SearchLogger,
    ) -> None:
        self.problem: StateSpaceProblem = problem
        self.logger: SearchLogger = logger
        self.initial_state: State = problem.get_initial_state()
        self.initial_node: Node = TreeSearchNode(state=self.initial_state)
        self.initial_node.set_path_cost(0)
        self.initial_node.set_depth(0)

    def solve(self) -> Optional[list[Node]]:
        self.logger.start()
        path: Optional[list[Node]] = self.recursive_search(
            current_node=self.initial_node
        )
        self.logger.stop()
        self.logger.print()
        return path

    def recursive_search(self, current_node: Node) -> Optional[list[Node]]:
        self.logger.expanded += 1
        if self.problem.is_goal_state(state=current_node.state):
            return current_node.get_path()
        else:
            actions: list[Action] = self.problem.get_available_actions(
                state=current_node.state
            )
            for action in actions:
                next_state: State = self.problem.get_next_state(
                    state=current_node.state, action=action
                )
                next_node: Node = TreeSearchNode(state=next_state)
                next_node.set_parent(parent=current_node)
                next_node.set_depth(depth=current_node.depth + 1)
                next_node.set_path_cost(
                    cost=self.problem.get_action_cost(
                        state=current_node.state, action=action
                    )
                )
                self.logger.generated += 1
                path: Optional[list[Node]] = self.recursive_search(
                    current_node=next_node
                )
                if path is not None:
                    return path
        return None

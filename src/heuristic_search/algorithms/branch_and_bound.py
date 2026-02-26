from typing_extensions import override

from ..base import Action, Cost, Node, State, StateSpaceProblem
from ..utils import SearchLogger


# TODO: SearchNode to Common implementation?
class BnBSearchNode(Node):
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
    def get_parent(self) -> Node | None:
        return self.parent

    @override
    def get_path(self) -> list[Node]:
        current_node: Node | None = self
        paths: list[Node] = []

        while current_node is not None and hasattr(current_node, "parent"):
            paths.append(current_node)
            current_node = current_node.get_parent()

        return paths


class BranchAndBoundRecursiveSearch(object):
    logger: SearchLogger
    initial_state: State
    initial_node: Node
    problem: StateSpaceProblem

    def __init__(self, problem: StateSpaceProblem, logger: SearchLogger) -> None:
        self.logger: SearchLogger = logger
        self.problem: StateSpaceProblem = problem
        self.initial_state: State = problem.get_initial_state()
        self.initial_node: Node = BnBSearchNode(state=self.initial_state)
        self.current_best_cost: Cost = float("inf")
        self.current_best_path: list[Node] | None = None
        self.initial_node.set_path_cost(cost=0)
        self.initial_node.set_depth(depth=0)

    def solve(self) -> list[Node] | None:
        self.logger.start()
        path, cost = self.recursive_search(current_node=self.initial_node)
        self.logger.stop()
        self.logger.print()
        return path

    def recursive_search(self, current_node: Node) -> tuple[list[Node] | None, Cost]:
        self.logger.expanded += 1
        if self.problem.is_goal_state(state=current_node.state):
            if current_node.path_cost < self.current_best_cost:
                self.current_best_cost = current_node.path_cost
                self.current_best_path = current_node.get_path()
        else:
            actions: list[Action] = self.problem.get_available_actions(
                state=current_node.state
            )
            # (path_cost, heuristic_cost, action, state)
            candidates: list[tuple[Cost, Cost, Action, State]] = []
            for action in actions:
                next_state: State = self.problem.get_next_state(
                    state=current_node.state, action=action
                )
                path_cost: Cost = current_node.path_cost + self.problem.get_action_cost(
                    state=current_node.state, action=action
                )
                heuristic_cost: Cost = self.problem.heuristic(state=next_state)
                candidates.append((path_cost, heuristic_cost, action, next_state))

            candidates.sort(key=lambda candidate: candidate[0] + candidate[1])

            for candidate in candidates:
                next_node: Node = BnBSearchNode(state=candidate[3])
                next_node.set_parent(parent=current_node)
                next_node.set_depth(depth=current_node.depth + 1)
                next_node.set_path_cost(cost=candidate[0])
                if next_node.path_cost + candidate[1] < self.current_best_cost:
                    self.logger.generated += 1
                    self.current_best_path, self.current_best_cost = (
                        self.recursive_search(current_node=next_node)
                    )
                else:
                    self.logger.pruned += 1

            # Must be sorted actions by heuristic value?
            # actions.sort(
            #     key=lambda action: current_node.path_cost
            #     + self.problem.get_action_cost(state=current_node.state, action=action)
            #     + self.problem.heuristic(
            #         state=self.problem.get_next_state(
            #             state=current_node.state, action=action
            #         )
            #     )
            # )

            # for action in actions:
            #     next_state: State = self.problem.get_next_state(
            #         state=current_node.state, action=action
            #     )
            #     next_node: Node = BnBSearchNode(state=next_state)
            #     next_node.set_parent(parent=current_node)
            #     next_node.set_depth(depth=current_node.depth + 1)
            #     next_node.set_path_cost(
            #         cost=current_node.path_cost
            #         + self.problem.get_action_cost(
            #             state=current_node.state, action=action
            #         )
            #     )
            #     if (
            #         next_node.path_cost + self.problem.heuristic(state=next_node.state)
            #         < self.current_best_cost
            #     ):
            #         self.logger.generated += 1
            #         self.current_best_path, self.current_best_cost = (
            #             self.recursive_search(current_node=next_node)
            #         )
            #     else:
            #         self.logger.pruned += 1

        return self.current_best_path, self.current_best_cost

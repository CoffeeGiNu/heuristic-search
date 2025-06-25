from typing import Callable

from overrides import override

from ..base import Action, Cost, Node, State, StateSpaceProblem
from ..utils.logger import SearchLogger
from .graph_search import GraphSearch, GraphSearchNode


class CostLimitedDepthFirstSearch(GraphSearch):
    def __init__(
        self,
        problem: StateSpaceProblem,
        logger: SearchLogger,
        priority_function: Callable = lambda node: node.depth,
        cost_limit: Cost = float("inf"),
    ):
        super().__init__(
            problem=problem,
            priority_function=priority_function,
            logger=logger,
        )
        self.cost_limit: Cost = cost_limit
        self.candidate_cost_limit: Cost = float("inf")

    @override
    def solve(self):
        self.logger.start()
        while len(self.open_list) > 0:
            self.open_list.sort(key=lambda node: node.depth, reverse=False)
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
                    priority: Cost = self.priority_function(next_node)
                    if priority <= self.cost_limit and not self.is_explored(
                        node=next_node
                    ):
                        next_node.set_parent(parent=current_node)
                        self.logger.generated += 1
                        self.open_list.append(next_node)
                        self.closed_list.add(next_node)
                    else:
                        if self.cost_limit < priority < self.get_candidate_cost_limit():
                            self.candidate_cost_limit = priority
                        self.logger.pruned += 1
        self.logger.stop()
        self.logger.print()
        return None

    # TODO: この辺りの実装がこれで良いか要検討.
    def update_cost_limit(self, cost_limit: Cost) -> None:
        self.cost_limit = cost_limit
        self.reset()

    def get_candidate_cost_limit(self) -> Cost:
        return self.candidate_cost_limit

    def reset(self) -> None:
        self.open_list.clear()
        self.closed_list.clear()
        self.open_list.append(self.initial_node)
        self.closed_list.add(self.initial_node)
        self.candidate_cost_limit = float("inf")

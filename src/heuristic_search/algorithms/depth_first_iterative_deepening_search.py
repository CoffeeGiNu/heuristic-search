from typing import Callable, Optional

from ..base import Cost, Node, StateSpaceProblem
from ..utils.logger import SearchLogger
from .cost_limited_depth_first_search import CostLimitedDepthFirstSearch


class DepthFirstIterativeDeepeningSearch(object):
    problem: StateSpaceProblem
    logger: SearchLogger
    cost_limit: Cost
    cost_limited_search: CostLimitedDepthFirstSearch

    def __init__(
        self,
        problem: StateSpaceProblem,
        logger: SearchLogger,
        priority_function: Callable = lambda node: node.depth,
        is_use_candidate_cost_limit: bool = False,
    ):
        self.problem: StateSpaceProblem = problem
        self.logger: SearchLogger = logger
        self.cost_limit: Cost = 0
        self.is_use_candidate_cost_limit: bool = is_use_candidate_cost_limit
        self.cost_limited_search: CostLimitedDepthFirstSearch = (
            CostLimitedDepthFirstSearch(
                problem=problem,
                logger=logger,
                cost_limit=self.cost_limit,
                priority_function=priority_function,
            )
        )

    def solve(self) -> Optional[list[Node]]:
        # TODO: loggerを各searchに持たせるのが良いかどうか要検討.
        # self.logger.start()
        while True:
            solution: Optional[list[Node]] = self.cost_limited_search.solve()
            if solution is not None:
                return solution
            if self.is_use_candidate_cost_limit:
                self.cost_limit = self.cost_limited_search.get_candidate_cost_limit()
            else:
                self.cost_limit += 1
            self.cost_limited_search.update_cost_limit(cost_limit=self.cost_limit)

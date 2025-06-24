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
    ):
        self.problem: StateSpaceProblem = problem
        self.logger: SearchLogger = logger
        self.cost_limit: Cost = 0
        self.cost_limited_search: CostLimitedDepthFirstSearch = (
            CostLimitedDepthFirstSearch(
                problem=problem,
                logger=logger,
                cost_limit=self.cost_limit,
                priority_function=lambda node: node.path_cost,
            )
        )

    def solve(self):
        # self.logger.start()
        # TODO: Implement iterative deepening search.
        pass

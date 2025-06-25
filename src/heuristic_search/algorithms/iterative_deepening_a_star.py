from ..base import StateSpaceProblem
from ..utils.logger import SearchLogger
from .depth_first_iterative_deepening_search import (
    DepthFirstIterativeDeepeningSearch,
)


class IterativeDeepeningAStar(DepthFirstIterativeDeepeningSearch):
    def __init__(self, problem: StateSpaceProblem, logger: SearchLogger):
        super().__init__(
            problem=problem,
            logger=logger,
            priority_function=lambda node: node.path_cost
            + problem.heuristic(state=node.state),
            is_use_candidate_cost_limit=True,
        )

from ..base import StateSpaceProblem
from ..utils import SearchLogger
from .graph_search import GraphSearch


class GreedyBestFirstSearch(GraphSearch):
    def __init__(
        self,
        problem: StateSpaceProblem,
        logger: SearchLogger,
    ) -> None:
        super().__init__(
            problem=problem,
            priority_function=lambda node: problem.heuristic(node.state),
            logger=logger,
        )

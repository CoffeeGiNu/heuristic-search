from ..base import StateSpaceProblem
from ..utils import SearchLogger
from .graph_search import GraphSearch


class AStarSearch(GraphSearch):
    def __init__(
        self,
        problem: StateSpaceProblem,
        logger: SearchLogger,
    ) -> None:
        super().__init__(
            problem=problem,
            priority_function=lambda node: node.path_cost
            + problem.heuristic(node.state),
            logger=logger,
        )


class WeightedAStarSearch(GraphSearch):
    def __init__(
        self,
        problem: StateSpaceProblem,
        logger: SearchLogger,
        weight: float = 1.0,
    ) -> None:
        super().__init__(
            problem=problem,
            priority_function=lambda node: node.path_cost
            + weight * problem.heuristic(node.state),
            logger=logger,
        )

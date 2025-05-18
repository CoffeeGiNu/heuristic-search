from ..base import StateSpaceProblem
from ..utils import SearchLogger
from .graph_search import GraphSearch


class DepthFirstSearch(GraphSearch):
    def __init__(
        self,
        problem: StateSpaceProblem,
        logger: SearchLogger,
    ) -> None:
        super().__init__(
            problem=problem,
            priority_function=lambda node: -node.depth,
            logger=logger,
        )

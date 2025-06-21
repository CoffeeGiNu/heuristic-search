from ..base import ClosedList, Node


class SetClosedList(ClosedList):
    """
    Closed list implementation using a set. Simple is best.
    """

    closed: set[Node]

    def __init__(self) -> None:
        self.closed: set[Node] = set()

    def push(self, node: Node) -> None:
        self.closed.add(node)

    def is_explored(self, node: Node) -> bool:
        return node in self.closed

from typing import Optional

from ..base import Cost, Node, OpenList


class BucketPriorityQueue(OpenList):
    min_priority: int
    max_priority: int
    buckets: list[list[Node]]
    size: int
    beam_width: Optional[int]

    def __init__(
        self,
        min_priority: int,
        max_priority: int,
        beam_width: Optional[int] = None,
    ) -> None:
        self.min_priority: int = min_priority
        self.max_priority: int = max_priority
        self.buckets: list[list[Node]] = [
            [] for _ in range(max_priority - min_priority + 1)
        ]
        self.size: int = 0
        self.beam_width: Optional[int] = beam_width

    def __len__(self) -> int:
        return self.size

    def pop(self) -> Node:
        for bucket in self.buckets:
            if bucket:
                self.size -= 1
                return bucket.pop()
        raise ValueError("Open list is empty.")

    def push(self, node: Node, priority: Cost) -> None:
        if not isinstance(priority, int):
            raise ValueError(f"Priority {priority} is not an integer")
        if priority < self.min_priority or priority > self.max_priority:
            raise ValueError(f"Priority {priority} is out of range.")
        self.buckets[priority - self.min_priority].append(node)
        self.size += 1

        if self.beam_width:
            self.shrink(self.beam_width)

    def shrink(self, beam_width: Optional[int] = None) -> None:
        if beam_width is None:
            beam_width = self.beam_width
        if beam_width:
            current_priority = self.max_priority - self.min_priority
            while self.size > beam_width:
                if self.buckets[current_priority]:
                    self.buckets[current_priority].pop()
                    self.size -= 1
                else:
                    current_priority -= 1

import time
from typing import Optional


# TODO: Add doctring.
# TODO: Add log file save method.
class SearchLogger(object):
    expanded: int
    generated: int
    pruned: int
    # log_file_path: Optional[str]
    start_perform_time: float
    start_time: float
    end_perform_time: float
    end_time: float

    def __init__(self, log_file_path: Optional[str] = None) -> None:
        self.expanded: int = 0
        self.generated: int = 0
        self.pruned: int = 0
        # self.log_file_path: Optional[str] = log_file_path
        self.start_perform_time: float = 0.0
        self.start_time: float = 0.0
        self.end_perform_time: float = 0.0
        self.end_time: float = 0.0

    def start(self) -> None:
        self.start_perform_time = time.perf_counter()
        self.start_time = time.time()

    def stop(self) -> None:
        self.end_perform_time = time.perf_counter()
        self.end_time = time.time()

    def branching_factor(self) -> float:
        return self.generated / self.expanded if self.expanded > 0 else 0.0

    def pruned_rate(self) -> float:
        return (
            self.pruned / (self.generated + self.pruned)
            if (self.generated + self.pruned) > 0
            else 0.0
        )

    def elapsed_time(self) -> float:
        return self.end_time - self.start_time if self.start_time > 0 else 0.0

    def elapsed_performance_time(self) -> float:
        return (
            self.end_perform_time - self.start_perform_time
            if self.start_perform_time > 0
            else 0.0
        )

    def expansion_rate(self) -> float:
        return self.expanded / self.elapsed_time() if self.elapsed_time() > 0 else 0.0

    def generation_rate(self) -> float:
        return self.generated / self.elapsed_time() if self.elapsed_time() > 0 else 0.0

    def print(self) -> None:
        print(f"Time: {self.elapsed_time():.3f} seconds")
        print(f"Performance time: {self.elapsed_performance_time():.3f} seconds")
        print(f"Expanded: {self.expanded}")
        print(f"Generated: {self.generated}")
        print(f"Pruned: {self.pruned}")
        print(f"Expansion rate: {self.expansion_rate():.3f} nodes/second")
        print(f"Generation rate: {self.generation_rate():.3f} nodes/second")
        print(f"Branching factor: {self.branching_factor():.3f}")
        print(f"Pruned rate: {self.pruned_rate():.3f}")

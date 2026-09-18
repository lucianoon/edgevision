from collections import deque
from time import perf_counter


class PerformanceMetrics:
    def __init__(self, window_size: int = 100):
        self.latencies = deque(maxlen=window_size)
        self.frame_count = 0
        self._start = None

    def start_frame(self):
        self._start = perf_counter()

    def end_frame(self):
        if self._start is None:
            return

        elapsed = perf_counter() - self._start

        self.latencies.append(elapsed * 1000)
        self.frame_count += 1
        self._start = None

    @property
    def average_latency_ms(self) -> float:
        if not self.latencies:
            return 0.0

        return sum(self.latencies) / len(self.latencies)

    @property
    def fps(self) -> float:
        latency = self.average_latency_ms

        if latency == 0:
            return 0.0

        return 1000 / latency

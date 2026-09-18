from collections import defaultdict, deque
from contextlib import contextmanager
from statistics import mean, median
from time import perf_counter

END_TO_END = "end_to_end"


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[index]


class PerformanceMetrics:
    """Rolling per-stage latencies (ms). Stages are timed with `stage(name)`;
    `start_frame()/end_frame()` time the end-to-end stage and count frames."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.stages: dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.frame_count = 0
        self._start = None

    @property
    def latencies(self) -> deque:
        return self.stages[END_TO_END]

    def record(self, stage: str, latency_ms: float):
        self.stages[stage].append(latency_ms)

    @contextmanager
    def stage(self, name: str):
        start = perf_counter()
        try:
            yield
        finally:
            self.record(name, (perf_counter() - start) * 1000)

    def start_frame(self):
        self._start = perf_counter()

    def end_frame(self):
        if self._start is None:
            return
        self.record(END_TO_END, (perf_counter() - self._start) * 1000)
        self.frame_count += 1
        self._start = None

    def average_latency(self, stage: str = END_TO_END) -> float:
        values = self.stages.get(stage)
        return mean(values) if values else 0.0

    @property
    def average_latency_ms(self) -> float:
        return self.average_latency(END_TO_END)

    @property
    def fps(self) -> float:
        latency = self.average_latency_ms
        return 1000 / latency if latency else 0.0

    def summary(self) -> dict[str, dict[str, float]]:
        """Per-stage mean / p50 / p95 / max in ms, for the current window."""
        result = {}
        for name, values in self.stages.items():
            if not values:
                continue
            samples = list(values)
            result[name] = {
                "mean_ms": mean(samples),
                "p50_ms": median(samples),
                "p95_ms": _percentile(samples, 95),
                "max_ms": max(samples),
                "samples": len(samples),
            }
        return result

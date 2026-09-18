from edgevision.metrics import PerformanceMetrics


def test_empty_metrics_are_zero():
    metrics = PerformanceMetrics()

    assert metrics.average_latency_ms == 0.0
    assert metrics.fps == 0.0
    assert metrics.frame_count == 0


def test_end_without_start_is_ignored():
    metrics = PerformanceMetrics()

    metrics.end_frame()

    assert metrics.frame_count == 0


def test_latency_and_fps_from_recorded_frames():
    metrics = PerformanceMetrics(window_size=2)

    metrics.latencies.extend([10.0, 30.0])
    metrics.frame_count = 2

    assert metrics.average_latency_ms == 20.0
    assert metrics.fps == 50.0


def test_window_keeps_only_latest_samples():
    metrics = PerformanceMetrics(window_size=2)

    for _ in range(3):
        metrics.start_frame()
        metrics.end_frame()

    assert len(metrics.latencies) == 2
    assert metrics.frame_count == 3


def test_stage_context_manager_records_named_latency():
    metrics = PerformanceMetrics()

    with metrics.stage("preprocess"):
        pass

    summary = metrics.summary()
    assert set(summary) == {"preprocess"}
    assert summary["preprocess"]["samples"] == 1
    assert summary["preprocess"]["mean_ms"] >= 0.0


def test_summary_percentiles():
    metrics = PerformanceMetrics()
    for value in [10.0, 20.0, 30.0, 40.0, 100.0]:
        metrics.record("inference", value)

    stats = metrics.summary()["inference"]
    assert stats["mean_ms"] == 40.0
    assert stats["p50_ms"] == 30.0
    assert stats["p95_ms"] == 100.0
    assert stats["max_ms"] == 100.0

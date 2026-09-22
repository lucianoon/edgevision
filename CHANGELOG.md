# Changelog

All notable changes to EdgeVision. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/). Measurements live in `benchmarks/README.md`.

## [Unreleased]

### Added
- Events layer (`edgevision.events`, command `edgevision-events`): zones (polygons, per-class,
  dwell threshold), counting lines with direction labels, lost-track and end-of-stream handling,
  JSONL / stdout / webhook sinks, YAML rules with validation (`configs/rules.example.yaml`).
  35 tests on synthetic trajectories plus a regression on the real T4 track dump
  (`tests/data/tracks_pedestrian_512.jsonl`: 6 east / 8 west gate crossings, 44 zone entries, 3 dwell alerts).
- Observability: `--metrics-port` on `edgevision` and `edgevision-events` serves Prometheus metrics
  (`/metrics`, `/healthz`) from a dependency-free exporter (`edgevision.observability`); `observability/`
  holds a Prometheus + Grafana compose stack with a provisioned dashboard (fps, latency window, stage
  breakdown, detections rate, line crossings, zone occupancy, events by type).

## [0.2.0] - 2026-09-21

Quality hardening: no change to what the runtime measures; the repository becomes buildable, testable and
reviewable by others. Validated on the T4 with `scripts/gpu_check_runtime.sh` (build, CTests, both decoders,
multi-stream, batched, tracking, parity tests).

### Added
- `edgevision` is a pip-installable package with `edgevision` and `edgevision-benchmark` console scripts.
- `requirements.lock` (exact CPU environment) and `requirements-dev.txt`; the CI installs from them.
- C++ `edgevision_cpu` library (tracker, metrics, class names, CLI parsing, JSON report) built without
  CUDA under `-DEDGEVISION_CPU_ONLY=ON`, with strict warnings and three CTests (`tracker`, `cli`, `report`).
- CI gates: ruff (lint + format), mypy, shellcheck, pytest coverage floor (80%), clang-format, clang-tidy.
- Unit tests for the Python frame loop, benchmark loop, video source and factory (coverage 96%).
- `CONTRIBUTING.md`, this changelog, Dependabot, pre-commit and EditorConfig.
- `scripts/gpu_check_runtime.sh`: build-and-smoke check of the C++ runtime on the GPU box.

### Changed
- ONNX graphs and TensorRT engines now define the pre-processing size; `image_size` in `app.yaml`
  applies to the PyTorch backend and dynamic graphs only. The benchmark report records the effective size.
- `main.py` and `benchmark.py` expose `process_stream()`, `measure()` and `build_report()`; the OpenCV
  window is a per-frame callback.
- `cpp/src/main.cpp` only wires `cli.cpp`, `stream_runner.cpp`, `batch_pipeline.cpp` and `report.cpp`.
- `YoloDetector` reads boxes vectorised from the Ultralytics result; `build_detector()` returns the
  `Detector` protocol.
- Whole Python and C++ trees formatted (`ruff format`, `cpp/.clang-format`).

### Fixed
- The Python TensorRT path pre-processed at 640 for the default 512 engine (upload would fail on reshape).
- Six shellcheck warnings in the GPU and AWS scripts; a raw carriage return in `infra/README.md`.

## [0.1.0] - 2026-09-21

End of Sprint 8. Everything below was measured on an EC2 g4dn.xlarge (Tesla T4) with 1080p H.264
input and YOLO26n; raw reports are in `benchmarks/results/`.

### Added
- Sprint 1: PyTorch/Ultralytics baseline with webcam, FPS counter and per-stage metrics.
- Sprint 2: own letterbox, decode and NMS; ONNX Runtime backend; benchmark harness with JSON reports.
- Sprint 3: TensorRT FP32/FP16 engines via `trtexec`, Python TensorRT runtime, CloudFormation GPU
  dev box with SSM-only access, idle/uptime cost guards and zero-cost standby.
- Sprint 4: NMS inside the graph; C++ TensorRT runtime with a CUDA letterbox kernel and pinned frames
  (3.0 ms/frame).
- Sprint 5: NVDEC hardware decode through libav and RTSP input (2.0 ms + 0.4 ms decode wait).
- Sprint 6: N independent streams per GPU and batched inference with ping-pong buffers; 12 live
  RTSP cameras at 25 fps with zero drops, 675 fps aggregate ceiling.
- Sprint 7: accuracy vs speed on COCO val2017 (mAP50-95 0.404 @ 640, 0.378 @ 512); INT8 PTQ
  rejected (-3.5 mAP, unbuildable with NMS in graph). Production default: FP16 @ 512.
- Sprint 8: dependency-free ByteTrack (Kalman + Hungarian, two passes) in the C++ runtime,
  tens of microseconds per frame; tracking demo and annotated frame.
- MIT license; GitHub Actions CI for the CPU path.

[Unreleased]: https://github.com/lucianoon/edgevision/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/lucianoon/edgevision/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/lucianoon/edgevision/releases/tag/v0.1.0

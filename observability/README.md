# Observability

Both Python entry points can serve Prometheus metrics while they run; this folder holds a
Prometheus + Grafana stack that scrapes them and a provisioned dashboard.

```bash
# terminal 1: the app (or the C++ runtime piped into the events layer, see below)
edgevision --backend onnx --source videos/pedestrian_area_1080p25.webm --no-display --metrics-port 9108

# terminal 2: events over a track stream
edgevision-events benchmarks/results/tracks_cpp_512.jsonl --rules configs/rules.example.yaml --metrics-port 9109 --out events.jsonl

# terminal 3: the stack
docker compose -f observability/docker-compose.yml up
```

Grafana: http://localhost:3000 (anonymous admin, development only). Prometheus: http://localhost:9090.

## Metrics

| metric | type | labels | from |
|---|---|---|---|
| `edgevision_build_info` | gauge | version, backend | both |
| `edgevision_fps` | gauge | backend | app: end-to-end mean of the window |
| `edgevision_frames_total` | counter | backend | app |
| `edgevision_detections_total` | counter | backend | app |
| `edgevision_stage_latency_window_ms` | gauge | backend, stage, stat=mean/p50/p95/max | app: last N frames per stage |
| `edgevision_stage_window_samples` | gauge | backend, stage | app |
| `edgevision_line_crossings_total` | counter | line, label | events |
| `edgevision_zone_entries_total` | counter | zone | events |
| `edgevision_zone_occupancy` | gauge | zone | events |
| `edgevision_zone_dwell_alerts_total` | counter | zone | events |
| `edgevision_tracks_active` | gauge | | events |
| `edgevision_events_total` | counter | type | events |

Latency metrics are statistics of the metrics window (default 100 frames), not histograms: they
answer "how is the stream doing right now" cheaply and match the columns of the benchmark tables.
`/healthz` returns `ok` while the process runs; both endpoints are served from a daemon thread and
never block the frame loop.

The exporter is `python/edgevision/observability.py`, dependency-free (text exposition format
0.0.4 over `http.server`). The C++ runtime does not expose metrics itself yet; its tracks reach
Prometheus through the events layer.

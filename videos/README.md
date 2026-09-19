# Benchmark clips (not versioned; re-download with the commands below)

| file | source | license | notes |
|------|--------|---------|-------|
| `sample.mp4` | generated from Ultralytics `bus.jpg` (60 identical frames, 810x1080) | - | Sprint 1-3 smoke/benchmark clip; unrealistic (still image) |
| `traffic.mp4` | https://github.com/ultralytics/assets/releases/download/v0.0.0/solutions_ci_demo.mp4 | Ultralytics assets | 640x360, 62 frames; too small for 1080p benchmarks |
| `pedestrian_area_1080p25.webm` | https://commons.wikimedia.org/wiki/File:Video_Codec_Test_pedestrian_area_1080p25.y4m.webm | CC0 (Taurus Media Technik) | 1920x1080 @ 25 fps, crowded pedestrians: many overlapping `person` boxes, good NMS load. **Default benchmark clip from Sprint 4 on.** |
| `pedestrian_area_1080p25_h264.mp4` | re-encode of the clip above (`ffmpeg -c:v libx264 -crf 20`, done by `scripts/gpu_sprint5.sh` on the box) | CC0 | H.264 for the NVDEC/RTSP tests, what real cameras send |
| `street_traffic_1080p.webm` | https://commons.wikimedia.org/wiki/File:Street_traffic.webm | CC BY 3.0 | 1920x1080, cars/buses/trucks; secondary clip |

```bash
curl -L -o videos/pedestrian_area_1080p25.webm "https://upload.wikimedia.org/wikipedia/commons/a/ae/Video_Codec_Test_pedestrian_area_1080p25.y4m.webm"
curl -L -o videos/street_traffic_1080p.webm "https://upload.wikimedia.org/wikipedia/commons/7/70/Street_traffic.webm"
```

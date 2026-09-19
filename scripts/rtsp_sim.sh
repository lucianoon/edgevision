#!/usr/bin/env bash
# Host-side RTSP camera simulator for the GPU box: MediaMTX server + ffmpeg pushing the
# 1080p clip in a loop as H.264. Then, inside the container (with --network host):
#   cpp/build/edgevision_trt --engine ... --source rtsp://127.0.0.1:8554/cam --decoder nvdec
#   scripts/rtsp_sim.sh start|stop
set -euo pipefail
CLIP=${CLIP:-/opt/edgevision/repo/videos/pedestrian_area_1080p25_h264.mp4}
case "${1:-start}" in
  start)
    [ -f "$CLIP" ] || { echo "clip not found: $CLIP (run scripts/gpu_sprint5.sh once to create the H.264 file)" >&2; exit 1; }
    docker rm -f mediamtx >/dev/null 2>&1 || true
    docker run -d --name mediamtx --network host bluenviron/mediamtx:latest >/dev/null
    sleep 2
    docker rm -f rtsp-push >/dev/null 2>&1 || true
    docker run -d --name rtsp-push --network host -v "$(dirname "$CLIP"):/videos:ro" linuxserver/ffmpeg:latest \
        -re -stream_loop -1 -i "/videos/$(basename "$CLIP")" -c:v copy -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/cam >/dev/null
    echo "rtsp://127.0.0.1:8554/cam is being served (25 fps, H.264, looped)"
    ;;
  stop)
    docker rm -f rtsp-push mediamtx >/dev/null 2>&1 || true
    echo "stopped"
    ;;
esac

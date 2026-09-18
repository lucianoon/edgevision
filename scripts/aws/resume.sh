#!/usr/bin/env bash
# Leave standby: launch a fresh instance from the AMI, wait for SSM, push code +
# models and rebuild the docker image (the previous disk was deleted on standby).
set -euo pipefail
source "$(dirname "$0")/common.sh"
"$(dirname "$0")/deploy.sh" LaunchInstance=true "$@"
"$(dirname "$0")/sync-up.sh"
TIMEOUT_SECONDS=3600 "$(dirname "$0")/run.sh" \
    "docker build -f docker/Dockerfile.tensorrt -t edgevision:trt . 2>&1 | tail -3 && nvidia-smi --query-gpu=name,driver_version --format=csv"

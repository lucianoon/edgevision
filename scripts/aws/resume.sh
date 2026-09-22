#!/usr/bin/env bash
# Leave standby: launch a fresh instance from the AMI, wait for SSM, push code +
# models and rebuild the docker image (the previous disk was deleted on standby).
set -euo pipefail
source "$(dirname "$0")/common.sh"
"$(dirname "$0")/deploy.sh" LaunchInstance=true "$@"
# SSM can come online before the user data has created /opt/edgevision (seen on g5g):
# wait for cloud-init before pushing code.
"$(dirname "$0")/run.sh" "cloud-init status --wait >/dev/null; ls -ld /opt/edgevision"
"$(dirname "$0")/sync-up.sh"
TIMEOUT_SECONDS=3600 "$(dirname "$0")/run.sh" \
    "docker build -f docker/Dockerfile.tensorrt -t edgevision:trt . 2>&1 | tail -3 && nvidia-smi --query-gpu=name,driver_version --format=csv"

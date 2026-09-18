#!/usr/bin/env bash
# Zero-cost standby: sync results/engines to S3, then remove the instance AND its
# EBS volume (LaunchInstance=false). Network, role, launch template and bucket stay
# and cost nothing. Resume with scripts/aws/resume.sh (fresh instance from the AMI).
set -euo pipefail
source "$(dirname "$0")/common.sh"

ID=$(instance_id)
if [ -n "$ID" ] && [ "$ID" != "None" ]; then
    if [ "$(instance_state)" = "stopped" ]; then
        echo "instance is stopped; starting it briefly to sync artifacts to S3"
        "$(dirname "$0")/start.sh"
    fi
    "$(dirname "$0")/sync-down.sh" || echo "WARNING: sync-down failed; continuing (check S3 before relying on results)"
fi

cd "$REPO_ROOT"
aws cloudformation deploy \
    --stack-name "$STACK_NAME" --region "$AWS_REGION" \
    --template-file infra/gpu-dev.yaml \
    --capabilities CAPABILITY_NAMED_IAM --tags Project=edgevision \
    --parameter-overrides LaunchInstance=false
echo "standby: instance and volume removed. Remaining cost: S3 objects only (cents)."

#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
aws ec2 start-instances --instance-ids "$(instance_id)" --region "$AWS_REGION" --output text >/dev/null
wait_for_ssm

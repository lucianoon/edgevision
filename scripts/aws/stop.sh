#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
aws ec2 stop-instances --instance-ids "$(instance_id)" --region "$AWS_REGION" --output text >/dev/null
echo "stopping $(instance_id) (EBS is kept; only storage is billed while stopped)"

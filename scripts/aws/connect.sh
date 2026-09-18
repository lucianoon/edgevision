#!/usr/bin/env bash
# Interactive shell on the GPU box via SSM Session Manager (no SSH, no open ports).
set -euo pipefail
source "$(dirname "$0")/common.sh"
exec aws ssm start-session --target "$(instance_id)" --region "$AWS_REGION"

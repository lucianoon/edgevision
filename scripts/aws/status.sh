#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
echo "stack:    $(aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$AWS_REGION" --query 'Stacks[0].StackStatus' --output text)"
echo "instance: $(instance_id) ($(instance_state))"
echo "bucket:   $(bucket_name)"

#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
echo "stack:    $(aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$AWS_REGION" --query 'Stacks[0].StackStatus' --output text)"
ID=$(instance_id)
if [ -z "$ID" ] || [ "$ID" = "None" ]; then
    echo "instance: none (standby - only S3 objects are billed)"
else
    echo "instance: $ID ($(instance_state))"
fi
echo "bucket:   $(bucket_name)"

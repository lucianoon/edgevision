#!/usr/bin/env bash
# Delete the stack (instance, network, role). The S3 bucket is retained on purpose.
set -euo pipefail
source "$(dirname "$0")/common.sh"
aws cloudformation delete-stack --stack-name "$STACK_NAME" --region "$AWS_REGION"
echo "deleting $STACK_NAME ..."
aws cloudformation wait stack-delete-complete --stack-name "$STACK_NAME" --region "$AWS_REGION"
echo "deleted. Bucket kept: $(aws s3 ls --region "$AWS_REGION" 2>/dev/null | grep -o 'edgevision-[^ ]*' || echo '(list with aws s3 ls)')"

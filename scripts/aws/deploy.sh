#!/usr/bin/env bash
# Create or update the GPU dev stack. Extra args are passed as parameter overrides,
# e.g. scripts/aws/deploy.sh InstanceType=g6.xlarge MaxUptimeHours=4
set -euo pipefail
source "$(dirname "$0")/common.sh"

cd "$REPO_ROOT"  # relative paths: Git Bash's /c/... form breaks cfn-lint's glob
cfn-lint infra/gpu-dev.yaml

aws cloudformation deploy \
    --stack-name "$STACK_NAME" \
    --region "$AWS_REGION" \
    --template-file infra/gpu-dev.yaml \
    --capabilities CAPABILITY_NAMED_IAM \
    --tags Project=edgevision \
    ${1:+--parameter-overrides "$@"}

aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$AWS_REGION" \
    --query 'Stacks[0].Outputs' --output table

ID=$(instance_id)
if [ -z "$ID" ] || [ "$ID" = "None" ]; then
    echo "no instance in this stack (LaunchInstance=false)"
else
    wait_for_ssm
fi

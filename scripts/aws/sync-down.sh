#!/usr/bin/env bash
# Pull benchmark results and engines produced on the box back into the repo.
set -euo pipefail
source "$(dirname "$0")/common.sh"
BUCKET=$(bucket_name)

"$(dirname "$0")/run.sh" "cd /opt/edgevision/repo \
 && aws s3 sync benchmarks/results s3://$BUCKET/benchmarks/results \
 && aws s3 sync models/tensorrt s3://$BUCKET/models/tensorrt --exclude '.gitkeep'"

cd "$REPO_ROOT"
aws s3 sync "s3://$BUCKET/benchmarks/results" benchmarks/results --region "$AWS_REGION"
aws s3 sync "s3://$BUCKET/models/tensorrt" models/tensorrt --region "$AWS_REGION"
ls -la benchmarks/results models/tensorrt

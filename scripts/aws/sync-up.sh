#!/usr/bin/env bash
# Push the working tree (tracked + untracked, minus .gitignore) + ONNX models and the
# sample video to S3, then unpack on the box.
set -euo pipefail
source "$(dirname "$0")/common.sh"
BUCKET=$(bucket_name)

cd "$REPO_ROOT"
TMP=$(mktemp -d)
TMP_NATIVE=$(cygpath -w "$TMP" 2>/dev/null || echo "$TMP")  # aws.exe on Git Bash needs a Windows path
git ls-files -z --cached --others --exclude-standard | tar --null -T - -czf "$TMP/repo.tgz"
aws s3 cp "$TMP_NATIVE/repo.tgz" "s3://$BUCKET/code/repo.tgz" --region "$AWS_REGION"
aws s3 sync models/onnx "s3://$BUCKET/models/onnx" --region "$AWS_REGION" --exclude ".gitkeep"
aws s3 sync videos "s3://$BUCKET/videos" --region "$AWS_REGION" --exclude ".gitkeep"
rm -rf "$TMP"

"$(dirname "$0")/run.sh" "mkdir -p /opt/edgevision/repo && cd /opt/edgevision/repo \
 && aws s3 cp s3://$BUCKET/code/repo.tgz /tmp/repo.tgz && tar -xzf /tmp/repo.tgz \
 && aws s3 sync s3://$BUCKET/models/onnx models/onnx \
 && aws s3 sync s3://$BUCKET/videos videos && ls models/onnx videos"

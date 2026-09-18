#!/usr/bin/env bash
# Run a command on the GPU box as the ubuntu user, inside /opt/edgevision/repo,
# through SSM Run Command, streaming the output back. Exit code is propagated.
#   scripts/aws/run.sh 'nvidia-smi'
#   scripts/aws/run.sh 'docker compose ...'  TIMEOUT_SECONDS=3600 for long jobs
set -euo pipefail
source "$(dirname "$0")/common.sh"
: "${TIMEOUT_SECONDS:=1800}"

# sudo -i resets the cwd to the home dir, so the repo path must be spelled out inside the login shell.
CMD="sudo -u ubuntu -i bash -lc $(printf '%q' "cd /opt/edgevision/repo 2>/dev/null || cd /opt/edgevision || cd ~; $*")"
CMD_ID=$(aws ssm send-command --region "$AWS_REGION" \
    --instance-ids "$(instance_id)" \
    --document-name AWS-RunShellScript \
    --timeout-seconds "$TIMEOUT_SECONDS" \
    --parameters "{\"commands\":[$(printf '%s' "$CMD" | python -c 'import json,sys; print(json.dumps(sys.stdin.read()))')],\"executionTimeout\":[\"$TIMEOUT_SECONDS\"]}" \
    --query 'Command.CommandId' --output text)

while :; do
    STATUS=$(aws ssm get-command-invocation --region "$AWS_REGION" --command-id "$CMD_ID" \
        --instance-id "$(instance_id)" --query Status --output text 2>/dev/null || echo Pending)
    case "$STATUS" in
        Pending|InProgress|Delayed) sleep 5 ;;
        *) break ;;
    esac
done

aws ssm get-command-invocation --region "$AWS_REGION" --command-id "$CMD_ID" \
    --instance-id "$(instance_id)" --query 'StandardOutputContent' --output text
aws ssm get-command-invocation --region "$AWS_REGION" --command-id "$CMD_ID" \
    --instance-id "$(instance_id)" --query 'StandardErrorContent' --output text >&2
[ "$STATUS" = "Success" ] || { echo "remote command status: $STATUS" >&2; exit 1; }

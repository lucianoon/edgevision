# Shared settings for the EdgeVision GPU dev box scripts. Source, don't run.
# Works from Git Bash (Windows) and WSL/Linux.
export MSYS_NO_PATHCONV=1
export AWS_PAGER=""
: "${AWS_PROFILE:=default}"
: "${AWS_REGION:=us-east-1}"
: "${STACK_NAME:=edgevision-gpu-dev}"
export AWS_PROFILE AWS_REGION STACK_NAME

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

stack_output() {  # stack_output <OutputKey>
    aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$AWS_REGION" \
        --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text
}

instance_id() { stack_output InstanceId; }
bucket_name() { stack_output BucketName; }

instance_state() {
    aws ec2 describe-instances --instance-ids "$(instance_id)" --region "$AWS_REGION" \
        --query 'Reservations[0].Instances[0].State.Name' --output text
}

wait_for_ssm() {  # block until the instance is an Online SSM managed node
    local id; id=$(instance_id)
    echo "waiting for SSM agent on $id ..."
    for _ in $(seq 1 60); do
        local status
        status=$(aws ssm describe-instance-information --region "$AWS_REGION" \
            --filters "Key=InstanceIds,Values=$id" \
            --query 'InstanceInformationList[0].PingStatus' --output text 2>/dev/null || true)
        [ "$status" = "Online" ] && { echo "SSM online"; return 0; }
        sleep 10
    done
    echo "SSM agent did not come online" >&2; return 1
}

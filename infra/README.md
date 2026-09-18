# GPU dev box on AWS (Sprint 3+)

The laptop has no NVIDIA GPU, so TensorRT engines are built and benchmarked on a
single EC2 GPU instance defined in `gpu-dev.yaml` (CloudFormation).

```
laptop ──git ls-files→ tar → S3 ──→ EC2 g4dn.xlarge (T4)  ── docker (TensorRT) ──┐
   ▲                                    │ trtexec → engines, benchmark JSON         │
   └──────────── S3 sync ◄──────────────┴──────────────────────────────────────────┘
```

Design points (also recorded in the template `Metadata`):

- **Access only through SSM Session Manager.** No SSH key, no inbound rule.
- **IMDSv2 required**, hop limit 2 so containers can reach the instance role.
- **Deep Learning Base OSS NVIDIA Driver GPU AMI (Ubuntu 24.04)** resolved from
  Parameter Store: driver, Docker and NVIDIA container toolkit preinstalled.
- **Two cost guards:** CloudWatch alarm stops the instance after 1 h with CPU < 5 %
  (`IdleStopPeriods`), and a systemd unit powers it off `MaxUptimeHours` (8) after
  every boot. Both *stop*; the EBS volume and everything on it survive.
- **S3 bucket `edgevision-<account>-<region>`** is the only transfer channel and is
  retained when the stack is deleted.
- Everything is tagged `Project=edgevision`; the IAM user policy in
  `iam-policy-edgevision-dev.json` is scoped on that tag, on `edgevision-*` names
  and on `us-east-1`.

## One-time: IAM permissions for the CLI user

The day-to-day IAM user (`luciano`) has no EC2/CloudFormation rights. Attach the
scoped policy once with an administrative identity:

```bash
aws iam create-policy --policy-name edgevision-gpu-dev \
    --policy-document file://infra/iam-policy-edgevision-dev.json --profile <admin>
aws iam attach-user-policy --user-name luciano \
    --policy-arn arn:aws:iam::<account>:policy/edgevision-gpu-dev --profile <admin>
```

Instances of the G family also need vCPU quota (`L-DB2E81BA`, "Running On-Demand
G and VT instances"); new accounts often start at 0 and must request >= 4.

## Lifecycle (from Git Bash or WSL, in the repo root)

| step | command |
|------|---------|
| create / update stack | `scripts/aws/deploy.sh [InstanceType=g6.xlarge ...]` |
| status | `scripts/aws/status.sh` |
| shell on the box | `scripts/aws/connect.sh` |
| run one command remotely | `scripts/aws/run.sh 'nvidia-smi'` |
| push code + ONNX | `scripts/aws/sync-up.sh` |
| pull results + engines | `scripts/aws/sync-down.sh` |
| stop / start | `scripts/aws/stop.sh` / `scripts/aws/start.sh` |
| delete everything but the bucket | `scripts/aws/destroy.sh` |

`AWS_PROFILE`, `AWS_REGION` and `STACK_NAME` can be overridden in the environment.

## On the box

```bash
cd /opt/edgevision/repo
docker build -f docker/Dockerfile.tensorrt -t edgevision:trt .
docker run --rm --gpus all -v $PWD:/workspace/edgevision edgevision:trt \
    bash -lc 'scripts/build_engine.sh && \
              python -m edgevision.benchmark --backend tensorrt --model models/tensorrt/yolo26n_fp32.engine --label fp32 && \
              python -m edgevision.benchmark --backend tensorrt --model models/tensorrt/yolo26n_fp16.engine --label fp16 && \
              python -m edgevision.benchmark --backend onnx --label cuda && \
              python -m edgevision.benchmark --backend pytorch --label cuda && \
              pytest -q'
```

Pick the NGC TensorRT tag (`--build-arg TRT_TAG=`) whose CUDA major is supported by
the AMI's driver (`nvidia-smi` shows the max CUDA version). 26.04 = TensorRT 10.16 /
CUDA 13.2. Engines are specific to the GPU model and TensorRT version: a T4 engine
does not run on a Jetson.

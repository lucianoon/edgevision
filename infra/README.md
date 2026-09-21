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
- **Two cost guards:** CloudWatch alarm stops the instance after 30 min with CPU < 5 %
  (`IdleStopPeriods`), and a systemd unit powers it off `MaxUptimeHours` (4) after
  every boot. Both *stop*; the EBS volume and everything on it survive.
- **Zero-cost standby:** `scripts/aws/standby.sh` syncs engines + results to S3 and
  removes the instance *and its volume* (`LaunchInstance=false`). What remains
  (VPC, role, launch template, bucket) costs nothing. `scripts/aws/resume.sh`
  launches a fresh instance, pushes code/models and rebuilds the docker image.
- **S3 bucket `edgevision-<account>-<region>`** is the only transfer channel and is
  retained when the stack is deleted.
- Everything is tagged `Project=edgevision`; the IAM user policy in
  `iam-policy-edgevision-dev.json` is scoped on that tag, on `edgevision-*` names
  and on `us-east-1`.

## One-time: IAM permissions for the CLI user

The day-to-day IAM user (`<iam-user>`) has no EC2/CloudFormation rights. Attach the
scoped policy once with an administrative identity:

```bash
sed -i "s/<ACCOUNT_ID>/$(aws sts get-caller-identity --query Account --output text)/g" infra/iam-policy-edgevision-dev.json
aws iam create-policy --policy-name edgevision-gpu-dev \
    --policy-document file://infra/iam-policy-edgevision-dev.json --profile <admin>
aws iam attach-user-policy --user-name <iam-user> \
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
| stop / start (keeps disk, ~US$ 8/month) | `scripts/aws/stop.sh` / `scripts/aws/start.sh` |
| **standby (zero cost) / resume** | `scripts/aws/standby.sh` / `scripts/aws/resume.sh` |
| delete everything but the bucket | `scripts/aws/destroy.sh` |

`AWS_PROFILE`, `AWS_REGION` and `STACK_NAME` can be overridden in the environment.

Notes learned on the first run (2026-09-18):

- `aws cloudformation deploy` keeps the *previous* value of any parameter you do not
  override, so new template defaults only apply when passed explicitly
  (e.g. `deploy.sh MaxUptimeHours=4 IdleStopPeriods=6`).
- After a vCPU quota increase is approved, Service Quotas shows the new value a few
  minutes before EC2 enforces it; `run-instances --dry-run` does **not** check the
  quota, so a real launch is the only test. A failed update rolls back cleanly.
- The idle-stop alarm needs `cloudwatch:PutMetricAlarm` on the deploying identity
  (statement `IdleStopAlarm` in the policy file). Until that policy version is
  attached, deploy with `CreateIdleAlarm=false`; the in-instance uptime cap still runs.
- Inside `scripts/aws/run.sh` commands, use absolute paths for docker bind mounts
  (`-v /opt/edgevision/repo:/workspace/edgevision`), not `$PWD`.
- NVDEC inside the container needs the `video` driver capability
  (`ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video` in the Dockerfile); without it
  `av_hwdevice_ctx_create(CUDA)` fails because libnvcuvid is not mounted.
- Long runs survive the laptop going idle only because they run detached on the box (SSM
  Run Command) and write to `benchmarks/results/`; the local watcher may be killed. Check
  the log on the box, then `sync-down`. The uptime cap is what limited a forgotten
  overnight instance to 4 h of cost.
- `aws login` while the browser is signed in as root offers to rebind `default` to root
  and aborts without a tty; sign in to the console as the IAM user first.
- Files patched with Python on Windows come out CRLF unless written with
  `newline="
"`; a CRLF shell script on the box fails with
  `set: pipefail: invalid option name`. `git ls-files --eol | grep w/crlf` finds them
  (Git Bash's `grep $''` does not).
- Keep remote logic in scripts inside the repo (`scripts/gpu_*.sh`) and call them with
  `run.sh 'bash scripts/x.sh'`; inline multi-line commands lose `$VAR`/`$!` through the
  sudo / SSM quoting layers.

## Cost (us-east-1 on-demand, Pricing API 2026-09-18)

| item | price |
|------|-------|
| g4dn.xlarge (T4 16 GB) running | US$ 0.526/h (+ US$ 0.005/h public IPv4) |
| gp3 100 GB root volume | US$ 8.00/month, billed while the instance exists (running or stopped) |
| CloudWatch alarm | US$ 0.10/month while the instance exists |
| VPC, IGW, SG, IAM, SSM, launch template | free (no NAT gateway) |
| S3 artifacts (< 1 GB) | cents |

A full Sprint 3 run (image build, engines, four benchmarks, tests) is ~2 h ≈ US$ 1.10;
with debugging ~4 h ≈ US$ 2.15. In standby the account pays only for S3 objects.
Resuming costs the image rebuild again (~15 min ≈ US$ 0.15).

## On the box

```bash
cd /opt/edgevision/repo
docker build -f docker/Dockerfile.tensorrt -t edgevision:trt .   # ~5 min, 9.8 GB
docker run --rm --gpus all -v /opt/edgevision/repo:/workspace/edgevision \
    -e FRAMES=300 -e WARMUP=30 edgevision:trt scripts/gpu_sprint3.sh
```

`scripts/gpu_sprint3.sh` builds the FP32/FP16 engines, benchmarks the four backends
(TensorRT FP32, TensorRT FP16, ONNX Runtime CUDA, PyTorch CUDA) on the same clip,
runs the tests and leaves a log next to the JSON reports in `benchmarks/results/`.
`scripts/gpu_nms_compare.sh` compares the raw-head and NMS-in-graph engines;
`scripts/gpu_sprint4.sh` builds and benchmarks the C++ runtime (`cpp/`). All three are
meant to be run as the container command, as above.

The image rebuild after a standby takes 5-13 min (the 28 GB unpack is bound by the gp3
volume); a fresh box also has no engines, so the scripts rebuild them (~6 min).

Pick the NGC TensorRT tag (`--build-arg TRT_TAG=`) whose CUDA major is supported by
the AMI's driver (`nvidia-smi` shows the max CUDA version). 26.04 = TensorRT 10.16 /
CUDA 13.2. Engines are specific to the GPU model and TensorRT version: a T4 engine
does not run on a Jetson.

# Step 6 — AWS + Kubernetes Setup

## Date: 2026-03-24

---

## Session Summary

### WSL2 Migration (documented retroactively)

Before starting the AWS setup, the WSL2 migration was completed and logged. See `05_wsl2_aws_kubernetes.md` for the full account. Short version:
- WSL2 + Ubuntu (Store) installed manually
- Project copied from Windows → WSL2 native filesystem by Claude
- venv re-created in WSL2
- Windows project copy deleted by Claude
- Motivation: W&B daemon fails to spawn on Windows

---

## Phase 1 — Tool Installation (2026-03-24) ✅ COMPLETE

All CLI tools needed for AWS + Kubernetes were installed from scratch in WSL2.

### Tools Installed

| Tool | Version | Purpose |
|------|---------|---------|
| aws-cli | 2.34.16 | Interact with all AWS services |
| docker | 29.3.0 | Build + run container images |
| kubectl | v1.32.0 | Control Kubernetes clusters |
| eksctl | 0.224.0 | Create/manage EKS clusters (one-command cluster creation) |

### What eksctl is

`eksctl` is a CLI tool that creates a full EKS (Elastic Kubernetes Service) cluster with one command — handles VPC, node groups, IAM roles, etc. automatically. Without it you'd configure ~10 AWS resources by hand.

### Installation Notes

**Copy-paste issue:** WSL2 terminal was wrapping long commands across multiple lines when pasting, causing the shell to treat the second line as a separate command. This broke several installs. Workarounds used:
- Split piped commands into two separate steps (curl to file, then process file)
- Used heredoc (`<< 'EOF'`) instead of `echo | sudo tee` for writing files
- Wrote files via Claude to `/tmp/` then `sudo cp` to destination
- Used hardcoded version numbers instead of `$(curl ...)` substitutions when those failed

**AWS CLI:** Downloaded zip from `awscli.amazonaws.com`, unzipped, ran `sudo ./aws/install`

**Docker:**
```bash
# GPG key — split into two steps to avoid pipe wrapping issue
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /tmp/docker.gpg
sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg /tmp/docker.gpg

# Repo file — written via Claude to /tmp then copied
sudo cp /tmp/docker.list /etc/apt/sources.list.d/docker.list
# Contents: deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu noble stable

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io
sudo usermod -aG docker $USER
```

**kubectl:**
```bash
# Used hardcoded v1.32.0 — stable.txt lookup returned XML error page
curl -LO https://dl.k8s.io/release/v1.32.0/bin/linux/amd64/kubectl
chmod +x kubectl && sudo mv kubectl /usr/local/bin/
```

**eksctl:**
```bash
curl -sLO https://github.com/eksctl-io/eksctl/releases/latest/download/eksctl_linux_amd64.tar.gz
tar -xzf eksctl_linux_amd64.tar.gz && sudo mv eksctl /usr/local/bin/
```

---

## Phase 2 — AWS Credentials + S3 + ECR (TOMORROW)

### Step 2.1 — AWS credentials

Need an IAM user with programmatic access. In AWS Console:
> IAM → Users → your user → Security credentials → Create access key → CLI

Then:
```bash
aws configure
# AWS Access Key ID: AKIA...
# AWS Secret Access Key: ...
# Default region: us-east-1
# Default output format: json
```

Verify:
```bash
aws sts get-caller-identity
```

### Step 2.2 — S3 bucket

Create bucket and upload training data:
```bash
aws s3 mb s3://dfine-visdrone --region us-east-1

# Upload dataset (4GB), weights (40MB), checkpoints (583MB)
aws s3 sync ~/projects/DFine/D-FINE/dataset/ s3://dfine-visdrone/dataset/
aws s3 sync ~/projects/DFine/D-FINE/weight/   s3://dfine-visdrone/weight/
aws s3 sync ~/projects/DFine/D-FINE/output/   s3://dfine-visdrone/output/
```

### Step 2.3 — Dockerfile

The existing `Dockerfile` pulls from a Chinese Aliyun registry — not suitable for AWS.
Need to rewrite it based on `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` with:
- Python 3.12
- PyTorch 2.5.1+cu124
- All requirements.txt deps
- W&B, tensorboard
- Training entrypoint that syncs from S3 on start and uploads checkpoints on finish

### Step 2.4 — ECR

```bash
aws ecr create-repository --repository-name dfine-training --region us-east-1
# Then: docker build, docker tag, docker push
```

### Step 2.5 — EKS cluster

```bash
eksctl create cluster \
  --name dfine-training \
  --region us-east-1 \
  --nodegroup-name gpu-nodes \
  --node-type g4dn.xlarge \
  --nodes 1 \
  --spot
```

> Note: cluster creation takes ~15-20 minutes

### Step 2.6 — NVIDIA device plugin

```bash
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.0/deployments/static/nvidia-device-plugin.yml
```

### Step 2.7 — K8s training Job

Write `training-job.yaml` with:
- GPU request: `nvidia.com/gpu: 1`
- S3 sync at container startup (download dataset + checkpoint)
- W&B API key as a K8s secret
- Checkpoint upload to S3 on finish

---

## Cost Reminder

| Instance | GPU | VRAM | Spot price |
|----------|-----|------|-----------|
| g4dn.xlarge | T4 | 16GB | ~$0.16/hr |

72 epochs × ~15 min/epoch ≈ 18 hrs → **~$3 total on spot**

---

## Current Status

| Task | Status |
|------|--------|
| Install aws-cli, docker, kubectl, eksctl | ✅ COMPLETE |
| AWS credentials (`aws configure`) | ✅ COMPLETE |
| Create S3 bucket + upload data | ✅ COMPLETE |
| Write Dockerfile for AWS | ✅ COMPLETE |
| ECR repo + push image | ✅ COMPLETE |
| EKS cluster creation | ❌ ABANDONED — see Phase 3 below |
| NVIDIA device plugin | ⏳ N/A |
| K8s training Job YAML | ✅ WRITTEN — `k8s/training-job.yaml` |
| EC2 spot training | ⏳ BLOCKED — waiting for GPU quota approval |
| GPU quota increase request | ✅ SUBMITTED — PENDING AWS approval |

---

## Phase 2 — What Actually Happened (2026-03-25)

### AWS Credentials
- Ran `aws configure` — region accidentally set to `"Israel"` (text) instead of `il-central-1`
- Fixed with `aws configure set region il-central-1`
- Output format also corrupted — fixed with `aws configure set output json`
- Verified with `aws sts get-caller-identity`

### IAM Permissions
- Initial `s3:CreateBucket` call failed — IAM user `danziv` had no policies attached
- Fixed by attaching `AdministratorAccess` policy via AWS Console (IAM → Users → danziv → Add permissions)

### S3 Bucket
- Created `dfine-visdrone` in `il-central-1`:
  ```bash
  aws s3api create-bucket --bucket dfine-visdrone --region il-central-1 \
    --create-bucket-configuration LocationConstraint=il-central-1
  ```
  > Note: `il-central-1` requires explicit `LocationConstraint` — unlike `us-east-1` which is the default and does NOT accept this flag
- Uploaded 4.5 GiB (19,080 files): dataset/, weight/, output/

### Docker Build
- Existing `Dockerfile` uses Aliyun (Chinese) registry — not reachable from AWS
- Wrote `Dockerfile.aws` based on `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`
- Issues encountered during build:
  - `python3.12` not in Ubuntu 22.04 default apt repos → switched to `python3` (3.10)
  - `awscli` not in apt → installed via official binary zip from `awscli.amazonaws.com`
  - Docker socket permission denied (user not in `docker` group yet) → fixed with `newgrp docker` + `sg docker -c "..."` workaround since Claude Code process runs with old groups
- Final image: ~10GB (CUDA base + PyTorch 2.5.1)

### Docker group issue — root cause
`sudo usermod -aG docker $USER` adds the user to the docker group, but the change only takes
effect in **new login sessions**. Running `newgrp docker` updates the current terminal, but
Claude Code's subprocess still has the old group membership. Workaround: prefix all docker
commands with `sg docker -c "..."` which executes with the docker group active.

### ECR
- Created repos in both `il-central-1` and `us-east-1`:
  ```bash
  aws ecr create-repository --repository-name dfine-training --region il-central-1
  aws ecr create-repository --repository-name dfine-training --region us-east-1
  ```
- Image pushed to both. Login command:
  ```bash
  aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin \
    366447948783.dkr.ecr.us-east-1.amazonaws.com
  ```

### Training code improvements made before push
1. **Cosine LR** — base config had `MultiStepLR` with milestone=500, which never fires in 72
   epochs (LR stays flat forever). Replaced in `dfine_hgnetv2_s_visdrone.yml` with
   `CosineAnnealingLR` (T_max=72, eta_min=1e-6).
2. **W&B logging** — previously only logged total loss + lr_pg0. Now logs all individual loss
   components (loss_fgl, loss_ddf, loss_vfl, etc.) and all LR param groups per epoch.
3. **W&B daemon** — removed Windows `WANDB_START_METHOD=thread` workaround from
   `det_solver.py` (we are on Linux now, default fork mode works correctly).
4. **W&B enabled** — flipped `use_wandb: True` in visdrone config.
5. **W&B tested** — confirmed working in WSL2 with a test run to `dfine-visdrone` project.

---

## Phase 3 — EKS Abandoned, Switching to EC2 Spot (2026-03-25)

### Why EKS was abandoned

**Problem 1 — `g4dn.xlarge` not available in `il-central-1`**
All three AZs in il-central-1 (a/b/c) reject `g4dn.xlarge`. GPU instances are not offered in
the Israel region. Had to switch the compute region to `us-east-1`.

**Problem 2 — Spot capacity timeout**
EKS cluster creation with `--managed --spot` timed out after ~20 min waiting on CloudFormation
(`eksctl-dfine-training-nodegroup-gpu-nodes`). Despite active spot prices ($0.207-0.214/hr in
us-east-1), the managed node group failed to provision.

**Problem 3 — Cost**
EKS charges $0.10/hr for the control plane regardless of whether jobs are running.
For an 18-hour training run:
- EKS control plane: $0.10 × 18 = $1.80
- g4dn.xlarge spot: $0.207 × 18 = $3.73
- Total: **~$5.53 — over the $5 budget**

### EKS vs EC2 — conceptual difference

**EC2** = a virtual machine. You rent it, it runs Docker, it trains, it shuts down.
**EKS** = a cluster manager that sits on top of EC2. Useful for many jobs across many machines.
For a single training run, EKS is overkill ("hiring a manager to hire one worker to hammer one nail").

The K8s training Job YAML (`k8s/training-job.yaml`) is fully written and documented.
The EKS path is a valid resume talking point — we just don't execute it due to cost/simplicity.

### EC2 Spot Plan
- Instance: `g4dn.xlarge` (T4 16GB VRAM) in `us-east-1`
- Spot price: ~$0.207/hr
- Estimated runtime: ~18 hrs (72 epochs × ~15 min/epoch)
- Estimated cost: ~$3.73 + EBS + S3 transfer ≈ **~$3.90 total ✅**
- S3 bucket stays in `il-central-1` — cross-region access works fine (small egress cost ~$0.09)
- Startup: user-data script runs docker, syncs S3, trains, uploads checkpoints, shuts down

---

## Phase 4 — EC2 Spot Setup (2026-03-25)

### Why EC2 instead of EKS (cost breakdown)

| Option | Control plane | GPU instance | Total (18hr) |
|--------|--------------|--------------|-------------|
| EKS + spot | $0.10/hr × 18 = $1.80 | $0.207/hr × 18 = $3.73 | **$5.53 ❌** |
| EC2 spot only | $0 | $0.207/hr × 18 = $3.73 | **~$3.90 ✅** |

For a single training job, EKS is overkill. EC2 spot is simpler and cheaper.

### IAM role for EC2 instance

The EC2 instance needs permissions to pull from ECR and read/write S3.
Created a role and attached to an instance profile:

```bash
# Create role that EC2 can assume
aws iam create-role \
  --role-name dfine-ec2-role \
  --assume-role-policy-document '{
    "Version":"2012-10-17",
    "Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]
  }'

# Attach S3 + ECR read permissions
aws iam attach-role-policy \
  --role-name dfine-ec2-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

aws iam attach-role-policy \
  --role-name dfine-ec2-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly

# Wrap role in an instance profile (what you actually attach to EC2)
aws iam create-instance-profile \
  --instance-profile-name dfine-ec2-profile

aws iam add-role-to-instance-profile \
  --instance-profile-name dfine-ec2-profile \
  --role-name dfine-ec2-role
```

> **Why an instance profile?** EC2 instances can't use IAM roles directly — roles must be
> wrapped in an "instance profile" first. It's just an IAM container for one role.
> When the instance starts, its metadata endpoint serves temporary credentials from this role.
> The AWS CLI inside the container automatically picks these up — no access keys needed.

### Deep Learning AMI

Using the official AWS Deep Learning Base AMI (Ubuntu 22.04) which has:
- NVIDIA drivers pre-installed
- Docker pre-installed with nvidia-container-toolkit
- No need to install anything manually

```bash
# Find latest AMI ID
aws ec2 describe-images \
  --region us-east-1 \
  --owners amazon \
  --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
  --query 'sort_by(Images, &CreationDate)[-1].[ImageId,Name]' \
  --output table
# Result: ami-0f1dd29282e265995 (2026-03-20)
```

### Instance type selection

```bash
# Check spot prices across AZs
aws ec2 describe-spot-price-history \
  --region us-east-1 \
  --instance-types g4dn.xlarge g4dn.2xlarge g5.xlarge \
  --product-descriptions "Linux/UNIX" \
  --query 'SpotPriceHistory[*].[InstanceType,AvailabilityZone,SpotPrice]' \
  --max-items 15 --output table
```

Cheapest: `g4dn.xlarge` in `us-east-1f` at **$0.207/hr**
- GPU: NVIDIA T4, 16GB VRAM (vs our local RTX 4060 8GB — can use batch_size=16)
- vCPUs: 4, RAM: 16GB

Cheapest subnet in us-east-1f:
```bash
aws ec2 describe-subnets \
  --region us-east-1 \
  --filters "Name=availabilityZone,Values=us-east-1f" \
  --query 'Subnets[0].SubnetId' --output text
# Result: subnet-09ea28a8d0d0979ee
```

### User-data script (`k8s/ec2-userdata.sh`)

This script runs automatically when the EC2 instance boots. It:
1. Logs into ECR
2. Pulls the Docker training image
3. Runs training with `docker run` (mounts /dev/shm for DataLoader workers)
4. Shuts down the instance when training completes (so billing stops automatically)

All training output goes to `/var/log/dfine-training.log` on the instance.

### GPU quota issue (BLOCKER)

New AWS accounts have a default quota of **0 vCPUs** for G and VT instances
(the entire GPU family including g4dn). Attempting to launch `g4dn.xlarge` returns:

```
InvalidParameterCombination: The specified instance type is not eligible for Free Tier.
```

This misleading error actually means "you have no quota for this instance family."

**Check quota:**
```bash
aws service-quotas get-service-quota \
  --region us-east-1 \
  --service-code ec2 \
  --quota-code L-DB2E81BA \
  --query 'Quota.[QuotaName,Value]' --output table
# Shows: Running On-Demand G and VT instances | 0.0
```

**Request increase (submitted 2026-03-25):**
```bash
aws service-quotas request-service-quota-increase \
  --region us-east-1 \
  --service-code ec2 \
  --quota-code L-DB2E81BA \
  --desired-value 4
# Request ID: 54a2326c9eae4711abdceb08e3efcfcb1iYR4P73
# Status: PENDING
```

> g4dn.xlarge uses 4 vCPUs, so requesting 4 is the minimum needed for one instance.
> AWS typically approves within a few hours to 1 business day.

### Launch command (run once quota is approved)

```bash
aws ec2 run-instances \
  --region us-east-1 \
  --image-id ami-0f1dd29282e265995 \
  --instance-type g4dn.xlarge \
  --subnet-id subnet-09ea28a8d0d0979ee \
  --iam-instance-profile Name=dfine-ec2-profile \
  --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time"}}' \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --user-data file://k8s/ec2-userdata.sh \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=dfine-training}]' \
  --query 'Instances[0].[InstanceId,State.Name]' \
  --output table
```

### Monitor training progress

```bash
# Check instance state
aws ec2 describe-instances \
  --region us-east-1 \
  --filters "Name=tag:Name,Values=dfine-training" \
  --query 'Reservations[0].Instances[0].[InstanceId,State.Name,PublicIpAddress]' \
  --output table

# SSH into instance (if needed)
ssh -i your-key.pem ubuntu@<PublicIpAddress>
tail -f /var/log/dfine-training.log

# Watch W&B for live metrics
# https://wandb.ai/danziv/dfine-visdrone
```

### After training completes

Instance shuts itself down automatically. Checkpoints are on S3:
```bash
aws s3 ls s3://dfine-visdrone/output/dfine_hgnetv2_s_visdrone/ --region il-central-1
```

# Step 5 — WSL2 + AWS + Kubernetes Migration

## Motivation

- Local training on RTX 4060 Laptop (8GB VRAM) blocks the machine and hits OOM at batch_size > 4
- W&B service process fails to spawn on Windows (known bug) — WSL2 fixes this natively
- Moving to AWS + Kubernetes adds MLOps skills to resume:
  Docker · ECR · S3 · EKS · kubectl · K8s Jobs · GPU scheduling · IAM

---

## Phase 1 — WSL2 Setup (Local, Windows 11)

### Why WSL2 instead of native Windows

| Issue | Windows | WSL2 |
|-------|---------|------|
| W&B service process | Fails to spawn daemon | Works natively (Linux) |
| CUDA support | Native | Pass-through via Windows NVIDIA driver |
| I/O performance | Baseline | 10-20x faster on native WSL2 filesystem |
| Shell / tooling | bash via Git Bash | Native bash, full Linux userland |
| Docker | Docker Desktop required | Native Docker daemon |

### Prerequisites

- Windows 11 (already installed)
- NVIDIA driver 555.97 already installed on Windows — **no separate Linux GPU driver needed inside WSL2**
  - The Windows driver exposes CUDA to WSL2 via `/usr/lib/wsl/lib/`

### 1.1 Install WSL2 + Ubuntu

Open **PowerShell as Administrator**:

```powershell
wsl --install
```

- Installs WSL2 kernel + Ubuntu (latest LTS) by default
- **Reboot required**
- On first Ubuntu launch: set a UNIX username + password

Verify after reboot:
```bash
wsl --list --verbose
# Should show Ubuntu running with VERSION 2
```

### 1.2 Verify GPU access inside WSL2

```bash
nvidia-smi
# Should show RTX 4060 Laptop, CUDA 12.x
```

If `nvidia-smi` is not found:
```bash
# CUDA libs are at /usr/lib/wsl/lib/ — add to path
echo 'export PATH=/usr/lib/wsl/lib:$PATH' >> ~/.bashrc
source ~/.bashrc
nvidia-smi
```

### 1.3 Install system dependencies

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.12 python3.12-venv python3.12-dev \
    git curl wget build-essential libgl1 libglib2.0-0
```

### 1.4 Clone the project into WSL2 filesystem

> **Important:** Do NOT work from `/mnt/c/projects/...` — that crosses the WSL/Windows
> filesystem boundary and is 10-20x slower. Clone fresh into the WSL2 native filesystem.

```bash
mkdir -p ~/projects
cd ~/projects
git clone <repo-url> DFine
cd DFine/D-FINE
```

### 1.5 Create venv + install dependencies

```bash
python3.12 -m venv venv
source venv/bin/activate

# PyTorch with CUDA 12.4 (matches Windows driver)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# Project dependencies
pip install -r requirements.txt

# W&B (should work natively on Linux)
pip install wandb tensorboard
```

### 1.6 Verify CUDA + W&B

```bash
source venv/bin/activate

# Check PyTorch sees the GPU
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# Expected: True, NVIDIA GeForce RTX 4060 Laptop GPU

# Check W&B login
wandb login
# Follow URL, paste API key — should work without thread workaround
```

### 1.7 Copy dataset + weights from Windows

The dataset (~several GB) lives on the Windows drive. Two options:

**Option A — Copy to WSL2 filesystem (recommended for training performance):**
```bash
cp -r /mnt/c/projects/claude_code/DFine/D-FINE/dataset ~/projects/DFine/D-FINE/
cp -r /mnt/c/projects/claude_code/DFine/D-FINE/weight ~/projects/DFine/D-FINE/
cp -r /mnt/c/projects/claude_code/DFine/D-FINE/output ~/projects/DFine/D-FINE/
```

**Option B — Symlink (saves disk space, slower I/O):**
```bash
ln -s /mnt/c/projects/claude_code/DFine/D-FINE/dataset ~/projects/DFine/D-FINE/dataset
```

### 1.8 Test training run in WSL2

```bash
cd ~/projects/DFine/D-FINE
source venv/bin/activate

python train.py \
  -c configs/dfine/dfine_hgnetv2_s_visdrone.yml \
  --device cuda:0 \
  --resume output/dfine_hgnetv2_s_visdrone/best_stg1.pth \
  -u epochs=1 train_dataloader.total_batch_size=4
```

### 1.9 Enable W&B (now that it works on Linux)

In `configs/dfine/dfine_hgnetv2_s_visdrone.yml`, change:
```yaml
use_wandb: False
```
to:
```yaml
use_wandb: True
```

---

## Phase 2 — AWS + Kubernetes

> Status: Planned. Start after WSL2 is verified working.

### Architecture Overview

```
Local (WSL2)                 AWS
─────────────────            ──────────────────────────────────────
docker build                 ECR (image registry)
  → push image       ──────► ECR repo: dfine-training
                             │
                             S3 (data + checkpoints)
                             │  s3://dfine-visdrone/dataset/
                             │  s3://dfine-visdrone/checkpoints/
                             │
                             EKS cluster
                             │  node group: g4dn.xlarge (T4 GPU, 16GB VRAM)
                             │
                             K8s Job: dfine-train
                             │  resources: nvidia.com/gpu: 1
                             │  mounts S3 via s3fs or downloads at startup
                             └─► TensorBoard via port-forward or S3 sync
```

### AWS Services Used

| Service | Purpose | Resume Skill |
|---------|---------|--------------|
| ECR | Docker image registry | Container registries |
| S3 | Dataset + checkpoint storage | Cloud object storage |
| EKS | Managed Kubernetes cluster | Kubernetes, EKS |
| IAM | Roles for EKS nodes to access S3/ECR | IAM, RBAC |
| CloudWatch | Logs from training pods | Observability |

### Key Steps (detailed docs to be written per step)

1. **Dockerize** — build training image from existing `Dockerfile`
2. **ECR** — create repo, push image
3. **S3** — upload dataset + weights, configure checkpoint sync
4. **EKS** — create cluster with GPU node group (`g4dn.xlarge`)
5. **NVIDIA device plugin** — install so K8s can schedule GPU workloads
6. **K8s Job** — write `training-job.yaml` with GPU request + S3 env vars
7. **TensorBoard** — either `kubectl port-forward` or sync event files to S3 and serve

### Cost Estimate

| Instance | GPU | VRAM | On-Demand | Spot |
|----------|-----|------|-----------|------|
| g4dn.xlarge | T4 | 16GB | ~$0.53/hr | ~$0.16/hr |
| g4dn.2xlarge | T4 | 16GB | ~$0.75/hr | ~$0.23/hr |
| p3.2xlarge | V100 | 16GB | ~$3.06/hr | ~$0.92/hr |

**Recommendation:** `g4dn.xlarge` spot instance. 72 epochs × ~15 min/epoch = ~18 hrs.
Estimated cost: 18 × $0.16 ≈ **~$3 for full training run** on spot.

---

## What Actually Happened (2026-03-24)

The WSL2 migration was completed manually. Here is the actual sequence of events:

### Trigger
- W&B daemon failed to spawn on Windows (known Windows-specific bug)
- General Windows environment instability was causing friction
- Decision made to migrate fully to WSL2

### Steps Taken

1. **WSL2 installed manually** — opened PowerShell as Administrator, ran `wsl --install`
2. **Ubuntu installed via Microsoft Store** — not the default WSL distro, explicitly chosen from the Store
3. **Project copied from Windows to WSL2 native filesystem** — Claude ran the copy from `/mnt/c/projects/claude_code/DFine/` into `~/projects/DFine/` (native WSL2 filesystem for full I/O performance)
4. **venv re-created inside WSL2** — fresh `python3.12 -m venv venv`, reinstalled PyTorch 2.5.1+cu124 and all dependencies
5. **Windows project directory deleted** — Claude deleted the Windows-side copy at `/mnt/c/projects/claude_code/DFine/` to free space and avoid confusion between two copies

> **Note:** Steps 3–5 were done by Claude in the previous session. The Windows deletion happened before this log entry was written — that's why this is being reconstructed from memory rather than live observation.

### Outcome
- Training now runs from `~/projects/DFine/D-FINE/` (WSL2 native filesystem)
- W&B issue resolved — Linux daemon spawns correctly
- Single canonical copy of the project exists (WSL2 only)
- Windows copy is gone and cannot be recovered

---

## Current Status

| Phase | Step | Status |
|-------|------|--------|
| WSL2 | Install WSL2 + Ubuntu (via Store) | ✅ COMPLETE |
| WSL2 | Verify GPU (nvidia-smi) | ✅ COMPLETE |
| WSL2 | Copy project + recreate venv + deps | ✅ COMPLETE |
| WSL2 | Delete Windows copy | ✅ COMPLETE |
| WSL2 | Verify W&B works | ⏳ Pending — not yet tested post-migration |
| WSL2 | Test training run | ⏳ Pending |
| AWS | Dockerize | ⏳ Pending |
| AWS | ECR + S3 setup | ⏳ Pending |
| AWS | EKS cluster + GPU node group | ⏳ Pending |
| AWS | K8s training Job | ⏳ Pending |
| AWS | TensorBoard on cloud | ⏳ Pending |

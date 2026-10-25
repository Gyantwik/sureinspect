# AWS Production Deployment Evidence — SureInspect (OpenCV AI Competition 2026)

**Target System:** SureInspect Industrial Visual Inspection  
**Specification Reference:** SureInspect v8.1 Master Architecture  
**Document Status:** IN PROGRESS (Authoritative Evidence Log)  
**Last Updated (UTC):** 2026-10-25T17:55:00Z  
**Git HEAD Commit:** `fc9006e4c7494862ffab2b93edffa2b4312df69a`  

---

## 1. Master Deployment Status Checklist

| ID | Verification Item | Status | Evidence / Notes |
| :--- | :--- | :--- | :--- |
| **T1.1** | Repository & Architecture Audit (`README.md`, `docs/aws_section.md`) | **PASS** | Completed. Verified routes, proxy architecture, resource limits. |
| **T1.2** | Deployment Files Inspection (`Dockerfile`, `docker-compose.yml`, `Caddyfile`) | **PASS** | Verified multi-stage Docker build, Caddy `:80` default binding, private port 8000. |
| **T1.3** | Backend Entrypoint & Health Route Audit (`backend/main.py`, `app_factory.py`) | **PASS** | Confirmed `/api/health`, `/api/inspect`, 25 MP image ceiling, bounded queue. |
| **T1.4** | Model Snapshot Manifest & Hash Verification | **BLOCKED** | Model snapshot `pcb1-wide_resnet50_2-256-edc526f4da11-agent-v1` is git-ignored (`.gitignore`). Awaiting S3 artifact transfer to host. |
| **T1.5** | Frontend Same-Origin API Proxy Verification | **PASS** | Frontend routes mapped to relative `/api/*`, proxied to backend via Caddy. |
| **T2.1** | Safe Provisioning Authorization & Cost Gate | **PASS** | User explicitly authorized Option A (existing instance `i-0025a98d800b7485e`, $0.00 incremental cost). |
| **T2.2** | Host Infrastructure (EC2 / Security Group / Elastic IP) | **PASS** | Authorized instance confirmed: `m7i-flex.large`, 30 GiB gp3, IP `3.27.171.151`, Sydney. |
| **T3.1** | Host Provisioning & Git Connection | **IN PROGRESS** | Awaiting command execution on host via EC2 Instance Connect. |
| **T4.1** | Container Build & Service Startup | **NOT TESTED** | Awaiting Task 3 completion. |
| **T4.2** | Port 8000 Isolation (Strictly Private) | **PASS (Config)** | Verified in compose file; pending live host network scan. |
| **T5.1** | Live OpenCV 5 C++ DNN Inference Verification | **NOT TESTED** | Requires active container runtime with model snapshot. |
| **T5.2** | 3-Scenario Evaluation (Normal, Defect, Optical Rejection) | **NOT TESTED** | Requires active container runtime with model snapshot. |
| **T6.1** | Browser-Trusted HTTPS / Domain Verification | **NOT TESTED** | Pending domain binding and ACME TLS issuance. |
| **T7.1** | Recovery After Restart & Survival Check | **NOT TESTED** | Pending live test (`docker compose restart`). |

---

## 2. Pre-Deployment Audit Details (Task 1)

### 2.1 Git State
- **Branch:** `main`
- **Commit:** `f8f99124ad825294c4e187b14b76421fcf1ed806`
- **Cleanliness:** Working tree clean, deployment configurations committed.

### 2.2 Security & Isolation Checks
- **Port 8000 Exposure:** In `deploy/docker-compose.yml`, backend service uses `expose: ["8000"]`, which restricts communication strictly to the internal Docker bridge network `internal_net`. No host port mapping (`8000:8000`) is configured.
- **Edge 10 MiB Body Cap:** Enforced in `deploy/Caddyfile` using `request_body { max_size 10MB }` at the TLS reverse-proxy layer.
- **Decompression Bomb Protection:** Pre-initialized before imports in `backend/main.py`: `OPENCV_IO_MAX_IMAGE_PIXELS=25000000`.
- **CPU Concurrency Caps:** Threading strictly bounded in `backend/main.py` and `deploy/docker-compose.yml` (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`).

### 2.3 Model Snapshot Artifact Requirement
- Target Snapshot ID: `artifacts/snapshots/pcb1-wide_resnet50_2-256-edc526f4da11-agent-v1`
- Required files in snapshot:
  - `model.onnx` (WideResNet-50-2 backbone)
  - `bank.npz` (PatchCore coreset memory bank)
  - `calibration.json` (Conformal statistical thresholds)
  - `quality_gate.json` (Fitted gate thresholds)
  - `splits.json` (Dataset splits and hashes)
  - `manifest.json` (Policy and runtime manifest)
  - `SHA256SUMS` (LF-delimited checksums)
- **Status:** Because large model artifacts are excluded from version control (`.gitignore`), they must be populated into `/home/ubuntu/sureinspect/artifacts/snapshots/` on the deployment host (via Amazon S3 `aws s3 cp s3://sureinspect-production-artifacts/...` or secure transfer).

---

## 3. Proposed AWS Provisioning & Cost Plan (Task 2)

Per Mandatory Deployment Safeguard A, explicit authorization is required before provisioning or modifying paid cloud resources.

### Proposal A (Existing Provisioned Instance — Zero Incremental Setup Cost)
- **Instance ID:** `i-0025a98d800b7485e` (Name: `SureInspect`)
- **Region:** `ap-southeast-2` (Sydney, `ap-southeast-2a`)
- **Instance Type:** `m7i-flex.large` (2 vCPUs, ~7.6 GiB RAM)
- **Storage:** 30 GiB gp3 encrypted EBS
- **Public IP:** `3.27.171.151`
- **Docker Engine:** `29.1.3`, Compose `2.40.3` already installed.
- **Cost:** Uses user's existing AWS promotional credits.

### Proposal B (Fresh Instance — Pinned Spec)
- **Service:** AWS EC2
- **Region:** `us-east-1` (N. Virginia)
- **Instance Type:** `t3.medium` (2 vCPUs, 4 GiB RAM, x86_64)
- **Storage:** 30 GiB gp3 Encrypted EBS Volume
- **Networking:** Attached Elastic IP, Security Group (22 restricted to admin IP, 80/443 public, 8000 deny)
- **Estimated Cost:** ~$0.0416/hr (~$1.00/day; ~$15.00 for 15-day competition window)
- **Budget Alert:** CloudWatch Billing Budget alarm set at $5.00 USD.

---

## 4. Teardown & Rollback Plan

### Teardown Script
```bash
# Decommissioning commands for AWS CLI:
aws ec2 terminate-instances --instance-ids <INSTANCE_ID>
aws ec2 release-address --allocation-id <EIP_ALLOC_ID>
aws ec2 delete-security-group --group-id <SG_ID>
aws s3 rm s3://sureinspect-production-artifacts --recursive
aws s3api delete-bucket --bucket sureinspect-production-artifacts
```

---

## 5. Host Deployment Runbook (Task 3 & Task 4)

Target Instance: `i-0025a98d800b7485e` (`SureInspect`, `ap-southeast-2`, `3.27.171.151`)

### 5.1 Git Synchronization & Setup
```bash
cd /home/ubuntu/sureinspect

# Fetch and checkout deployment branch
git fetch origin
git checkout deploy/aws-ec2
git pull origin deploy/aws-ec2

# Configure environment
cat << 'EOF' > deploy/.env
DOMAIN=:80
AWS_DEFAULT_REGION=ap-southeast-2
ARTIFACTS_BUCKET=sureinspect-production-artifacts
USE_MOCK=false
OPENCV_IO_MAX_IMAGE_PIXELS=25000000
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
EOF
```

### 5.2 Model Snapshot Verification (Task 1.4 / Safeguard C)
```bash
# Check if snapshot directory exists
ls -la /home/ubuntu/sureinspect/artifacts/snapshots/pcb1-wide_resnet50_2-256-edc526f4da11-agent-v1/ || true

# If using S3 artifact sync:
# aws s3 sync s3://sureinspect-production-artifacts/artifacts/ /home/ubuntu/sureinspect/artifacts/
```

### 5.3 Build & Container Startup (Task 4)
```bash
cd /home/ubuntu/sureinspect/deploy
sudo docker compose up -d --build
sudo docker compose ps
```

### 5.4 Live Verification & Port 8000 Security Audit (Task 4.2 & Task 5)
```bash
# Verify Health Endpoint locally
curl -s http://localhost:8000/api/health | jq .

# Verify Reverse Proxy through Caddy (port 80)
curl -s http://localhost/api/health | jq .

# Verify Port 8000 Isolation (must reject external connections)
# curl from an external machine to http://3.27.171.151:8000 must timeout or be refused.
```


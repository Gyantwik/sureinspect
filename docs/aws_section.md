# SureInspect: AWS Deployment, Architecture, Security & Cost Analysis

> **Author:** Person 2 (@Gyantwik) — Backend, Integration, AWS & Evaluation Lead  
> **Project:** SureInspect (OpenCV AI Competition 2026)  
> **Architecture Target:** AWS EC2 Graviton (ARM64) + Private S3 + Caddy TLS Reverse Proxy  

---

## 1. System Architecture Overview

```
                      Internet (HTTPS Port 443)
                                  │
                                  ▼
               ┌─────────────────────────────────────┐
               │         Caddy Reverse Proxy         │
               │   - Automatic Let's Encrypt TLS     │
               │   - 10 MiB Body Limit Enforcement   │
               │   - Real-IP Header Forwarding       │
               └──────────────────┬──────────────────┘
                                  │ localhost:8000
                                  ▼
               ┌─────────────────────────────────────┐
               │        FastAPI Docker Container     │
               │   - Single Uvicorn Worker           │
               │   - Bounded Async Queue (1+4)       │
               │   - 25 MP Decompression Bomb Guard  │
               │   - 200 MiB / 100 Item LRU Cache    │
               │   - OpenCV 5 DNN Perception Engine  │
               └──────────────────┬──────────────────┘
                                  │
                                  ▼
               ┌─────────────────────────────────────┐
               │          Private AWS S3             │
               │   - /artifacts (Read-Only)          │
               │   - /traces (Write-Only Audit)      │
               │   - /feedback (Write-Only Audit)    │
               └─────────────────────────────────────┘
```

The SureInspect runtime is deployed on AWS utilizing energy-efficient **AWS Graviton (ARM64)** compute, paired with a local Caddy reverse proxy and an isolated private Amazon S3 bucket for immutable artifact loading and audit retention.

---

## 2. Compute Selection: AWS Graviton ARM64

### 2.1 Instance Type & Sizing
- **Selected Tier:** `t4g.small` (2 vCPU, 2 GiB memory) or `c7g.medium` (1 vCPU, 2 GiB memory).
- **Architecture:** `aarch64` / `arm64` (AWS Graviton2 / Graviton3 Neoverse cores).
- **Cost Efficiency:** Graviton instances provide up to 40% better price-performance over comparable x86_64 instances and qualify for the AWS Graviton Free Trial (750 hours/month of `t4g.small` through 2026).

### 2.2 Memory & CPU Thread Concurrency Guard
- **Single Worker Process:** Bound to 1 uvicorn worker process to prevent memory fragmentation and thread contention.
- **Thread Capping:** OpenMP, OpenBLAS, and MKL CPU thread pools are strictly capped to 1 thread via environment variables:
  ```bash
  OMP_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1
  MKL_NUM_THREADS=1
  ```
- **Peak RSS Budget:** The total memory footprint (including Linux kernel, Docker engine, Caddy, Python runtime, OpenCV 5 DNN model weights, and the 200 MiB LRU cache) stays under **1.1 GiB**, safely within the 2 GiB physical RAM envelope.

---

## 3. Storage Architecture: Private Amazon S3

All persistent data is decoupled into an isolated private S3 bucket (`sureinspect-production-artifacts`):

| Prefix | Access Pattern | Lifecycle Policy | Purpose |
| :--- | :--- | :--- | :--- |
| `artifacts/` | **Read-Only** from EC2 | Permanent (Immutable) | Frozen ONNX weights, coreset memory bank (`.npy`), and calibrated thresholds. |
| `traces/` | **Write-Only** from EC2 | 14-day automatic expiration | Serialized inspection traces and decision logs. |
| `feedback/` | **Write-Only** from EC2 | 14-day automatic expiration | Operator human labels (`NORMAL`/`DEFECT`) logged for audit. |
| `reports/` | **Read/Write** (Admin) | Permanent | Published immutable evaluation bundles (`reports/<bundle-id>/`). |

---

## 4. Least-Privilege IAM Security Model

### 4.1 Production Server IAM Role
The EC2 Graviton instance runs under an IAM Instance Profile (`SureInspectRuntimeRole`) that enforces strict separation between read-only production assets and write-only audit logs:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowReadImmutableArtifacts",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::sureinspect-production-artifacts",
        "arn:aws:s3:::sureinspect-production-artifacts/artifacts/*",
        "arn:aws:s3:::sureinspect-production-artifacts/reports/*"
      ]
    },
    {
      "Sid": "AllowWriteAuditOnlyLogs",
      "Effect": "Allow",
      "Action": [
        "s3:PutObject"
      ],
      "Resource": [
        "arn:aws:s3:::sureinspect-production-artifacts/traces/*",
        "arn:aws:s3:::sureinspect-production-artifacts/feedback/*"
      ]
    },
    {
      "Sid": "DenyMutatingProductionBank",
      "Effect": "Deny",
      "Action": [
        "s3:DeleteObject",
        "s3:PutObjectTagging"
      ],
      "Resource": [
        "arn:aws:s3:::sureinspect-production-artifacts/artifacts/*"
      ]
    }
  ]
}
```

> **Security Guarantee:** Even if a remote web execution vulnerability were discovered, the server process lacks IAM permissions to delete, overwrite, or mutate the frozen production memory bank or model weights.

---

## 5. Network Security & Reverse Proxy

### 5.1 Security Group Configuration
| Direction | Port | Protocol | Source | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| Inbound | `80` | TCP | `0.0.0.0/0` | HTTP challenge (ACME) & automatic HTTPS 301 redirect |
| Inbound | `443` | TCP | `0.0.0.0/0` | Secure HTTPS judge and operator traffic |
| Inbound | `22` | TCP | Authorized IP only | Secure administrative SSH |
| Outbound | All | All | `0.0.0.0/0` | S3 API calls, ACME certificate renewal, OS updates |

### 5.2 Edge Request Filtering & Caddy TLS
- **Edge 10 MiB Limit:** Enforced in Caddy via `request_body { max_size 10MB }` before data ever reaches Python memory.
- **Decompression Bomb Guard:** Enforced via `OPENCV_IO_MAX_IMAGE_PIXELS=25000000` inside Python, rejecting header-inflated images (e.g. 20000×20000 PNGs) with HTTP 413.
- **Rate & Queue Admission:** Bounded async semaphore admits at most 1 running inference and 4 queued waiting jobs, rejecting excess bursts with HTTP 503 (`QUEUE_SATURATED`).

---

## 6. Cost Analysis & Budget Monitoring

| Resource | Specification | Monthly Cost (USD) |
| :--- | :--- | :--- |
| **EC2 Compute** | `t4g.small` (750 hours/mo) | **$0.00** (Free Tier eligible) / ~$12.26 without tier |
| **EBS Storage** | 20 GB gp3 general-purpose SSD | **$1.60** / month |
| **S3 Storage** | < 10 GB artifacts and audit traces | **$0.23** / month |
| **Data Transfer** | Outbound images & traces (< 5 GB) | **$0.00** (Within AWS 100 GB/mo free outbound) |
| **Domain & TLS** | Public DNS + Let's Encrypt | **$0.00** |
| **Total Estimated Operating Cost** | | **$1.83 / month** (or < $15/mo without free tier) |

- **Billing Alarm:** Configured in AWS CloudWatch to trigger an email notification at **$10.00 USD**, preventing accidental billing overruns.
- **No Marketplace Upgrades:** Requires zero paid third-party AMI subscriptions or proprietary API tokens.

---

## 7. Teardown & Decommissioning Instructions

To decommission all AWS resources after the judging period:

```bash
# 1. Terminate EC2 Graviton Instance
aws ec2 terminate-instances --instance-ids <INSTANCE_ID>

# 2. Empty and delete S3 Bucket
aws s3 rm s3://sureinspect-production-artifacts --recursive
aws s3api delete-bucket --bucket sureinspect-production-artifacts

# 3. Delete Security Group and IAM Role
aws ec2 delete-security-group --group-id <SG_ID>
aws iam remove-role-from-instance-profile --instance-profile-name SureInspectProfile --role-name SureInspectRuntimeRole
aws iam delete-role --role-name SureInspectRuntimeRole
```

#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# deploy/setup_ec2.sh
# Automated host provisioning script for AWS EC2 Graviton (ARM64) instance
# Configures Docker, Docker Compose plugin, firewall, and runs SureInspect.
# ---------------------------------------------------------------------------
set -euo pipefail

echo "==> [1/5] Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y --no-install-recommends \
    curl \
    git \
    ca-certificates \
    gnupg \
    lsb-release \
    ufw

echo "==> [2/5] Installing Docker Engine (ARM64)..."
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Grant current user docker access
sudo usermod -aG docker "$USER"

echo "==> [3/5] Configuring Host Firewall (UFW)..."
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp   comment 'SSH'
sudo ufw allow 80/tcp   comment 'HTTP for ACME / redirect'
sudo ufw allow 443/tcp  comment 'HTTPS secure traffic'
sudo ufw --force enable

echo "==> [4/5] Verifying S3 Instance Profile Role..."
if command -v aws &>/dev/null; then
    echo "Checking AWS CLI IAM role credentials..."
    aws sts get-caller-identity || echo "Warning: IAM instance role not yet active or AWS CLI not configured."
else
    echo "Note: Install awscli-v2 if manual CLI verification is needed."
fi

echo "==> [5/5] Setup complete! You can now run:"
echo "    cd deploy && docker compose up -d --build"

#!/bin/bash
# Carbon Shadow v2 — Start Device Agent
# Usage: ./start_agent.sh [server_url] [region] [device_type]

SERVER=${1:-http://localhost:5000}
REGION=${2:-ap-south-1}
DEVTYPE=${3:-laptop}

cd "$(dirname "$0")/.."
echo "🔌 Carbon Shadow Device Agent"
echo "   Server: $SERVER"
echo "   Region: $REGION"
echo "   Type:   $DEVTYPE"
echo ""

if ! python -c "import psutil,requests" 2>/dev/null; then
  pip install psutil requests --quiet
fi

python device_agent/agent.py \
  --server "$SERVER" \
  --region "$REGION" \
  --device-type "$DEVTYPE"

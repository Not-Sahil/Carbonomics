#!/bin/bash
# Carbon Shadow v2 — Start Backend
set -e
cd "$(dirname "$0")/.."
echo "🌿 Carbon Shadow v2 — Starting backend..."
echo ""

# Install deps if needed
if ! python -c "import flask" 2>/dev/null; then
  echo "📦 Installing Python dependencies..."
  pip install -r backend/requirements.txt --quiet
fi

echo "✅ Dependencies OK"
echo "🚀 Starting Flask API on http://0.0.0.0:5000"
echo ""
cd backend
python api/app.py

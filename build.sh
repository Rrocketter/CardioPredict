#!/bin/bash
set -e

echo "🐍 Python version:"
python --version

echo "📦 Upgrading pip and build tools..."
pip install --upgrade pip setuptools wheel

echo "📋 Installing requirements..."
cd web_platform
pip install -r requirements.txt

echo "✅ Build completed successfully!"

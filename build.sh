#!/usr/bin/env bash
# Build script for Railway deployment

# Change to web platform directory
cd web_platform

# Install dependencies
pip install -r requirements.txt

echo "Build completed successfully!"

#!/bin/bash

# Render deployment script for TechGyant Insights API
echo "🚀 Starting TechGyant Insights deployment..."

# Upgrade pip and build tools
echo "📦 Upgrading pip and build tools..."
pip install --upgrade pip setuptools wheel

# Install dependencies
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/models
mkdir -p data/processed
mkdir -p logs

# Run the setup script
echo "🔧 Setting up models and data..."
python setup_and_run.py

echo "✅ Setup complete! Starting API server..."

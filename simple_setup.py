#!/usr/bin/env python3
"""
Simple setup script for Render deployment
Creates necessary directories and basic files
"""

import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary directories"""
    directories = [
        "data",
        "data/models",
        "data/processed",
        "data/raw",
        "logs"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        else:
            logger.info(f"Directory exists: {directory}")

def main():
    """Main setup function"""
    logger.info("🚀 Starting TechGyant Insights setup...")
    
    # Create directories
    create_directories()
    
    logger.info("✅ Setup complete!")
    logger.info("🌟 TechGyant Insights API is ready to start!")

if __name__ == "__main__":
    main()

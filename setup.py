#!/usr/bin/env python3
"""
Setup script for ISight Sign Language Detector

Run: python setup.py
"""

import os
import sys


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required.")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✓ Python version: {sys.version.split()[0]}")


def install_dependencies():
    """Install required packages."""
    print("\n📦 Installing dependencies...")
    os.system(f"{sys.executable} -m pip install --upgrade pip")
    os.system(f"{sys.executable} -m pip install -r requirements.txt")
    print("✓ Dependencies installed")


def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    directories = ['training_data', 'logs', 'models']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✓ Created {directory}/")


def main():
    """Main setup routine."""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║         ISight Sign Language Detector Setup             ║
    ║                  Author: Blazehue                       ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    check_python_version()
    install_dependencies()
    create_directories()
    
    print("""
    \n✅ Setup complete!
    
    Next steps:
    1. Collect training data: python data_collector.py
    2. Train models: python model_trainer.py
    3. Run detector: python sign_language_detector.py
    
    For more information, see README.md
    """)


if __name__ == "__main__":
    main()

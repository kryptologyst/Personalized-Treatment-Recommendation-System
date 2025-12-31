#!/usr/bin/env python3
"""
Setup script for Personalized Treatment Recommendation System.

This script helps users set up the environment and verify installation.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("❌ Python 3.10+ is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def check_pip():
    """Check if pip is available."""
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, capture_output=True)
        print("✅ pip is available")
        return True
    except subprocess.CalledProcessError:
        print("❌ pip is not available")
        return False


def install_requirements():
    """Install required packages."""
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    try:
        print("📦 Installing required packages...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ], check=True)
        print("✅ All packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False


def create_directories():
    """Create necessary directories."""
    directories = [
        "data/raw",
        "data/processed", 
        "logs",
        "checkpoints",
        "assets",
        "models"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")


def verify_installation():
    """Verify that the installation works."""
    try:
        # Test basic imports
        sys.path.append(str(Path(__file__).parent / "src"))
        
        from data import TreatmentDataProcessor
        from models import TreatmentRecommenderFactory
        from utils import set_seed
        
        print("✅ Core modules import successfully")
        
        # Test basic functionality
        config = {'max_samples': 5}
        processor = TreatmentDataProcessor(config)
        data = processor.generate_synthetic_data(5)
        
        if len(data) == 5:
            print("✅ Data generation works")
        else:
            print("❌ Data generation failed")
            return False
        
        # Test model creation
        model = TreatmentRecommenderFactory.create_model('tfidf', {'max_features': 100})
        if model is not None:
            print("✅ Model creation works")
        else:
            print("❌ Model creation failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False


def main():
    """Main setup function."""
    print("🏥 Personalized Treatment Recommendation System Setup")
    print("=" * 60)
    print()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check pip
    if not check_pip():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Verify installation
    if not verify_installation():
        print("\n❌ Setup verification failed")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print()
    print("Next steps:")
    print("1. Run the basic demo: python 0470.py")
    print("2. Launch interactive demo: streamlit run demo/app.py")
    print("3. Run tests: python -m pytest tests/")
    print("4. Train models: python scripts/train.py")
    print()
    print("⚠️  Remember: This is for research and educational purposes only!")
    print("   NOT intended for clinical use or medical diagnosis.")


if __name__ == "__main__":
    main()

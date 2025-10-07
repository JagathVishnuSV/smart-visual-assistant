"""
Setup script for Accessibility Vision Assistant
Handles installation and configuration
"""
import os
import sys
import subprocess
import platform


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True


def install_dependencies():
    """Install required packages"""
    print_header("Installing Dependencies")
    
    try:
        # Upgrade pip first
        print("📦 Upgrading pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install requirements
        print("\n📦 Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        print("\n✅ All dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error installing dependencies: {e}")
        return False


def setup_environment():
    """Setup environment variables"""
    print_header("Setting Up Environment")
    
    env_file = ".env"
    env_example = ".env.example"
    
    if os.path.exists(env_file):
        print(f"✅ {env_file} already exists")
        return True
    
    if os.path.exists(env_example):
        try:
            with open(env_example, 'r') as src:
                content = src.read()
            with open(env_file, 'w') as dst:
                dst.write(content)
            print(f"✅ Created {env_file} from {env_example}")
            print("\n⚠️  IMPORTANT: Edit .env and add your GEMINI_API_KEY")
            print("   Get free API key from: https://makersuite.google.com/app/apikey")
            return True
        except Exception as e:
            print(f"❌ Error creating .env: {e}")
            return False
    else:
        print(f"⚠️  {env_example} not found")
        return True


def check_camera():
    """Check if camera is available"""
    print_header("Checking Camera")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Camera detected and accessible")
            cap.release()
            return True
        else:
            print("⚠️  Camera not detected. Some features may not work.")
            print("   Please connect a webcam for full functionality.")
            return False
    except Exception as e:
        print(f"⚠️  Could not check camera: {e}")
        return False


def download_models():
    """Download required models"""
    print_header("Checking Models")
    
    models = {
        "yolov8n.pt": "YOLOv8 Nano model (will auto-download on first run)"
    }
    
    for model, desc in models.items():
        if os.path.exists(model):
            print(f"✅ {model} found")
        else:
            print(f"⚠️  {model} not found - {desc}")
    
    return True


def run_tests():
    """Run basic tests"""
    print_header("Running Tests")
    
    try:
        # Test imports
        print("Testing imports...")
        import cv2
        print("  ✅ OpenCV")
        
        import numpy
        print("  ✅ NumPy")
        
        import gradio
        print("  ✅ Gradio")
        
        import pyttsx3
        print("  ✅ pyttsx3 (TTS)")
        
        try:
            import torch
            print("  ✅ PyTorch")
        except ImportError:
            print("  ⚠️  PyTorch not found (will be installed)")
        
        print("\n✅ All core imports successful!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("   Try running: pip install -r requirements.txt")
        return False


def print_next_steps():
    """Print next steps for user"""
    print_header("Setup Complete!")
    
    print("🎉 The Accessibility Vision Assistant is ready to use!\n")
    print("Next steps:")
    print("1. Edit .env file and add your GEMINI_API_KEY")
    print("   Get free API key: https://makersuite.google.com/app/apikey\n")
    print("2. Run the application:")
    print("   python app.py\n")
    print("3. Choose GUI mode (option 1) for the best experience\n")
    print("For help, see README.md or check the documentation.\n")


def main():
    """Main setup function"""
    print_header("Accessibility Vision Assistant - Setup")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version.split()[0]}")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed. Please check errors above.")
        sys.exit(1)
    
    # Setup environment
    setup_environment()
    
    # Check camera
    check_camera()
    
    # Download models
    download_models()
    
    # Run tests
    run_tests()
    
    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Setup failed with error: {e}")
        sys.exit(1)

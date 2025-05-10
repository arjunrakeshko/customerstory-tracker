import subprocess
import sys
import platform
import os
from pathlib import Path
import logging
from typing import Optional
import json

def check_ollama_installation() -> bool:
    """Check if Ollama is installed"""
    try:
        subprocess.run(['ollama', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_ollama() -> bool:
    """Install Ollama based on the operating system"""
    system = platform.system().lower()
    
    if system == 'darwin':  # macOS
        print("Installing Ollama on macOS...")
        subprocess.run(['curl', '-fsSL', 'https://ollama.com/install.sh', '|', 'sh'], shell=True)
    elif system == 'linux':
        print("Installing Ollama on Linux...")
        subprocess.run(['curl', '-fsSL', 'https://ollama.com/install.sh', '|', 'sh'], shell=True)
    elif system == 'windows':
        print("Please download and install Ollama from https://ollama.com/download")
        return False
    else:
        print(f"Unsupported operating system: {system}")
        return False
    
    return check_ollama_installation()

def pull_model(model_name: str = "mistral") -> bool:
    """Pull the specified model"""
    try:
        print(f"Pulling {model_name} model...")
        subprocess.run(['ollama', 'pull', model_name], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error pulling model: {e}")
        return False

def test_ollama_connection() -> bool:
    """Test the connection to Ollama"""
    try:
        data = {
            "model": "mistral",
            "prompt": "Hello, are you working?",
            "stream": False
        }
        json_str = json.dumps(data)
        cmd = f"curl -s http://localhost:11435/api/generate -d '{json_str}'"
        response = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return response.returncode == 0
    except Exception as e:
        print(f"Error testing Ollama connection: {e}")
        return False

def setup_environment():
    """Set up environment variables and create necessary directories"""
    # Create .env file if it doesn't exist
    env_path = Path('.env')
    if not env_path.exists():
        with open(env_path, 'w') as f:
            f.write("""# Ollama Configuration
OLLAMA_HOST=http://localhost:11435
OLLAMA_MODEL=mistral

# Logging Configuration
LOG_LEVEL=INFO
LOG_DIR=logs
""")
        print("Created .env file with default configuration")

def main():
    print("Setting up local LLM service...")
    
    # Check if Ollama is installed
    if not check_ollama_installation():
        print("Ollama is not installed. Installing now...")
        if not install_ollama():
            print("Failed to install Ollama. Please install it manually from https://ollama.com/download")
            sys.exit(1)
    
    # Pull the model
    if not pull_model():
        print("Failed to pull the model. Please check your internet connection and try again.")
        sys.exit(1)
    
    # Test the connection
    if not test_ollama_connection():
        print("Failed to connect to Ollama. Please make sure the service is running.")
        print("You can start Ollama by running 'ollama serve' in a terminal.")
        sys.exit(1)
    
    # Set up environment
    setup_environment()
    
    print("\nSetup completed successfully!")
    print("\nTo start using the customer story tracker:")
    print("1. Make sure Ollama is running (run 'ollama serve' in a terminal)")
    print("2. Run the tracker with: python main.py")
    print("\nFor testing without Ollama, use: python main.py --mock-llm")

if __name__ == '__main__':
    main() 
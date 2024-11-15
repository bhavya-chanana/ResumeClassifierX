import torch
import subprocess
import sys
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def diagnose_cuda_issues():
    """Diagnose why CUDA isn't being detected"""
    
    print("=== CUDA Detection Diagnostic ===\n")
    
    # 1. Check PyTorch CUDA Build
    print(f"PyTorch Version: {torch.__version__}")
    print(f"PyTorch CUDA Build Available: {torch.version.cuda is not None}")
    
    # 2. Check NVIDIA Driver
    try:
        nvidia_smi = subprocess.check_output("nvidia-smi", shell=True)
        print("\nNVIDIA Driver Detected:")
        print(nvidia_smi.decode())
    except:
        print("\nError: NVIDIA Driver not found or nvidia-smi not accessible")
    
    # 3. Check Environment Variables
    cuda_path = os.environ.get('CUDA_PATH')
    cuda_home = os.environ.get('CUDA_HOME')
    
    print("\nEnvironment Variables:")
    print(f"CUDA_PATH: {cuda_path}")
    print(f"CUDA_HOME: {cuda_home}")
    
    # 4. Check PyTorch CUDA Details
    if torch.cuda.is_available():
        print("\nCUDA Device Properties:")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {props.name}")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Total Memory: {props.total_memory / 1024**2:.0f} MB")
    else:
        print("\nPossible issues:")
        print("1. NVIDIA drivers not properly installed")
        print("2. CUDA toolkit not properly installed")
        print("3. PyTorch installed without CUDA support")
        print("4. Environment variables not set correctly")
        print("\nTry running:")
        print("pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

if __name__ == "__main__":
    diagnose_cuda_issues()

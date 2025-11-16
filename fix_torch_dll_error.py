"""
Fix Torch DLL Error on Windows
==============================

This script helps diagnose and fix the common torch DLL initialization error on Windows.
"""

import sys
import subprocess
import os

def check_vc_redist():
    """Check if Visual C++ Redistributables are installed"""
    print("Checking for Visual C++ Redistributables...")
    # Common installation paths
    vc_paths = [
        r"C:\Program Files\Microsoft Visual C++ Redistributable",
        r"C:\Program Files (x86)\Microsoft Visual C++ Redistributable"
    ]
    
    found = False
    for path in vc_paths:
        if os.path.exists(path):
            print(f"  ✓ Found at: {path}")
            found = True
    
    if not found:
        print("  ✗ Visual C++ Redistributables not found")
        print("\n  Please install from:")
        print("  https://aka.ms/vs/17/release/vc_redist.x64.exe")
        return False
    return True

def reinstall_torch():
    """Reinstall torch with CPU-only version (more stable on Windows)"""
    print("\n" + "="*60)
    print("Reinstalling torch (CPU-only version)...")
    print("="*60)
    
    try:
        # Uninstall existing torch
        print("\n1. Uninstalling existing torch...")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "torch", "-y"], check=True)
        
        # Install CPU-only version
        print("\n2. Installing torch (CPU-only)...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "torch",
            "--index-url", "https://download.pytorch.org/whl/cpu"
        ], check=True)
        
        print("\n✓ Torch reinstalled successfully!")
        print("\nTry importing torch now:")
        print("  python -c 'import torch; print(torch.__version__)'")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error during reinstallation: {e}")
        return False

def test_torch():
    """Test if torch can be imported"""
    print("\n" + "="*60)
    print("Testing torch import...")
    print("="*60)
    
    try:
        import torch
        print(f"✓ Torch imported successfully!")
        print(f"  Version: {torch.__version__}")
        print(f"  Device: {torch.device('cpu')}")
        return True
    except Exception as e:
        print(f"✗ Torch import failed: {e}")
        return False

def main():
    """Main fix routine"""
    print("="*60)
    print("Torch DLL Error Fixer")
    print("="*60)
    
    # Check VC++ Redistributables
    vc_ok = check_vc_redist()
    
    # Test torch
    torch_ok = test_torch()
    
    if torch_ok:
        print("\n✓ Torch is working correctly!")
        return
    
    print("\n" + "="*60)
    print("Torch is not working. Attempting to fix...")
    print("="*60)
    
    if not vc_ok:
        print("\n⚠ Please install Visual C++ Redistributables first:")
        print("  https://aka.ms/vs/17/release/vc_redist.x64.exe")
        print("\nThen run this script again.")
        return
    
    # Ask user if they want to reinstall
    response = input("\nDo you want to reinstall torch (CPU-only version)? (y/n): ").strip().lower()
    if response == 'y':
        reinstall_torch()
        test_torch()
    else:
        print("\nManual fix options:")
        print("1. Install Visual C++ Redistributables:")
        print("   https://aka.ms/vs/17/release/vc_redist.x64.exe")
        print("\n2. Reinstall torch (CPU-only):")
        print("   pip uninstall torch -y")
        print("   pip install torch --index-url https://download.pytorch.org/whl/cpu")
        print("\n3. Or use CUDA version if you have NVIDIA GPU:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

if __name__ == "__main__":
    main()


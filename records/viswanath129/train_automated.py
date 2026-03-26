#!/usr/bin/env python3
"""
OpenAI Parameter Golf Challenge - Automated Training Script
Runs complete training pipeline with verification
Usage: python train_automated.py
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime

# Colors for output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_section(title):
    """Print a section header"""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}{title.center(60)}{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}\n")

def print_success(msg):
    """Print success message"""
    print(f"{Colors.GREEN}✓{Colors.END} {msg}")

def print_error(msg):
    """Print error message"""
    print(f"{Colors.RED}✗{Colors.END} {msg}")
    sys.exit(1)

def print_info(msg):
    """Print info message"""
    print(f"{Colors.YELLOW}→{Colors.END} {msg}")

def run_command(cmd, description=""):
    """Run a shell command with error handling"""
    if description:
        print_info(description)
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=False)
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed: {cmd}")
        return False

def check_gpu():
    """Check if GPUs are available"""
    print_section("STEP 1: Verifying GPU Setup")

    try:
        result = subprocess.run(
            "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        gpus = result.stdout.strip().split('\n')
        print_info(f"Found {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus, 1):
            print(f"  GPU {i}: {gpu}")

        if len(gpus) < 8:
            print_error(f"Need 8 GPUs, found {len(gpus)}")

        print_success(f"{len(gpus)} GPUs ready for training")
        return True
    except Exception as e:
        print_error(f"CUDA not available: {e}")
        return False

def install_dependencies():
    """Install required Python packages"""
    print_section("STEP 2: Installing Dependencies")

    packages = ["torch", "sentencepiece", "numpy"]
    for pkg in packages:
        print_info(f"Installing {pkg}...")
        if not run_command(f"pip install -q {pkg}", ""):
            print_error(f"Failed to install {pkg}")

    print_success("All dependencies installed")
    return True

def prepare_data():
    """Prepare FineWeb dataset"""
    print_section("STEP 3: Preparing FineWeb Dataset")

    # Clone if not exists
    if not Path("parameter-golf").exists():
        print_info("Cloning official repository...")
        run_command("git clone --depth 1 https://github.com/openai/parameter-golf parameter-golf")

    os.chdir("parameter-golf")

    # Download data if not exists
    data_dir = Path("data/datasets/fineweb10B_sp1024")
    if not data_dir.exists() or len(list(data_dir.glob("fineweb_train_*.bin"))) == 0:
        print_info("Downloading FineWeb data (20-30 minutes)...")
        run_command("python data/cached_challenge_fineweb.py --variant sp1024")

    # Verify data
    tokenizer = Path("data/tokenizers/fineweb_1024_bpe.model")
    if not tokenizer.exists():
        print_error("Tokenizer not found")

    train_files = list(data_dir.glob("fineweb_train_*.bin"))
    val_files = list(data_dir.glob("fineweb_val_*.bin"))

    print_success(f"Data ready: {len(train_files)} train files, {len(val_files)} val files")
    return True

def setup_code():
    """Setup training code"""
    print_section("STEP 4: Setting Up Training Code")

    if not Path("train_gpt.py").exists():
        print_error("train_gpt.py not found")

    # Verify syntax
    print_info("Verifying Python syntax...")
    result = subprocess.run(
        "python -m py_compile train_gpt.py",
        shell=True,
        capture_output=True
    )

    if result.returncode != 0:
        print_error(f"Syntax error in train_gpt.py:\n{result.stderr.decode()}")

    print_success("Training code verified")
    return True

def run_training():
    """Run the training"""
    print_section("STEP 5: Running Training")

    print_info("Training on 8 GPUs (max 600 seconds)")
    print("  Expected time: ~10 minutes")

    # Create output directory
    Path("logs").mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/training_{timestamp}.log"

    print_info(f"Logging to: {log_file}\n")

    # Run training
    start_time = time.time()
    cmd = f"torchrun --standalone --nproc_per_node=8 train_gpt.py"

    with open(log_file, "w") as f:
        result = subprocess.run(
            cmd,
            shell=True,
            stdout=f,
            stderr=subprocess.STDOUT
        )

    elapsed = time.time() - start_time

    if result.returncode != 0:
        print_error(f"Training failed. Check {log_file}")

    print_success(f"Training completed in {elapsed:.1f} seconds")
    return log_file

def verify_results(log_file):
    """Verify training results"""
    print_section("STEP 6: Verifying Results")

    model_file = Path("final_model.int8.ptz")
    if not model_file.exists():
        print_error("Model artifact not created")

    size = model_file.stat().st_size
    size_mb = size / (1024 * 1024)

    if size > 16_000_000:
        print_error(f"Model too large: {size_mb:.1f} MB (limit: 16 MB)")

    print_success(f"Model artifact: {size_mb:.1f} MB (limit: 16 MB) ✓")

    # Try to extract metrics
    try:
        with open(log_file, "r") as f:
            content = f.read()

        # Look for BPB score
        for line in content.split('\n'):
            if 'bpb' in line.lower():
                print_info(f"Score: {line.strip()}")
                break
    except:
        pass

    print_success(f"Log file: {log_file}")
    return True

def main():
    """Main execution"""
    print(f"{Colors.BLUE}")
    print("╔" + "="*58 + "╗")
    print("║  OpenAI Parameter Golf - Automated Training Script  ║")
    print("║  Requires: 8x H100 GPUs, PyTorch 2.4+              ║")
    print("╚" + "="*58 + "╝")
    print(f"{Colors.END}")

    try:
        # Step 1: Check GPU
        if not check_gpu():
            return False

        # Step 2: Install dependencies
        if not install_dependencies():
            return False

        # Step 3: Prepare data
        if not prepare_data():
            return False

        # Step 4: Setup code
        if not setup_code():
            return False

        # Step 5: Run training
        log_file = run_training()

        # Step 6: Verify results
        if not verify_results(log_file):
            return False

        # Summary
        print_section("TRAINING COMPLETE! 🎉")
        print(f"{Colors.GREEN}")
        print("NEXT STEPS:")
        print("1. Update submission.json with metrics")
        print("2. Create GitHub repository")
        print("3. Fork https://github.com/openai/parameter-golf")
        print("4. Submit pull request")
        print()
        print("See FINAL_CHECKLIST.md for detailed instructions")
        print(f"{Colors.END}")

        return True

    except KeyboardInterrupt:
        print_error("\nTraining interrupted by user")
        return False
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

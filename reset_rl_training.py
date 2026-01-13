"""
Reset RL Training - Backup and clean old models for fresh start with optimized hyperparameters

This script will:
1. Backup old model files (optional)
2. Clean up old checkpoints and state
3. Prepare for fresh training with optimized hyperparameters
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

def backup_model(model_dir: str = "models/rl_models", backup_dir: str = "models/rl_models_backup"):
    """Backup existing model files before reset"""
    model_path = Path(model_dir)
    backup_path = Path(backup_dir)
    
    if not model_path.exists():
        print(f"[INFO] No model directory found at {model_dir}. Nothing to backup.")
        return False
    
    # Create backup directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_path / f"backup_{timestamp}"
    backup_path.mkdir(parents=True, exist_ok=True)
    
    print(f"[BACKUP] Backing up model from {model_dir} to {backup_path}...")
    
    # Copy important files
    files_to_backup = [
        "rl_training_state.json",
        "checkpoints",
        "experience_buffers.json",
        "tensorboard"
    ]
    
    backed_up = False
    for item in files_to_backup:
        source = model_path / item
        if source.exists():
            dest = backup_path / item
            if source.is_dir():
                shutil.copytree(source, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(source, dest)
            print(f"[BACKUP] ✓ Backed up: {item}")
            backed_up = True
    
    if backed_up:
        print(f"[BACKUP] ✅ Backup completed: {backup_path}")
        return True
    else:
        print(f"[BACKUP] ⚠️ No files found to backup")
        backup_path.rmdir()  # Remove empty backup directory
        return False

def clean_model(model_dir: str = "models/rl_models", keep_backup: bool = True):
    """Clean old model files for fresh start"""
    model_path = Path(model_dir)
    
    if not model_path.exists():
        print(f"[INFO] No model directory found at {model_dir}. Nothing to clean.")
        return
    
    print(f"[CLEAN] Cleaning model directory: {model_dir}")
    
    # Backup first if requested
    if keep_backup:
        backup_model(model_dir)
    
    # Files/directories to remove
    items_to_remove = [
        "rl_training_state.json",
        "checkpoints",
        "experience_buffers.json",
        "tensorboard",
        "final_model.zip",
        "best_model_*.zip"
    ]
    
    removed_count = 0
    for item_pattern in items_to_remove:
        if "*" in item_pattern:
            # Handle glob patterns
            for item in model_path.glob(item_pattern):
                if item.is_file():
                    item.unlink()
                    print(f"[CLEAN] ✓ Removed: {item.name}")
                    removed_count += 1
                elif item.is_dir():
                    shutil.rmtree(item)
                    print(f"[CLEAN] ✓ Removed directory: {item.name}")
                    removed_count += 1
        else:
            item = model_path / item_pattern
            if item.exists():
                if item.is_file():
                    item.unlink()
                    print(f"[CLEAN] ✓ Removed: {item.name}")
                    removed_count += 1
                elif item.is_dir():
                    shutil.rmtree(item)
                    print(f"[CLEAN] ✓ Removed directory: {item.name}")
                    removed_count += 1
    
    if removed_count > 0:
        print(f"[CLEAN] ✅ Cleaned {removed_count} item(s). Ready for fresh training!")
    else:
        print(f"[CLEAN] ⚠️ No files found to clean (may already be clean)")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Reset RL training for fresh start")
    parser.add_argument("--model-dir", type=str, default="models/rl_models",
                       help="Model directory to clean (default: models/rl_models)")
    parser.add_argument("--backup-dir", type=str, default="models/rl_models_backup",
                       help="Backup directory (default: models/rl_models_backup)")
    parser.add_argument("--no-backup", action="store_true",
                       help="Skip backup (WARNING: This will permanently delete old models!)")
    parser.add_argument("--backup-only", action="store_true",
                       help="Only backup, don't clean")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("RL Training Reset Tool")
    print("=" * 60)
    print(f"Model directory: {args.model_dir}")
    print(f"Backup directory: {args.backup_dir}")
    print()
    
    if args.backup_only:
        backup_model(args.model_dir, args.backup_dir)
    else:
        clean_model(args.model_dir, keep_backup=not args.no_backup)
    
    print()
    print("=" * 60)
    print("Next steps:")
    print("1. Start training with: python rl_training.py --dashboard")
    print("2. Monitor progress in dashboard and TensorBoard")
    print("3. The new model will use optimized hyperparameters")
    print("=" * 60)

if __name__ == "__main__":
    main()


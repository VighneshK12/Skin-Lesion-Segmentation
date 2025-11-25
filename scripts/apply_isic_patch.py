#!/usr/bin/env python3
import importlib
import shutil
from pathlib import Path

def main():
    # locate nnunetv2 install
    trainer_pkg = importlib.import_module("nnunetv2.training.nnUNetTrainer")
    pkg_dir = Path(trainer_pkg.__file__).parent  # .../nnunetv2/training/nnUNetTrainer

    dst = pkg_dir / "ISICTrainer.py"
    src = Path(__file__).resolve().parent.parent / "patches" / "ISICTrainer.py"

    print(f"[info] nnUNetTrainer package dir: {pkg_dir}")
    print(f"[info] copying {src} -> {dst}")

    if not src.exists():
        raise FileNotFoundError(f"Patch file not found at {src}")

    shutil.copy2(src, dst)

    # clear old .pyc if present
    pycache = pkg_dir / "__pycache__"
    if pycache.exists():
        for f in pycache.glob("ISICTrainer*.pyc"):
            print(f"[info] removing stale pyc: {f}")
            f.unlink()

    print("[ok] ISICTrainer patched")

if __name__ == "__main__":
    main()

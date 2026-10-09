#!/usr/bin/env python3
import sys
import os
from huggingface_hub import snapshot_download

def has_safetensors(path):
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(".safetensors"):
                return True
    return False


def main():
    if len(sys.argv) != 3:
        print("Usage: python download_models.py <repo_id> <local_dir>")
        sys.exit(1)

    repo_id = sys.argv[1]
    local_dir = sys.argv[2]

    print(f"Downloading HuggingFace model (safetensors-first):")
    print(f"  Repo ID:   {repo_id}")
    print(f"  Local dir: {local_dir}\n")

    # -------------------------------
    # Pass 1: try safetensors only
    # -------------------------------
    print("→ Attempt 1: downloading safetensors (preferred)\n")

    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,

        allow_patterns=[
            "*.json",
            "*.txt",
            "*.model",
            "*.safetensors"
        ],

        ignore_patterns=[
            "pytorch_model.bin",
            "tf_model.h5",
            "flax_model.msgpack",
            "*.md",
            "training_args.bin"
        ],
    )

    if has_safetensors(local_dir):
        print("\n✔ Safetensors found. Download complete.")
        print("✔ PyTorch will load this model without .bin files.")
        return

    # -------------------------------
    # Pass 2: fallback to .bin
    # -------------------------------
    print("\n⚠ No safetensors found.")
    print("→ Falling back to pytorch_model.bin\n")

    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,

        allow_patterns=[
            "*.json",
            "*.txt",
            "*.model",
            "pytorch_model.bin"
        ],

        ignore_patterns=[
            "*.safetensors",
            "tf_model.h5",
            "flax_model.msgpack",
            "*.md",
            "training_args.bin"
        ],
    )

    print("\n✔ Fallback download complete (pytorch_model.bin).")
    print("✔ Model is ready for offline PyTorch use.")


if __name__ == "__main__":
    main()
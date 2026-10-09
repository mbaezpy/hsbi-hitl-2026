# utils/download_gensim.py
#!/usr/bin/env python3
"""
Download a Gensim model into a specific directory without loading it in memory.

Usage:
  python utils/download_gensim.py <dataset_name> <output_dir>

Example:
  python utils/download_gensim.py glove-wiki-gigaword-50 models/
"""

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Download a Gensim model into a specific folder.")
    parser.add_argument("dataset_name", help="Name from gensim.downloader (e.g., glove-wiki-gigaword-50)")
    parser.add_argument("output_dir", help="Directory where the model folder will be created")
    args = parser.parse_args()

    # Resolve paths
    dataset = args.dataset_name
    out_dir = os.path.abspath(args.output_dir)

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Point Gensim's cache to the chosen directory (it creates <out_dir>/<dataset>/)
    os.environ["GENSIM_DATA_DIR"] = out_dir

    # Import after setting env var so gensim picks it up
    try:
        import gensim.downloader as api
    except Exception as e:
        print("ERROR: Could not import gensim. Install it with `pip install gensim`.", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Target cache dir: {out_dir}")
    print(f"[INFO] Dataset name: {dataset}")

    try:
        # return_path=True triggers download (if missing) and returns the local path,
        # without loading the model into memory.
        local_path = api.load(dataset, return_path=True)
    except Exception as e:
        print(f"ERROR: Failed to download '{dataset}'.", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        sys.exit(2)

    # Expect something like <out_dir>/<dataset>/...
    print(f"[OK] Local dataset path: {local_path}")

    # Sanity hint for how to load later from notebooks
    ds_dir = os.path.join(out_dir, dataset)
    if os.path.isdir(ds_dir):
        print(f"[HINT] Folder ready: {ds_dir}")
    else:
        # Some datasets are a single file directly under out_dir; this is fine.
        print(f"[HINT] Model files placed under: {out_dir}")

    print("[DONE] You can now load this model offline by setting GENSIM_DATA_DIR to your models folder.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Download a file from a Sciebo public share.

Usage:
    python download_sciebo.py \
        https://hsbi.sciebo.de/s/ABC123 \
        --output downloads
"""

from pathlib import Path
import argparse
import requests
from urllib.parse import urlparse


def make_download_url(share_url: str) -> str:
    """
    Convert a Sciebo share URL into a direct download URL.

    Example:
        https://hsbi.sciebo.de/s/ABC123

    becomes

        https://hsbi.sciebo.de/s/ABC123/download
    """
    return share_url.rstrip("/") + "/download"


def download_file(share_url: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    download_url = make_download_url(share_url)

    print(f"Downloading from:\n{download_url}")

    response = requests.get(download_url, stream=True, allow_redirects=True)
    response.raise_for_status()

    # Try to obtain filename from headers
    filename = None

    cd = response.headers.get("Content-Disposition")
    if cd and "filename=" in cd:
        filename = cd.split("filename=")[-1].strip("\"' ")

    if filename is None:
        filename = Path(urlparse(response.url).path).name
        if filename == "download":
            filename = "downloaded_file"

    destination = output_dir / filename

    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"✓ Saved to {destination}")


def main():
    parser = argparse.ArgumentParser(
        description="Download a file from a Sciebo public share."
    )

    parser.add_argument(
        "url",
        help="Sciebo share URL"
    )

    parser.add_argument(
        "output_dir",
        help="Destination directory"
    )

    args = parser.parse_args()

    download_file(
        args.url,
        Path(args.output_dir)
    )

if __name__ == "__main__":
    main()
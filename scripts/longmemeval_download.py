#!/usr/bin/env python3
"""Download LongMemEval benchmark data from HuggingFace.

Usage:
    python3 scripts/longmemeval_download.py              # Download M variant (default)
    python3 scripts/longmemeval_download.py --variant s   # Download S variant
    python3 scripts/longmemeval_download.py --variant oracle
"""

import argparse
import sys
from pathlib import Path

import httpx

BASE_URL = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"
VARIANTS = {
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
    "oracle": "longmemeval_oracle.json",
}

# Local output filenames (without _cleaned suffix for simplicity)
LOCAL_NAMES = {
    "s": "longmemeval_s.json",
    "m": "longmemeval_m.json",
    "oracle": "longmemeval_oracle.json",
}


def download(variant: str, output_dir: Path) -> Path:
    remote_filename = VARIANTS[variant]
    local_filename = LOCAL_NAMES[variant]
    url = f"{BASE_URL}/{remote_filename}"
    output = output_dir / local_filename

    if output.exists():
        size_mb = output.stat().st_size / (1024 * 1024)
        print(f"  Already exists: {output} ({size_mb:.1f} MB)")
        return output

    print(f"  Downloading {url}")
    with httpx.stream("GET", url, follow_redirects=True, timeout=300) as resp:
        if resp.status_code != 200:
            print(f"  ERROR: HTTP {resp.status_code}")
            sys.exit(1)
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(output, "wb") as f:
            for chunk in resp.iter_bytes(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    mb = downloaded / (1024 * 1024)
                    print(f"\r  {mb:.1f} MB ({pct:.0f}%)", end="", flush=True)
        print()

    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"  Saved: {output} ({size_mb:.1f} MB)")
    return output


def main():
    parser = argparse.ArgumentParser(description="Download LongMemEval benchmark data")
    parser.add_argument(
        "--variant",
        choices=list(VARIANTS.keys()),
        default="m",
        help="Dataset variant (default: m)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Output directory (default: current directory)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"LongMemEval '{args.variant}' variant")
    download(args.variant, output_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Download arXiv PDFs for all papers in meta.jsonl

Usage:
  python3 download_arxiv_pdfs.py [--output-dir /path/to/pdfs] [--limit 1000] [--sleep 1.0]

Features:
- Reads arxiv_id from meta.jsonl
- Downloads PDFs with polite rate limiting
- Handles both new (2506.04004) and legacy (cs/0604034) arXiv IDs
- Converts ID to safe filename (2506.04004 -> 2506_04004.pdf)
- Resumes interrupted downloads
- Progress tracking
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterator, Optional, Tuple


def normalize_arxiv_id(aid: str) -> str:
    """Normalize arXiv ID to versionless form."""
    if not isinstance(aid, str):
        return ""
    aid = aid.strip()
    if not aid:
        return ""
    # Remove version suffix if present
    aid = aid.split("v")[0]
    return aid


def arxiv_id_to_filename(aid: str) -> str:
    """Convert arXiv ID to safe filename."""
    if not aid:
        return ""
    # Replace dots and slashes with underscores
    safe_name = aid.replace(".", "_").replace("/", "_")
    return f"{safe_name}.pdf"


def get_pdf_url(aid: str) -> str:
    """Get arXiv PDF URL for given ID."""
    normalized = normalize_arxiv_id(aid)
    if not normalized:
        return ""
    
    # Handle new style (YYYY.NNNNN)
    if re.match(r"\d{4}\.\d{4,5}", normalized):
        return f"https://arxiv.org/pdf/{normalized}.pdf"
    
    # Handle legacy style (cs/YYNNNNN)
    if re.match(r"[a-z\-]+(?:\.[A-Z]{2})?/\d{7}", normalized):
        return f"https://arxiv.org/pdf/{normalized}.pdf"
    
    return ""


def download_pdf(url: str, output_path: Path, retries: int = 3) -> bool:
    """Download PDF file with retries."""
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "arxiv-pdf-downloader/1.0 (https://github.com/your-repo)"
                }
            )
            
            with urllib.request.urlopen(req, timeout=30) as response:
                if response.status == 200:
                    with open(output_path, 'wb') as f:
                        f.write(response.read())
                    return True
                else:
                    print(f"HTTP {response.status} for {url}", file=sys.stderr)
                    
        except Exception as e:
            if attempt < retries:
                print(f"Attempt {attempt + 1} failed for {url}: {e}", file=sys.stderr)
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"Failed to download {url} after {retries + 1} attempts: {e}", file=sys.stderr)
    
    return False


def iter_arxiv_ids(meta_path: Path) -> Iterator[Tuple[str, str]]:
    """Iterate over arxiv_id from meta.jsonl, yielding (arxiv_id, paper_id)."""
    with open(meta_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"JSON parse error at line {line_num}: {e}", file=sys.stderr)
                continue
            
            arxiv_id = record.get("arxiv_id", "")
            paper_id = record.get("id", f"line_{line_num}")
            
            if arxiv_id and normalize_arxiv_id(arxiv_id):
                yield arxiv_id, paper_id


def main():
    parser = argparse.ArgumentParser(description="Download arXiv PDFs from meta.jsonl")
    parser.add_argument("--meta-file", default="index/specter2_merged/meta.jsonl",
                       help="Path to meta.jsonl file")
    parser.add_argument("--output-dir", default="l",
                       help="Directory to save PDFs")
    parser.add_argument("--limit", type=int, default=0,
                       help="Limit number of downloads (0 = no limit)")
    parser.add_argument("--sleep", type=float, default=0.1,
                       help="Sleep between downloads (seconds)")
    parser.add_argument("--retries", type=int, default=3,
                       help="Number of retries per download")
    parser.add_argument("--progress", action="store_true",
                       help="Show progress")
    
    args = parser.parse_args()
    
    meta_path = Path(args.meta_file)
    if not meta_path.exists():
        print(f"Error: meta.jsonl not found at {meta_path}", file=sys.stderr)
        return 1
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Count total papers with arxiv_id
    print("Counting papers with arXiv IDs...")
    total_papers = 0
    for _ in iter_arxiv_ids(meta_path):
        total_papers += 1
    
    print(f"Found {total_papers} papers with arXiv IDs")
    
    # Download PDFs
    downloaded = 0
    skipped = 0
    failed = 0
    
    for i, (arxiv_id, paper_id) in enumerate(iter_arxiv_ids(meta_path)):
        if args.limit > 0 and i >= args.limit:
            break
        
        if args.progress and i % 100 == 0:
            print(f"Progress: {i}/{total_papers} ({i/total_papers*100:.1f}%)")
        
        normalized_id = normalize_arxiv_id(arxiv_id)
        filename = arxiv_id_to_filename(normalized_id)
        output_path = output_dir / filename
        
        # Skip if already exists
        if output_path.exists():
            skipped += 1
            if args.progress:
                print(f"Skipped (exists): {filename}")
            continue
        
        # Get PDF URL
        pdf_url = get_pdf_url(arxiv_id)
        if not pdf_url:
            print(f"Invalid arXiv ID: {arxiv_id} (paper: {paper_id})", file=sys.stderr)
            failed += 1
            continue
        
        # Download
        if args.progress:
            print(f"Downloading: {filename} ({pdf_url})")
        
        success = download_pdf(pdf_url, output_path, args.retries)
        if success:
            downloaded += 1
        else:
            failed += 1
        
        # Polite sleep
        if args.sleep > 0:
            time.sleep(args.sleep)
    
    # Summary
    print(f"\nDownload Summary:")
    print(f"  Total papers: {total_papers}")
    print(f"  Downloaded: {downloaded}")
    print(f"  Skipped (exists): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Output directory: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

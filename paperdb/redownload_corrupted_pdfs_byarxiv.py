#!/usr/bin/env python3
"""
Redownload corrupted PDFs for a given year prefix (e.g., 24 for 2024) using arxiv library.

This script:
1. Identifies corrupted PDFs (1853 bytes) in arxiv_pdfs directory for a year prefix
2. Saves corrupted arxiv_ids to a local file
3. Uses arxiv library to redownload and replace corrupted files

Examples:
  python3 redownload_corrupted_pdfs_byarxiv.py --year 24 --progress
  python3 redownload_corrupted_pdfs_byarxiv.py --year 2020 --dry-run  # Identify only
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Set
import arxiv
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
ARXIV_PDFS_DIR = "/shared/hdd/arxiv_pdfs"
# If not provided via CLI, will default to: corrupted_<yy>_pdf_ids.txt
DEFAULT_CORRUPTED_IDS_FILE = None
MAX_WORKERS = 10  # Concurrent downloads
CORRUPTED_SIZE = 1853  # Size in bytes that indicates corruption


def normalize_year_to_prefix(year_value: str) -> str:
    """Normalize year value to two-digit arXiv prefix string (e.g., '24')."""
    year_value = str(year_value).strip()
    # Accept forms like '24' or '2024'
    if len(year_value) >= 2:
        two_digit = year_value[-2:]
    else:
        two_digit = year_value.zfill(2)
    # Ensure it's two digits
    return f"{int(two_digit):02d}"


def find_corrupted_pdfs_for_year(pdfs_dir: Path, year_prefix: str) -> Set[str]:
    """Find all <year_prefix>-prefix PDFs with corrupted size (1853 bytes)."""
    corrupted_ids = set()
    
    print(f"Scanning for corrupted {year_prefix}-prefix PDFs in {pdfs_dir}...")
    
    for pdf_file in pdfs_dir.glob(f"{year_prefix}*.pdf"):
        if pdf_file.stat().st_size == CORRUPTED_SIZE:
            # Convert filename back to arxiv_id: 2408_12445.pdf -> 2408.12445
            filename = pdf_file.stem
            arxiv_id = filename.replace("_", ".")
            corrupted_ids.add(arxiv_id)
    
    return corrupted_ids


def save_corrupted_ids(corrupted_ids: Set[str], output_file: str):
    """Save corrupted arxiv_ids to a text file, one per line."""
    with open(output_file, 'w') as f:
        for arxiv_id in sorted(corrupted_ids):
            f.write(f"{arxiv_id}\n")
    print(f"Saved {len(corrupted_ids)} corrupted arxiv_ids to {output_file}")


def load_corrupted_ids(input_file: str) -> Set[str]:
    """Load corrupted arxiv_ids from a text file."""
    corrupted_ids = set()
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                corrupted_ids.add(line)
    return corrupted_ids


def arxiv_id_to_filename(arxiv_id: str) -> str:
    """Convert arxiv_id to safe filename."""
    return f"{arxiv_id.replace('.', '_')}.pdf"


def download_pdf_via_arxiv_lib(arxiv_id: str, output_path: Path, progress: bool = False) -> bool:
    """Download PDF using arxiv library."""
    try:
        # Rate limiting: wait 0.5 seconds before each request
        time.sleep(0.5)
        
        # Create arxiv client
        client = arxiv.Client()
        
        # Search for the paper by ID
        search = arxiv.Search(id_list=[arxiv_id])
        results = list(client.results(search))
        
        if not results:
            if progress:
                print(f"Paper {arxiv_id} not found in arXiv")
            return False
        
        # Get the first (and should be only) result
        paper = results[0]
        
        # Download PDF to the specified path
        paper.download_pdf(dirpath=str(output_path.parent), filename=output_path.name)
        
        # Verify file size is not corrupted
        if output_path.stat().st_size == CORRUPTED_SIZE:
            if progress:
                print(f"Warning: {arxiv_id} still has corrupted size after redownload")
            return False
        
        if progress:
            print(f"Successfully redownloaded {arxiv_id} ({output_path.stat().st_size} bytes)")
        return True
        
    except Exception as e:
        if progress:
            print(f"Failed to download {arxiv_id}: {e}")
        return False


def redownload_corrupted_pdfs(corrupted_ids: Set[str], pdfs_dir: Path, 
                             max_workers: int, progress: bool, dry_run: bool) -> dict:
    """Redownload all corrupted PDFs concurrently."""
    if dry_run:
        print(f"DRY RUN: Would redownload {len(corrupted_ids)} corrupted PDFs")
        return {"success": 0, "failed": 0, "skipped": len(corrupted_ids)}
    
    results = {"success": 0, "failed": 0, "skipped": 0}
    
    print(f"Starting redownload of {len(corrupted_ids)} corrupted PDFs with {max_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit download tasks
        future_to_id = {}
        for arxiv_id in corrupted_ids:
            filename = arxiv_id_to_filename(arxiv_id)
            output_path = pdfs_dir / filename
            
            future = executor.submit(download_pdf_via_arxiv_lib, arxiv_id, output_path, progress)
            future_to_id[future] = arxiv_id
        
        # Process completed downloads
        for future in as_completed(future_to_id):
            arxiv_id = future_to_id[future]
            try:
                success = future.result()
                if success:
                    results["success"] += 1
                else:
                    results["failed"] += 1
            except Exception as e:
                if progress:
                    print(f"Task failed for {arxiv_id}: {e}")
                results["failed"] += 1
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Redownload corrupted PDFs for a given year prefix using arxiv library")
    parser.add_argument("--year", required=True, help="Year as 2 or 4 digits (e.g., 24 or 2024)")
    parser.add_argument("--pdfs-dir", default=ARXIV_PDFS_DIR, help="Directory containing PDFs")
    parser.add_argument("--corrupted-ids-file", default=DEFAULT_CORRUPTED_IDS_FILE, 
                       help="File to save/load corrupted arxiv_ids (defaults to corrupted_<yy>_pdf_ids.txt)")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS, 
                       help="Maximum concurrent downloads")
    parser.add_argument("--progress", action="store_true", help="Show detailed progress")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Only identify corrupted files, don't download")
    parser.add_argument("--load-ids", action="store_true", 
                       help="Load corrupted IDs from file instead of scanning")
    
    args = parser.parse_args()
    
    pdfs_dir = Path(args.pdfs_dir)
    if not pdfs_dir.exists():
        print(f"Error: PDFs directory not found: {pdfs_dir}")
        return 1
    
    # Normalize and compute defaults based on year
    year_prefix = normalize_year_to_prefix(args.year)
    if args.corrupted_ids_file is None:
        args.corrupted_ids_file = f"corrupted_{year_prefix}_pdf_ids.txt"
    # Note: arxiv library handles User-Agent internally
    
    # Find or load corrupted IDs
    if args.load_ids:
        if not os.path.exists(args.corrupted_ids_file):
            print(f"Error: Corrupted IDs file not found: {args.corrupted_ids_file}")
            return 1
        corrupted_ids = load_corrupted_ids(args.corrupted_ids_file)
        print(f"Loaded {len(corrupted_ids)} corrupted arxiv_ids from {args.corrupted_ids_file}")
    else:
        corrupted_ids = find_corrupted_pdfs_for_year(pdfs_dir, year_prefix)
        print(f"Found {len(corrupted_ids)} corrupted {year_prefix}-prefix PDFs")
        
        if corrupted_ids:
            save_corrupted_ids(corrupted_ids, args.corrupted_ids_file)
    
    if not corrupted_ids:
        print("No corrupted PDFs found.")
        return 0
    
    # Redownload corrupted PDFs
    start_time = time.time()
    results = redownload_corrupted_pdfs(
        corrupted_ids, pdfs_dir, args.max_workers, args.progress, args.dry_run
    )
    
    elapsed = time.time() - start_time
    
    # Summary
    print(f"\nRedownload Summary:")
    print(f"  Total corrupted PDFs: {len(corrupted_ids)}")
    print(f"  Successfully redownloaded: {results['success']}")
    print(f"  Failed: {results['failed']}")
    print(f"  Skipped: {results['skipped']}")
    print(f"  Time elapsed: {elapsed:.1f}s")
    
    if not args.dry_run:
        print(f"  Corrupted IDs saved to: {args.corrupted_ids_file}")
        print(f"  PDFs directory: {pdfs_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

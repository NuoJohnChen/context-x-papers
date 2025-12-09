#!/usr/bin/env python3
"""
Download arXiv dataset from Kaggle and extract PDFs

Usage:
  python3 download_and_extract_arxiv.py [--output-dir /path/to/pdfs] [--limit 1000] [--sleep 1.0]

Features:
- Downloads arXiv metadata from Kaggle
- Reads arxiv_id from the downloaded metadata
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
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import kagglehub
except ImportError:
    print("Error: kagglehub not installed. Please install it with: pip install kagglehub")
    sys.exit(1)

try:
    import aiohttp
    import asyncio
    ASYNC_AVAILABLE = True
except ImportError:
    ASYNC_AVAILABLE = False


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


def download_pdf_parallel(args_tuple):
    """Download PDF file with retries - for parallel execution."""
    url, output_path, retries, progress_lock = args_tuple
    return download_pdf(url, output_path, retries)


async def download_pdf_async(session, url: str, output_path: Path, retries: int = 3) -> bool:
    """Download PDF file asynchronously with retries."""
    for attempt in range(retries + 1):
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    content = await response.read()
                    with open(output_path, 'wb') as f:
                        f.write(content)
                    return True
                else:
                    print(f"HTTP {response.status} for {url}", file=sys.stderr)
                    if attempt == retries:
                        return False
                    
        except Exception as e:
            if attempt < retries:
                print(f"Attempt {attempt + 1} failed for {url}: {e}", file=sys.stderr)
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"Failed to download {url} after {retries + 1} attempts: {e}", file=sys.stderr)
                return False
    
    return False


async def download_batch_async(download_tasks, max_concurrent: int, retries: int):
    """Download a batch of PDFs asynchronously with intelligent rate limiting."""
    connector = aiohttp.TCPConnector(
        limit=max_concurrent, 
        limit_per_host=max_concurrent,
        ttl_dns_cache=300,
        use_dns_cache=True
    )
    timeout = aiohttp.ClientTimeout(total=30)
    
    async with aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
        headers={
            "User-Agent": "arxiv-pdf-downloader/1.0 (https://github.com/your-repo)",
            "Accept": "application/pdf,*/*",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive"
        }
    ) as session:
        # Create semaphore to limit concurrent downloads
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def download_with_semaphore(task):
            async with semaphore:
                url, output_path, _, _ = task
                # Add small delay to be polite to arXiv
                await asyncio.sleep(0.05)
                return await download_pdf_async(session, url, output_path, retries)
        
        # Create all download tasks
        tasks = [download_with_semaphore(task) for task in download_tasks]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results


def iter_arxiv_ids_from_kaggle(meta_path: Path) -> Iterator[str]:
    """Iterate arXiv IDs from Kaggle arxiv-metadata JSONL (field name: id)."""
    with open(meta_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"JSON parse error at line {line_num}: {e}", file=sys.stderr)
                continue

            arxiv_id = record.get("id", "")
            if arxiv_id and normalize_arxiv_id(arxiv_id):
                yield arxiv_id


def download_kaggle_dataset(target_dir: str) -> str:
    """Download arXiv dataset from Kaggle and return the download path."""
    print(f"Downloading arXiv dataset to: {target_dir}")
    
    # Ensure target directory exists
    os.makedirs(target_dir, exist_ok=True)
    
    # Change to target directory for download
    original_cwd = os.getcwd()
    os.chdir(target_dir)
    
    try:
        # Download the dataset with simpler path handling
        download_path = kagglehub.dataset_download(
            "Cornell-University/arxiv"
        )
        
        print(f"Dataset downloaded to: {download_path}")
        
        # Verify the download
        print("\nFiles in the download directory:")
        try:
            for filename in os.listdir(download_path):
                print(f"- {filename}")
        except FileNotFoundError:
            print("Directory not found. The download might have failed.")
            sys.exit(1)
        
        return download_path
        
    finally:
        # Always restore original working directory
        os.chdir(original_cwd)


def find_metadata_file(download_path: str) -> str:
    """Find the metadata file in the downloaded dataset."""
    # Look for the metadata file
    metadata_files = [
        "arxiv-metadata-oai-snapshot.json",
        "arxiv-metadata-oai-snapshot.jsonl",
        "metadata.json",
        "metadata.jsonl"
    ]
    
    for filename in metadata_files:
        file_path = os.path.join(download_path, filename)
        if os.path.exists(file_path):
            print(f"Found metadata file: {file_path}")
            return file_path
    
    # If not found, list all files to help debug
    print("Metadata file not found. Available files:")
    for filename in os.listdir(download_path):
        print(f"- {filename}")
    
    raise FileNotFoundError("Could not find metadata file in downloaded dataset")


def main():
    parser = argparse.ArgumentParser(description="Download arXiv dataset from Kaggle and extract PDFs")
    parser.add_argument("--kaggle-dir", default="/shared/hdd/nuochen/arxivdata/kaggle/",
                       help="Directory to save Kaggle dataset")
    parser.add_argument("--output-dir", default="/shared/hdd/arxiv_pdfs",
                       help="Directory to save PDFs (incremental update)")
    parser.add_argument("--limit", type=int, default=0,
                       help="Limit number of downloads (0 = no limit)")
    parser.add_argument("--sleep", type=float, default=0.05,
                       help="Sleep between downloads (seconds, 0 for no sleep)")
    parser.add_argument("--retries", type=int, default=4,
                       help="Number of retries per download")
    parser.add_argument("--workers", type=int, default=40,
                       help="Number of parallel download workers")
    parser.add_argument("--async-mode", action="store_true",
                       help="Use async mode for maximum speed (requires aiohttp)")
    parser.add_argument("--progress", action="store_true",
                       help="Show progress")
    parser.add_argument("--skip-download", action="store_true",
                       help="Skip Kaggle download, use existing dataset")
    
    args = parser.parse_args()
    
    # Check async mode availability
    if args.async_mode and not ASYNC_AVAILABLE:
        print("Error: --async-mode requires aiohttp. Install with: pip install aiohttp")
        return 1
    
    # Download or use existing dataset
    if args.skip_download:
        # Look for existing dataset
        download_path = os.path.join(args.kaggle_dir, "Cornell-University", "arxiv")
        if not os.path.exists(download_path):
            print(f"Error: Existing dataset not found at {download_path}")
            print("Use --skip-download only if you have already downloaded the dataset")
            return 1
        print(f"Using existing dataset at: {download_path}")
    else:
        # Download the dataset
        download_path = download_kaggle_dataset(args.kaggle_dir)
    
    # Find the metadata file
    try:
        metadata_file = find_metadata_file(download_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    
    meta_path = Path(metadata_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Count total IDs
    print("Counting papers (from Kaggle metadata)...")
    total_papers = 0
    all_papers = []  # Store all papers for downloading
    
    for aid in iter_arxiv_ids_from_kaggle(meta_path):
        total_papers += 1
        all_papers.append(aid)

    print(f"Found {total_papers} papers with IDs")
    
    # Download PDFs (all papers)
    downloaded = 0
    skipped = 0
    failed = 0
    
    # Build cache of existing files in the output directory to speed skip checks
    existing_filenames = {p.name for p in output_dir.glob('*.pdf')}

    print(f"\nStarting {'async' if args.async_mode else 'parallel'} download of {total_papers} papers...")
    print(f"Using {args.workers} {'concurrent connections' if args.async_mode else 'parallel workers'}")
    
    # Prepare download tasks
    download_tasks = []
    for arxiv_id in all_papers:
        if args.limit > 0 and len(download_tasks) >= args.limit:
            break
            
        normalized_id = normalize_arxiv_id(arxiv_id)
        filename = arxiv_id_to_filename(normalized_id)
        output_path = output_dir / filename
        
        # Skip if already exists
        if filename in existing_filenames or output_path.exists():
            skipped += 1
            if args.progress:
                print(f"Skipped (exists): {filename}")
            continue
        
        # Get PDF URL
        pdf_url = get_pdf_url(arxiv_id)
        if not pdf_url:
            print(f"Invalid arXiv ID: {arxiv_id}", file=sys.stderr)
            failed += 1
            continue
        
        download_tasks.append((pdf_url, output_path, args.retries, None))
    
    print(f"Prepared {len(download_tasks)} papers for download")
    
    if args.async_mode:
        # Async download
        print("Starting async download...")
        try:
            results = asyncio.run(download_batch_async(download_tasks, args.workers, args.retries))
        except Exception as e:
            print(f"Async download failed: {e}", file=sys.stderr)
            print("Falling back to thread mode...")
            # Fallback to thread mode
            args.async_mode = False
            # Recursive call to thread mode
            return main()
        
        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Task {i} failed: {result}", file=sys.stderr)
                failed += 1
            elif result:
                downloaded += 1
                if args.progress:
                    task = download_tasks[i]
                    filename = arxiv_id_to_filename(normalize_arxiv_id(task[1].stem.replace('_', '.')))
                    print(f"Downloaded: {filename}")
            else:
                failed += 1
            
            # Progress update
            if args.progress and (downloaded + failed) % 100 == 0:
                total_processed = downloaded + failed + skipped
                print(f"Progress: {total_processed}/{len(download_tasks) + skipped} ({total_processed/(len(download_tasks) + skipped)*100:.1f}%)")
    
    else:
        # Parallel download (existing code)
        progress_lock = Lock()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(download_pdf_parallel, task): task 
                for task in download_tasks
            }
            
            # Process completed tasks
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    success = future.result()
                    if success:
                        downloaded += 1
                        if args.progress:
                            filename = arxiv_id_to_filename(normalize_arxiv_id(task[1].stem.replace('_', '.')))
                            print(f"Downloaded: {filename}")
                    else:
                        failed += 1
                except Exception as e:
                    print(f"Task failed: {e}", file=sys.stderr)
                    failed += 1
                
                # Progress update
                if args.progress and (downloaded + failed) % 100 == 0:
                    total_processed = downloaded + failed + skipped
                    print(f"Progress: {total_processed}/{len(download_tasks) + skipped} ({total_processed/(len(download_tasks) + skipped)*100:.1f}%)")
    
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

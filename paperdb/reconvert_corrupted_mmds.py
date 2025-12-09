#!/usr/bin/env python3
"""
Reconvert corrupted MMDs (130 bytes) to proper format, replacing wrong MMD files.

This script:
1. Scans /shared/hdd/arxiv_mmds for MMD files with size 130 bytes
2. Re-extracts MMD from the corresponding PDFs
3. Replaces the wrong MMD files with new ones

Usage:
  python3 reconvert_corrupted_mmds.py --progress
"""

import argparse
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Check PyMuPDF availability
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    print("Error: PyMuPDF (fitz) not installed. Please install it with: pip install PyMuPDF")
    sys.exit(1)


def find_corrupted_mmds(mmd_dir: Path) -> list:
    """Find all MMD files with corrupted size (130 bytes)."""
    corrupted_ids = []
    
    print(f"Scanning for corrupted MMD files (130 bytes) in {mmd_dir}...")
    
    for mmd_file in mmd_dir.glob("*.mmd"):
        if mmd_file.stat().st_size == 130:
            # Convert filename back to arxiv_id: 2508_12345.mmd -> 2508.12345
            filename = mmd_file.stem
            arxiv_id = filename.replace("_", ".")
            corrupted_ids.append(arxiv_id)
    
    return corrupted_ids


def extract_mmd_from_pdf(pdf_path: Path) -> str:
    """Extract text from first page using PyMuPDF and convert to markdown."""
    try:
        doc = None
        if hasattr(fitz, "open"):
            doc = fitz.open(pdf_path)
        elif hasattr(fitz, "Document"):
            doc = fitz.Document(pdf_path)
        else:
            raise RuntimeError("PyMuPDF fitz has neither open nor Document")
        
        page = doc[0]
        text = page.get_text()
        
        if not text:
            return ""
        
        # Convert to markdown format
        lines = [ln.strip() for ln in text.splitlines()]
        lines = [ln for ln in lines if ln]
        md_lines = []
        
        for ln in lines:
            # Simple heading heuristic: uppercase dominant and not too long
            if len(ln) <= 120 and sum(c.isupper() for c in ln) >= max(5, int(0.6 * len(ln.replace(" ", "")) or 0)):
                md_lines.append(f"## {ln}")
            else:
                md_lines.append(ln)
        
        return "\n".join(md_lines)
        
    finally:
        if doc is not None:
            doc.close()


def convert_single_pdf(args_tuple):
    """Convert a single PDF to MMD format."""
    arxiv_id, pdfs_dir, mmd_dir, progress_lock = args_tuple
    
    try:
        # Convert arxiv_id to filename
        pdf_filename = f"{arxiv_id.replace('.', '_')}.pdf"
        pdf_path = pdfs_dir / pdf_filename
        
        if not pdf_path.exists():
            return arxiv_id, False, "PDF not found"
        
        # Extract MMD content
        mmd_content = extract_mmd_from_pdf(pdf_path)
        if not mmd_content:
            return arxiv_id, False, "Failed to extract text"
        
        # Create output MMD filename
        mmd_filename = f"{arxiv_id.replace('.', '_')}.mmd"
        mmd_path = mmd_dir / mmd_filename
        
        # Write MMD file
        with open(mmd_path, 'w', encoding='utf-8') as f:
            f.write(mmd_content)
        
        return arxiv_id, True, f"Successfully converted ({len(mmd_content)} chars)"
        
    except Exception as e:
        return arxiv_id, False, f"Error: {e}"


def main():
    parser = argparse.ArgumentParser(description="Reconvert corrupted MMDs (130 bytes) to proper format")
    parser.add_argument("--pdfs-dir", default="/shared/hdd/arxiv_pdfs", 
                       help="Directory containing PDF files")
    parser.add_argument("--mmd-dir", default="/shared/hdd/arxiv_mmds", 
                       help="Directory containing MMD files")
    parser.add_argument("--workers", type=int, default=20, 
                       help="Number of parallel workers")
    parser.add_argument("--progress", action="store_true", help="Show progress")
    
    args = parser.parse_args()
    
    pdfs_dir = Path(args.pdfs_dir)
    mmd_dir = Path(args.mmd_dir)
    
    if not pdfs_dir.exists():
        print(f"Error: PDFs directory not found: {pdfs_dir}")
        return 1
    
    if not mmd_dir.exists():
        print(f"Error: MMD directory not found: {mmd_dir}")
        return 1
    
    # Find corrupted MMD files
    corrupted_ids = find_corrupted_mmds(mmd_dir)
    print(f"Found {len(corrupted_ids)} corrupted MMD files (130 bytes)")
    
    if not corrupted_ids:
        print("No corrupted MMD files found.")
        return 0
    
    # Start conversion
    start_time = time.time()
    print(f"Starting conversion with {args.workers} workers...")
    
    successful = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit conversion tasks
        future_to_id = {}
        for arxiv_id in corrupted_ids:
            future = executor.submit(
                convert_single_pdf, (arxiv_id, pdfs_dir, mmd_dir, None)
            )
            future_to_id[future] = arxiv_id
        
        # Process completed tasks
        for future in as_completed(future_to_id):
            arxiv_id, success, message = future.result()
            
            if success:
                successful += 1
                if args.progress:
                    print(f"✓ {arxiv_id}: {message}")
            else:
                failed += 1
                if args.progress:
                    print(f"✗ {arxiv_id}: {message}")
            
            # Progress update
            total_processed = successful + failed
            if args.progress and total_processed % 100 == 0:
                print(f"Progress: {total_processed}/{len(corrupted_ids)} ({total_processed/len(corrupted_ids)*100:.1f}%)")
    
    elapsed = time.time() - start_time
    
    # Summary
    print(f"\nConversion Summary:")
    print(f"  Total corrupted MMDs: {len(corrupted_ids)}")
    print(f"  Successfully converted: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Time elapsed: {elapsed:.1f}s")
    print(f"  MMD output directory: {mmd_dir}")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

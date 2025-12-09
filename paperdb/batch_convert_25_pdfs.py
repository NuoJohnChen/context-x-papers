#!/usr/bin/env python3
"""
Batch convert all PDFs starting with "25" from /shared/hdd/arxiv_pdfs to MMD format
using PyMuPDF and save to /shared/hdd/arxiv_mmds.

Usage:
  python3 batch_convert_25_pdfs.py [--workers 4] [--progress] [--limit 1000]
"""

import argparse
import os
import sys
import time
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Check PyMuPDF availability
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    print("Error: PyMuPDF (fitz) not installed. Please install it with: pip install PyMuPDF")
    sys.exit(1)


def extract_with_pymupdf(pdf_path: Path) -> tuple[str, float]:
    """Extract text from first page using PyMuPDF and return (text, time)."""
    t0 = time.time()
    doc = None
    try:
        if hasattr(fitz, "open"):
            doc = fitz.open(pdf_path)
        elif hasattr(fitz, "Document"):
            doc = fitz.Document(pdf_path)
        else:
            raise RuntimeError("PyMuPDF fitz has neither open nor Document")
        
        page = doc[0]
        text = page.get_text()
        return text, time.time() - t0
    finally:
        if doc is not None:
            doc.close()


def to_markdown(text: str) -> str:
    """Convert extracted text to markdown format."""
    if not text:
        return ""
    
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    md_lines = []
    
    for ln in lines:
        # Simple heading heuristic: uppercase dominant and not too long
        if len(ln) <= 120 and sum(c.isupper() for c in ln) >= max(5, int(0.6 * len(re.sub(r"[^A-Za-z]", "", ln)) or 0)):
            md_lines.append(f"## {ln}")
        else:
            md_lines.append(ln)
    
    return "\n".join(md_lines)


def convert_single_pdf(args_tuple):
    """Convert a single PDF to MMD format."""
    pdf_path, output_dir, progress_lock = args_tuple
    
    try:
        # Extract text
        text, elapsed = extract_with_pymupdf(pdf_path)
        
        if not text:
            return pdf_path.name, False, "Empty text extracted", elapsed
        
        # Convert to markdown
        md_content = to_markdown(text)
        
        # Create output filename
        base_name = pdf_path.stem
        output_path = output_dir / f"{base_name}.mmd"
        
        # Write MMD file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        
        return pdf_path.name, True, str(output_path), elapsed
        
    except Exception as e:
        return pdf_path.name, False, str(e), 0.0


def main():
    parser = argparse.ArgumentParser(description="Batch convert 25-prefix PDFs to MMD format")
    parser.add_argument("--pdf-dir", default="/shared/hdd/arxiv_pdfs", 
                       help="Directory containing PDF files")
    parser.add_argument("--output-dir", default="/shared/hdd/arxiv_mmds", 
                       help="Output directory for MMD files")
    parser.add_argument("--workers", type=int, default=4, 
                       help="Number of parallel workers")
    parser.add_argument("--progress", action="store_true", 
                       help="Show progress updates")
    parser.add_argument("--limit", type=int, default=0, 
                       help="Limit number of files to process (0 = no limit)")
    
    args = parser.parse_args()
    
    pdf_dir = Path(args.pdf_dir)
    output_dir = Path(args.output_dir)
    
    # Check directories
    if not pdf_dir.exists():
        print(f"Error: PDF directory not found: {pdf_dir}")
        return 1
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all PDFs starting with "25"
    print(f"Scanning for PDFs starting with '25' in {pdf_dir}...")
    pdf_files = []
    
    for pdf_file in pdf_dir.glob("*.pdf"):
        if pdf_file.stem.startswith("25"):
            pdf_files.append(pdf_file)
    
    if not pdf_files:
        print("No PDFs starting with '25' found.")
        return 0
    
    print(f"Found {len(pdf_files)} PDFs starting with '25'")
    
    # Apply limit if specified
    if args.limit > 0:
        pdf_files = pdf_files[:args.limit]
        print(f"Limited to {len(pdf_files)} files")
    
    # Check existing MMD files to avoid reprocessing
    existing_mmds = {p.stem for p in output_dir.glob("*.mmd")}
    new_pdfs = [pdf for pdf in pdf_files if pdf.stem not in existing_mmds]
    
    if not new_pdfs:
        print("All PDFs already have corresponding MMD files.")
        return 0
    
    print(f"Processing {len(new_pdfs)} new PDFs...")
    
    # Prepare conversion tasks
    progress_lock = Lock()
    conversion_tasks = [(pdf, output_dir, progress_lock) for pdf in new_pdfs]
    
    # Process files
    successful = 0
    failed = 0
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        future_to_pdf = {
            executor.submit(convert_single_pdf, task): task[0] 
            for task in conversion_tasks
        }
        
        # Process completed tasks
        for future in as_completed(future_to_pdf):
            pdf_name, success, result, elapsed = future.result()
            
            if success:
                successful += 1
                if args.progress:
                    with progress_lock:
                        print(f"✓ {pdf_name} -> {result} ({elapsed:.3f}s)")
            else:
                failed += 1
                with progress_lock:
                    print(f"✗ {pdf_name}: {result}")
            
            # Progress update
            total_processed = successful + failed
            if args.progress and total_processed % 100 == 0:
                elapsed_total = time.time() - start_time
                rate = total_processed / elapsed_total if elapsed_total > 0 else 0
                print(f"Progress: {total_processed}/{len(new_pdfs)} ({total_processed/len(new_pdfs)*100:.1f}%) - Rate: {rate:.1f} files/sec")
    
    # Summary
    total_time = time.time() - start_time
    print(f"\nConversion Summary:")
    print(f"  Total PDFs found: {len(pdf_files)}")
    print(f"  New PDFs processed: {len(new_pdfs)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Average rate: {len(new_pdfs)/total_time:.1f} files/sec")
    print(f"  Output directory: {output_dir}")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

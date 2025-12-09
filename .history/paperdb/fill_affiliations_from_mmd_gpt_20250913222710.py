#!/usr/bin/env python3
"""
Fill missing affiliation fields in addaffiliation.jsonl by reading corresponding .mmd
files and querying an OpenAI-compatible API concurrently.

- Input JSONL: /shared/hdd/nuochen/arxivdata/index/specter2_merged/addaffiliation.jsonl
- MMD dir:      /shared/hdd/arxiv_mmds
- API baseurl:  https://api.chatfire.cn/v1
- Model:        gpt-4o-mini
- Concurrency:  64 workers

Only process the first 10 entries whose "affiliation" is empty (for debugging).

Mapping from model output to fields:
- Output format from model: "Organization/Institution#College (N/A if none)#Country"
- JSON fields:
  - id, university, affiliation  -> Organization/Institution (same value)
  - department                   -> College (N/A if none)
  - affiliation_country         -> Country

Usage:
  python3 scripts/fill_affiliations_from_mmd_gpt.py --progress

API Key:
  - Provide via --api-key or environment variable CHATFIRE_API_KEY.
  - Avoid printing the key in logs.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random
import re

import requests
from requests.exceptions import RequestException, Timeout, ConnectionError as ReqConnectionError, HTTPError

INPUT_JSONL_DEFAULT = \
    "/shared/hdd/nuochen/arxivdata/index/specter2_merged/addaffiliation.jsonl"
MMD_DIR_DEFAULT = "/shared/hdd/arxiv_mmds"
API_BASEURL_DEFAULT = "https://api.chatfire.cn/v1"
MODEL_DEFAULT = "gpt-4o-mini"
WORKERS_DEFAULT = 256
MAX_RECORDS_DEFAULT = 999999

PROMPT_INSTRUCTION = (
    "You are extracting the first author affiliation info. Return EXACTLY three fields separated by a SINGLE '#' character, with NO extra spaces around '#', and NOTHING else.\n"
    "Format: Organization_or_Institution#College_or_Department_or_School_or_NA#Country_ISO3\n"
    "Rules:\n"
    "- Organization_or_Institution: Prefer the highest-level institution (usually a university). If multiple appear, choose the overarching university/institute/lab/team/company. If unknown, put N/A.\n"
    "- College_or_Department_or_School: Sub-unit within the organization (e.g., School/College/Department/Center). If none or unknown, put N/A.\n"
    "- Country_ISO3: Use ISO 3166 Alpha-3 code only (e.g., USA, CHN, GBR, DEU). If unknown, put N/A.\n"
    "- Output MUST have exactly two '#' separators and three fields.\n"
    "- Do NOT include labels, quotes, extra '#', punctuation, or explanations.\n"
    "- Examples (valid):\n"
    "  Massachusetts Institute of Technology#Computer Science and Artificial Intelligence Laboratory#USA\n"
    "  Shanghai Jiao Tong University#N/A#CHN\n"
    "  Johns Hopkins University#Space Telescope Science Institute#USA\n"
    "  DeepSeek-AI#N/A#CHN\n"
    "- Examples (invalid - DO NOT output):\n"
    "  Organization#College#Country#Extra\n"
    "  Org: MIT#Dept: EECS#Country: USA\n"
    "  Space Telescope Science Institute#Johns Hopkins University#USA\n"
    "  MIT#USA (missing middle field)\n"
)


def arxiv_id_to_mmd_paths(arxiv_id: str, mmd_dir: Path) -> list[Path]:
    """Generate candidate MMD paths for a given arxiv_id.
    2508.12345 -> 2508_12345.mmd (also try .pymupdf.mmd, .pdfplumber.mmd)
    """
    safe = arxiv_id.replace("/", "_").replace(".", "_")
    candidates = [
        mmd_dir / f"{safe}.mmd",
        mmd_dir / f"{safe}.pymupdf.mmd",
        mmd_dir / f"{safe}.pdfplumber.mmd",
    ]
    return candidates


def load_mmd_content(arxiv_id: str, mmd_dir: Path) -> Optional[str]:
    for p in arxiv_id_to_mmd_paths(arxiv_id, mmd_dir):
        if p.exists():
            try:
                return p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
    return None
EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# e.g., {chennuo,benyou}@gmail.com -> chennuo@gmail.com benyou@gmail.com
BRACE_EMAIL_REGEX = re.compile(r"\{\s*([^}]+?)\s*\}\s*@\s*([A-Za-z0-9.-]+\.[A-Za-z]{2,})")


def _expand_brace_emails(text: str) -> str:
    def _repl(m: re.Match) -> str:
        users_blob = m.group(1)
        domain = m.group(2)
        # split by comma/semicolon/whitespace
        users = [u.strip() for u in re.split(r"[\s,;]+", users_blob) if u.strip()]
        expanded = " ".join(f"{u}@{domain}" for u in users)
        return expanded
    return BRACE_EMAIL_REGEX.sub(_repl, text)


def extract_emails(text: str) -> list[str]:
    if not text:
        return []
    # First expand brace patterns into explicit emails
    text = _expand_brace_emails(text)
    # Deduplicate preserving order
    seen = set()
    emails = []
    for match in EMAIL_REGEX.finditer(text):
        e = match.group(0)
        if e not in seen:
            seen.add(e)
            emails.append(e)
    return emails


def extract_corresponding_email(mmd_text: str) -> Optional[str]:
    """Heuristics: prefer emails near 'corresponding/contact/email' cues, else first email."""
    if not mmd_text:
        return None
    # Quick pass for cue lines
    lines = mmd_text.splitlines()
    cue_keywords = ("corresponding", "correspondence", "contact", "email", "e-mail")
    best_email = None
    for ln in lines[:200]:  # focus on first page content window
        low = ln.lower()
        if any(k in low for k in cue_keywords):
            found = extract_emails(ln)
            if found:
                return found[0]
            # If line has no email, try adjacent context
    # Fallback: first email in whole text
    all_emails = extract_emails(mmd_text)
    if all_emails:
        best_email = all_emails[0]
    return best_email



def build_messages(title: str, authors: list, mmd_text: str) -> list:
    # Compact context for better extraction accuracy
    authors_str = ", ".join(authors) if authors else ""
    context = (
        f"Title: {title}\n"
        f"Authors: {authors_str}\n"
        f"--- First page (markdown):\n{mmd_text[:5000]}\n"
    )
    return [
        {"role": "system", "content": "You are an expert information extractor."},
        {"role": "user", "content": PROMPT_INSTRUCTION + "\n\n" + context},
    ]


def call_chat_api(baseurl: str, api_key: str, model: str, messages: list, timeout: float = 60.0) -> str:
    url = f"{baseurl}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 64,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    # OpenAI-compatible: choices[0].message.content
    return data["choices"][0]["message"]["content"].strip()


def call_chat_api_with_retry(baseurl: str, api_key: str, model: str, messages: list,
                             timeout: float = 30.0, max_retries: int = 3,
                             backoff_base: float = 1.0) -> str:
    """Call API with retries on timeouts, connection errors, 429 and 5xx responses."""
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            return call_chat_api(baseurl=baseurl, api_key=api_key, model=model, messages=messages, timeout=timeout)
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            # Retry on 429 and 5xx
            if status in (429, 500, 502, 503, 504):
                last_err = e
            else:
                raise
        except (Timeout, ReqConnectionError, RequestException) as e:
            last_err = e
        # backoff if we will retry
        if attempt < max_retries:
            # exponential backoff with jitter
            sleep_s = backoff_base * (2 ** attempt) + random.uniform(0, 0.5)
            time.sleep(sleep_s)
    # Exhausted retries
    if last_err:
        raise last_err
    raise RuntimeError("Unknown API error without exception")


def parse_affiliation_response(text: str) -> Optional[Tuple[str, str, str]]:
    """Parse 'Org#College#Country' into tuple. College may be 'N/A' or ''."""
    if not isinstance(text, str):
        return None
    parts = [p.strip() for p in text.split('#')]
    if len(parts) < 3:
        return None
    org = parts[0]
    college = parts[1]
    country = parts[2]
    if not org:
        return None
    return org, college, country


def process_record(record: dict, mmd_dir: Path, baseurl: str, api_key: str, model: str, progress: bool) -> Tuple[dict, bool, str]:
    """Return (updated_record, success, message)."""
    arxiv_id = record.get("arxiv_id", "")
    title = record.get("title", "")
    authors = record.get("authors", [])

    mmd_text = load_mmd_content(arxiv_id, mmd_dir)
    if not mmd_text:
        return record, False, "MMD not found"

    # Extract corresponding email from MMD first-page text
    corr_email = extract_corresponding_email(mmd_text) or ""

    messages = build_messages(title, authors, mmd_text)

    try:
        raw = call_chat_api_with_retry(baseurl=baseurl, api_key=api_key, model=model, messages=messages)
    except Exception as e:
        return record, False, f"API error: {e}"

    parsed = parse_affiliation_response(raw)
    if not parsed:
        return record, False, f"Parse failed: {raw[:200]}"

    org, college, country = parsed
    # Fill the fields
    record["id"] = org
    record["university"] = org
    record["affiliation"] = org
    record["department"] = college
    record["affiliation_country"] = country
    record["corresponding_email"] = corr_email

    return record, True, "ok"


def main():
    ap = argparse.ArgumentParser(description="Fill missing affiliation fields using MMD + GPT API (first 10)")
    ap.add_argument("--input", default=INPUT_JSONL_DEFAULT, help="Input JSONL file path")
    ap.add_argument("--output", default=INPUT_JSONL_DEFAULT, help="Output JSONL file path (in-place by default)")
    ap.add_argument("--mmd-dir", default=MMD_DIR_DEFAULT, help="Directory with .mmd files")
    ap.add_argument("--baseurl", default=API_BASEURL_DEFAULT, help="OpenAI-compatible API base URL")
    ap.add_argument("--model", default=MODEL_DEFAULT, help="Model name")
    ap.add_argument("--api-key", default='sk-8by4z5VkE9YTFoF16kJVaOwyLHLCKqAqjf3IrtxmOWXT5GaW', help="API key (or set CHATFIRE_API_KEY)")
    ap.add_argument("--workers", type=int, default=WORKERS_DEFAULT, help="Number of concurrent workers")
    ap.add_argument("--max-records", type=int, default=MAX_RECORDS_DEFAULT, help="Max records to process (with empty affiliation)")
    ap.add_argument("--all", action="store_true", help="Process ALL records with empty affiliation (ignores --max-records)")
    ap.add_argument("--progress", action="store_true", help="Show progress logs")
    ap.add_argument("--dry-run", action="store_true", help="Do not write output; just process and print summary")
    ap.add_argument("--year-prefixes", default="", help="Comma-separated arXiv ID year prefixes to include (e.g., '21,22,23,24')")
    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    mmd_dir = Path(args.mmd_dir)

    if not input_path.exists():
        print(f"Error: input JSONL not found: {input_path}", file=sys.stderr)
        return 1
    if not mmd_dir.exists():
        print(f"Error: MMD directory not found: {mmd_dir}", file=sys.stderr)
        return 1
    if not args.api_key:
        print("Error: missing API key. Provide --api-key or set CHATFIRE_API_KEY.", file=sys.stderr)
        return 1

    # Read all records first
    records = []
    with input_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(rec)

    # Build year prefix filter if provided
    year_prefixes: list[str] = []
    if args.year_prefixes:
        year_prefixes = [p.strip() for p in args.year_prefixes.split(',') if p.strip()]

    # Select records with empty affiliation and (optionally) matching year prefixes
    targets_idx = []
    for idx, rec in enumerate(records):
        if not rec.get("affiliation"):
            arxiv_id = str(rec.get("arxiv_id", ""))
            if year_prefixes:
                if not any(arxiv_id.startswith(p) for p in year_prefixes):
                    continue
            targets_idx.append(idx)
            # If not --all, limit by max_records
            if not args.all and len(targets_idx) >= args.max_records:
                break

    if not targets_idx:
        print("No records with empty affiliation found in the first scan.")
        return 0

    if args.progress:
        print(f"Found {len(targets_idx)} target records (max {args.max_records}). Starting processing with {args.workers} workers...")

    start_time = time.time()

    # Concurrent processing
    results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_idx = {}
        for idx in targets_idx:
            rec = records[idx]
            future = executor.submit(
                process_record, rec.copy(), mmd_dir, args.baseurl, args.api_key, args.model, args.progress
            )
            future_to_idx[future] = idx

        completed = 0
        success_cnt = 0
        fail_cnt = 0

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                updated, ok, msg = future.result()
            except Exception as e:
                ok, msg, updated = False, f"worker error: {e}", None
            completed += 1
            if ok and updated is not None:
                success_cnt += 1
                records[idx] = updated
                if args.progress:
                    org = updated.get('affiliation', '')
                    dept = updated.get('department', '')
                    country = updated.get('affiliation_country', '')
                    email = updated.get('corresponding_email', '')
                    print(f"[{completed}/{len(targets_idx)}] OK idx={idx} arxiv_id={updated.get('arxiv_id','')} -> {org} | {dept} | {country} | {email}")
            else:
                fail_cnt += 1
                if args.progress:
                    print(f"[{completed}/{len(targets_idx)}] FAIL idx={idx}: {msg}")

    if args.progress:
        elapsed = time.time() - start_time
        print(f"Completed. Success={success_cnt}, Fail={fail_cnt}, Time={elapsed:.1f}s")

    # If dry-run, skip writing
    if args.dry_run:
        print("Dry-run: skipping write-back.")
        return 0

    # If nothing changed, skip rewrite
    if success_cnt == 0:
        print("No successful updates. Skipping write-back.")
        return 0

    # Write back safely (temp then replace if same path) with large buffer and compact JSON
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open('w', encoding='utf-8', buffering=4*1024*1024) as f:
        dump = json.dumps
        for rec in records:
            f.write(dump(rec, ensure_ascii=False, separators=(',', ':')) + "\n")

    # Replace original
    tmp_path.replace(output_path)

    print(f"Updated file written: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

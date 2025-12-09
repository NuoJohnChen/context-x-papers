#!/usr/bin/env python3
"""
Build addaffiliation.jsonl by traversing Kaggle arxiv-metadata JSONL and extracting
records whose arxiv_id starts with specific year prefixes (e.g., '21', '22', '23', '24', '25').

Input:  /shared/hdd/nuochen/arxivdata/arxiv-metadata-oai-snapshot.json
Output: /shared/hdd/nuochen/arxivdata/index/specter2_merged/addaffiliation.jsonl

New schema per record:
{
  "id": "",
  "university": "",
  "title": <original title>,
  "abstract": <original abstract>,
  "text": "<title>[SEP]<abstract>",
  "authors": ["First Last", ...],  // parsed from original authors (comma-separated string)
  "categories": ["cs.CL", "cs.AI", ...], // split by whitespace
  "year": <int>,             // derive from update_date or versions[v1].created
  "arxiv_id": <original id>,
  "paperId": "",
  "authorId": "",
  "hIndex": "",
  "first_author": "First Last",   // first name in authors list
  "published_date": "YYYY-MM-DD", // from update_date (preferred) or v1 created date
  "affiliation": "",
  "department": "",
  "affiliation_country": ""
}

Usage:
  # Build for 2021-2024
  python3 scripts/build_addaffiliation_from_kaggle_25.py \
    --year-prefixes 21,22,23,24 \
    --input /shared/hdd/nuochen/arxivdata/arxiv-metadata-oai-snapshot.json \
    --output /shared/hdd/nuochen/arxivdata/index/specter2_merged/addaffiliation.jsonl \
    --limit 0 --progress
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from datetime import datetime


# Prefer structured authors when available; fallback to robust string split

def parse_authors_from_parsed(authors_parsed) -> list:
    if not isinstance(authors_parsed, list):
        return []
    names = []
    for item in authors_parsed:
        if not isinstance(item, (list, tuple)):
            continue
        # Expected format: [Last, First, Middle]
        last = str(item[0]).strip() if len(item) > 0 and item[0] is not None else ""
        first = str(item[1]).strip() if len(item) > 1 and item[1] is not None else ""
        middle = str(item[2]).strip() if len(item) > 2 and item[2] is not None else ""
        parts = [p for p in [first, middle, last] if p]
        if parts:
            names.append(" ".join(parts))
    return names


def parse_authors_string_robust(authors_str: str) -> list:
    """Robustly split authors like 'A, B and C & D' -> [A, B, C, D]."""
    if not isinstance(authors_str, str) or not authors_str.strip():
        return []
    s = authors_str.strip()
    # Normalize common conjunctions to commas
    # Handle Oxford comma patterns: ", and " -> ", "
    s = re.sub(r"\s*,\s*and\s+", ", ", s, flags=re.IGNORECASE)
    # Handle " and " and " & "
    s = re.sub(r"\s+and\s+", ", ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*&\s*", ", ", s)
    # Now split on commas
    parts = [p.strip() for p in s.split(',')]
    parts = [p for p in parts if p]
    return parts


def parse_authors_list(record: dict) -> list:
    # Prefer authors_parsed when present
    authors_parsed = record.get('authors_parsed')
    names = parse_authors_from_parsed(authors_parsed)
    if names:
        return names
    # Fallback to authors string
    authors_str = record.get('authors', '') or ''
    return parse_authors_string_robust(authors_str)


def parse_categories(categories_str: str) -> list:
    """Split whitespace-separated categories to list."""
    if not isinstance(categories_str, str):
        return []
    tokens = [t for t in categories_str.split() if t]
    return tokens


def parse_year(record: dict) -> int:
    """Derive year: prefer update_date, else versions[v1].created. Fallback to 0."""
    # Prefer update_date
    update_date = record.get('update_date')
    if isinstance(update_date, str) and len(update_date) >= 4:
        try:
            return int(update_date[:4])
        except Exception:
            pass
    # Else versions created for v1
    versions = record.get('versions')
    if isinstance(versions, list) and versions:
        for v in versions:
            if isinstance(v, dict) and v.get('version') == 'v1':
                created = v.get('created')  # e.g., 'Wed, 06 Aug 2025 15:59:18 GMT'
                if isinstance(created, str):
                    # Try parse year as last 4-digit number or via datetime
                    m = re.search(r'(19|20)\d{2}', created)
                    if m:
                        try:
                            return int(m.group(0))
                        except Exception:
                            pass
                    # Fallback datetime parsing
                    try:
                        dt = datetime.strptime(created, '%a, %d %b %Y %H:%M:%S %Z')
                        return dt.year
                    except Exception:
                        pass
                break
    return 0


def parse_published_date(record: dict) -> str:
    """Derive published_date as YYYY-MM-DD: prefer update_date, else v1 created date."""
    update_date = record.get('update_date')
    if isinstance(update_date, str) and re.match(r'^\d{4}-\d{2}-\d{2}$', update_date):
        return update_date
    # Else try versions[v1].created
    versions = record.get('versions')
    if isinstance(versions, list) and versions:
        for v in versions:
            if isinstance(v, dict) and v.get('version') == 'v1':
                created = v.get('created')
                if isinstance(created, str):
                    # Example: 'Wed, 06 Aug 2025 15:59:18 GMT' -> '2025-08-06'
                    try:
                        dt = datetime.strptime(created, '%a, %d %b %Y %H:%M:%S %Z')
                        return dt.strftime('%Y-%m-%d')
                    except Exception:
                        # Try relaxed parse: search for 'DD Mon YYYY'
                        m = re.search(r'(\d{1,2})\s+([A-Za-z]{3})\s+((?:19|20)\d{2})', created)
                        if m:
                            day = int(m.group(1))
                            mon_str = m.group(2)
                            year = int(m.group(3))
                            try:
                                dt = datetime.strptime(f'{day} {mon_str} {year}', '%d %b %Y')
                                return dt.strftime('%Y-%m-%d')
                            except Exception:
                                pass
                break
    return ""


def transform_record(record: dict) -> dict:
    title = record.get('title', '') or ''
    abstract = record.get('abstract', '') or ''
    categories_str = record.get('categories', '') or ''
    arxiv_id = record.get('id', '') or ''

    authors_list = parse_authors_list(record)

    transformed = {
        'id': '',
        'university': '',
        'title': title,
        'abstract': abstract,
        'text': f"{title}[SEP]{abstract}",
        'authors': authors_list,
        'categories': parse_categories(categories_str),
        'year': parse_year(record),
        'arxiv_id': arxiv_id,
        'paperId': '',
        'authorId': '',
        'hIndex': '',
        'first_author': authors_list[0] if authors_list else '',
        'published_date': parse_published_date(record),
        'affiliation': '',
        'department': '',
        'affiliation_country': ''
    }
    return transformed


def main():
    ap = argparse.ArgumentParser(description='Transform Kaggle arxiv-metadata to addaffiliation.jsonl (filter by year prefixes)')
    ap.add_argument('--input', default='/home/nuochen/.cache/kagglehub/datasets/Cornell-University/arxiv/versions/249/arxiv-metadata-oai-snapshot.json', help='Input JSONL file path')
    ap.add_argument('--output', default='/shared/hdd/nuochen/arxivdata/index/specter2_merged/addaffiliation.jsonl', help='Output JSONL file path')
    ap.add_argument('--limit', type=int, default=0, help='Limit number of records (0=no limit)')
    ap.add_argument('--progress', action='store_true', help='Show progress')
    ap.add_argument('--year-prefixes', default='25', help="Comma-separated arXiv ID year prefixes to include, e.g., '21,22,23,24,25'")
    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 1

    total = 0
    matched = 0
    written = 0

    # Parse year prefixes
    prefixes = [p.strip() for p in str(args.year_prefixes).split(',') if p.strip()]
    if not prefixes:
        prefixes = ['25']

    with input_path.open('r', encoding='utf-8') as fin, output_path.open('w', encoding='utf-8') as fout:
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"JSON error at line {line_num}: {e}", file=sys.stderr)
                continue

            arxiv_id = rec.get('id', '') or ''
            if not isinstance(arxiv_id, str):
                continue
            # Remove version suffix if present
            arxiv_id_norm = arxiv_id.split('v')[0]
            if not any(arxiv_id_norm.startswith(p) for p in prefixes):
                continue

            matched += 1
            transformed = transform_record(rec)
            fout.write(json.dumps(transformed, ensure_ascii=False) + '\n')
            written += 1

            if args.progress and written % 1000 == 0:
                print(f"Progress: processed={total}, matched={matched}, written={written}")

            if args.limit > 0 and written >= args.limit:
                break

    print(f"Done. Total lines: {total}, matched prefixes {prefixes}: {matched}, written: {written}")
    print(f"Output: {output_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())

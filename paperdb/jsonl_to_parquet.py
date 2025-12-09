#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # Skip malformed lines
                continue


def _join_list(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, list):
        return "; ".join(map(str, value))
    return str(value)

def _to_int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int,)):
            return int(value)
        s = str(value).strip()
        if s == "":
            return None
        return int(float(s))
    except Exception:
        return None


def _first_author_from(authors_value: Any) -> Optional[str]:
    if authors_value is None:
        return None
    # authors may be a list or a string
    if isinstance(authors_value, list):
        if len(authors_value) == 0:
            return None
        return str(authors_value[0]).strip() or None
    s = str(authors_value).strip()
    if s == "":
        return None
    # try common separators
    for sep in [";", ",", "|", " and "]:
        if sep in s:
            return s.split(sep)[0].strip() or None
    return s

def project_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    # Target fields:
    # {
    #   "id", "university", "title", "abstract", "text", "authors"(list),
    #   "categories"(list), "year"(int), "arxiv_id", "paperId", "authorId", "hIndex",
    #   "first_author", "published_date", "affiliation", "department", "affiliation_country", "corresponding_email"
    # }

    def ensure_list_of_str(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(x).strip() for x in value if str(x).strip() != ""]
        s = str(value).strip()
        if s == "":
            return []
        # Try common separators
        for sep in [";", ",", "|", " and "]:
            if sep in s:
                return [p.strip() for p in s.split(sep) if p.strip()]
        return [s]

    title_val = rec.get("title", "")
    abstract_val = rec.get("abstract", "")
    text_val = rec.get("text") or (f"{title_val}[SEP]{abstract_val}" if (title_val or abstract_val) else "")

    # authors/cateogries -> list[str]
    authors_raw = rec.get("authors") if rec.get("authors") is not None else rec.get("author")
    authors_list = ensure_list_of_str(authors_raw)
    categories_list = ensure_list_of_str(rec.get("categories"))

    # derive first_author
    first_author_val = rec.get("first_author", "")
    if first_author_val is None or str(first_author_val).strip() == "":
        first_author_guess = _first_author_from(authors_list)
        first_author_val = first_author_guess or ""

    # derive year (prefer existing year)
    year_val = _to_int_or_none(rec.get("year"))
    if year_val is None:
        # fallback from update_date or versions[0].created
        update_date = rec.get("update_date") or rec.get("published_date") or ""
        if isinstance(update_date, str) and len(update_date) >= 4:
            try:
                year_val = int(update_date[:4])
            except Exception:
                year_val = None
        if year_val is None:
            versions = rec.get("versions") or []
            if isinstance(versions, list) and versions:
                created = versions[0].get("created") if isinstance(versions[0], dict) else None
                if isinstance(created, str) and len(created) >= 4:
                    try:
                        year_val = int(created[:4])
                    except Exception:
                        year_val = None

    # published_date prefer update_date, else versions[0].created, else existing published_date
    published_date_val = None
    if isinstance(rec.get("update_date"), str) and rec.get("update_date").strip():
        published_date_val = rec.get("update_date").strip()
    elif isinstance(rec.get("published_date"), str) and rec.get("published_date").strip():
        published_date_val = rec.get("published_date").strip()
    else:
        versions = rec.get("versions") or []
        if isinstance(versions, list) and versions:
            created = versions[0].get("created") if isinstance(versions[0], dict) else None
            if isinstance(created, str) and created.strip():
                published_date_val = created.strip()
    if isinstance(published_date_val, str) and len(published_date_val) >= 10:
        published_date_val = published_date_val[:10]

    return {
        "id": str(rec.get("id", "")) if rec.get("id") is not None else "",
        "university": str(rec.get("university", "")) if rec.get("university") is not None else "",
        "title": title_val,
        "abstract": abstract_val,
        "text": text_val,
        "authors": authors_list,
        "categories": categories_list,
        "year": year_val,
        "arxiv_id": str(rec.get("arxiv_id")) if rec.get("arxiv_id") is not None else None,
        "paperId": str(rec.get("paperId", "")) if rec.get("paperId") is not None else "",
        "authorId": str(rec.get("authorId", "")) if rec.get("authorId") is not None else "",
        "hIndex": str(rec.get("hIndex", "")) if rec.get("hIndex") is not None else "",
        "first_author": first_author_val,
        "published_date": published_date_val or "",
        "affiliation": str(rec.get("affiliation", "")) if rec.get("affiliation") is not None else "",
        "department": str(rec.get("department", "")) if rec.get("department") is not None else "",
        "affiliation_country": str(rec.get("affiliation_country", "")) if rec.get("affiliation_country") is not None else "",
        "corresponding_email": str(rec.get("corresponding_email", "")) if rec.get("corresponding_email") is not None else "",
    }


def write_parquet_incremental(
    input_jsonl: Path,
    output_parquet: Path,
    chunk_size: int = 5000,
    schema: Optional[pa.schema] = None,
):
    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    writer: Optional[pq.ParquetWriter] = None
    buffer: List[Dict[str, Any]] = []

    def flush(buf: List[Dict[str, Any]], wr: Optional[pq.ParquetWriter]):
        if not buf:
            return wr
        df = pd.DataFrame(buf)
        table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
        if wr is None:
            wr = pq.ParquetWriter(
                output_parquet,
                table.schema,
                compression="snappy",
                use_dictionary=True,
            )
        wr.write_table(table)
        buf.clear()
        return wr

    for rec in iter_jsonl(input_jsonl):
        buffer.append(project_record(rec))
        if len(buffer) >= chunk_size:
            writer = flush(buffer, writer)

    writer = flush(buffer, writer)
    if writer is not None:
        writer.close()


def main():
    if len(sys.argv) < 3:
        print(
            "Usage: jsonl_to_parquet.py <input.jsonl> <output.parquet> [chunk_size]",
            file=sys.stderr,
        )
        sys.exit(1)

    input_jsonl = Path(sys.argv[1])
    output_parquet = Path(sys.argv[2])
    chunk_size = int(sys.argv[3]) if len(sys.argv) >= 4 else 5000

    # Predefine schema to keep types stable (and field order)
    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("university", pa.string()),
            pa.field("title", pa.string()),
            pa.field("abstract", pa.string()),
            pa.field("text", pa.string()),
            pa.field("authors", pa.list_(pa.string())),
            pa.field("categories", pa.list_(pa.string())),
            pa.field("year", pa.int64()),
            pa.field("arxiv_id", pa.string()),
            pa.field("paperId", pa.string()),
            pa.field("authorId", pa.string()),
            pa.field("hIndex", pa.string()),
            pa.field("first_author", pa.string()),
            pa.field("published_date", pa.string()),
            pa.field("affiliation", pa.string()),
            pa.field("department", pa.string()),
            pa.field("affiliation_country", pa.string()),
            pa.field("corresponding_email", pa.string()),
        ]
    )

    write_parquet_incremental(input_jsonl, output_parquet, chunk_size, schema)
    print(f"Wrote parquet: {output_parquet}")


if __name__ == "__main__":
    main()



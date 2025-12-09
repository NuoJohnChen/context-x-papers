import os
import sys
import ast
import json
import glob
import argparse
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import numpy as np
from numpy.lib.format import open_memmap
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
import logging
from datetime import datetime


def parse_papers_field(papers_field: object) -> List[Dict[str, str]]:
    """
    Parse the "papers" column value which is typically a stringified Python list
    of dictionaries from Semantic Scholar. Returns a list of dicts with keys
    'title' and 'abstract' (missing values become empty strings).
    """
    if not isinstance(papers_field, str):
        return []
    text = papers_field.strip()
    if not text:
        return []

    # Make evaluation more robust to line breaks and stray whitespace
    candidate = text.replace("\n", " ").replace("\r", " ")
    try:
        obj = ast.literal_eval(candidate)
    except Exception:
        return []

    results: List[Dict[str, str]] = []
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                title = item.get("title") or ""
                abstract = item.get("abstract") or ""
                paper_id = item.get("paperId") or ""
                if title or abstract:
                    results.append({"title": title, "abstract": abstract, "paperId": paper_id})
    return results


def build_texts_from_row(row: pd.Series, tokenizer: AutoTokenizer, csv_row_index: int) -> List[Tuple[str, Dict[str, object]]]:
    texts: List[Tuple[str, Dict[str, object]]] = []
    sep = tokenizer.sep_token or " [SEP] "

    # Preferred: expand the 'papers' list
    if "papers" in row and isinstance(row["papers"], str):
        inner_idx = 0
        for paper in parse_papers_field(row["papers"]):
            t = (paper.get("title") or "") + sep + (paper.get("abstract") or "")
            if t.strip():
                meta = {
                    "csv_row_index": csv_row_index,
                    "inner_paper_index": inner_idx,
                    "paperId": paper.get("paperId") or "",
                    "title": paper.get("title") or "",
                    "abstract": paper.get("abstract") or "",
                    "text": t,
                }
                texts.append((t, meta))
                inner_idx += 1

    # Fallback: direct title/abstract columns if present
    if not texts:
        title_keys = ["title", "paper_title"]
        abs_keys = ["abstract", "paper_abstract"]
        title_val: Optional[str] = None
        abs_val: Optional[str] = None
        for k in title_keys:
            if k in row:
                title_val = row[k]
                break
        for k in abs_keys:
            if k in row:
                abs_val = row[k]
                break
        if (title_val or abs_val):
            t = (title_val or "") + sep + (abs_val or "")
            meta = {
                "csv_row_index": csv_row_index,
                "inner_paper_index": 0,
                "paperId": "",
                "title": title_val or "",
                "abstract": abs_val or "",
                "text": t,
            }
            texts.append((t, meta))

    return texts


def iter_texts_from_jsonl(
    jsonl_path: str,
    tokenizer: AutoTokenizer,
    limit_rows: Optional[int] = None,
) -> Iterable[Tuple[str, Dict[str, object]]]:
    """
    Stream texts built from JSONL rows. Each line should contain a JSON object with 'text' field.
    """
    count = 0
    with open(jsonl_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_num, line in enumerate(f, start=1):
            if limit_rows is not None and count >= limit_rows:
                break
            
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                text = data.get("text", "")
                if not text.strip():
                    continue
                    
                meta = {
                    "line_number": line_num,
                    "id": data.get("id", ""),
                    "title": data.get("title", ""),
                    "abstract": data.get("abstract", ""),
                    "text": text,
                    "arxiv_id": data.get("arxiv_id", ""),
                    "authors": data.get("authors", []),
                    "categories": data.get("categories", []),
                    "year": data.get("year", ""),
                }
                yield (text, meta)
                count += 1
                
            except json.JSONDecodeError as e:
                logging.warning(f"Skipping invalid JSON line {line_num}: {e}")
                continue
            except Exception as e:
                logging.warning(f"Error processing line {line_num}: {e}")
                continue


def iter_texts_from_csv(
    csv_path: str,
    tokenizer: AutoTokenizer,
    limit_rows: Optional[int] = None,
    chunksize: Optional[int] = None,
) -> Iterable[Tuple[str, Dict[str, object]]]:
    """
    Stream texts built from CSV rows. Supports chunked reading to limit RAM.
    """
    usecols: Optional[List[str]] = None
    # Try to restrict read columns for speed
    try:
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            header_line = f.readline().strip()
        headers = header_line.split(",") if header_line else []
        for candidate in ["papers", "title", "paper_title", "abstract", "paper_abstract"]:
            if candidate in headers:
                usecols = list({*(usecols or []), candidate})
    except Exception:
        usecols = None

    reader_kwargs = dict(
        usecols=usecols,
        encoding="utf-8",
        on_bad_lines="skip",
    )

    def stream_rows(engine: str | None):
        if engine:
            local_kwargs = dict(reader_kwargs, engine=engine)
        else:
            local_kwargs = dict(reader_kwargs)

        if chunksize and (limit_rows is None or limit_rows > chunksize):
            read_rows = 0
            for chunk in pd.read_csv(csv_path, chunksize=chunksize, **local_kwargs):
                if limit_rows is not None and read_rows >= limit_rows:
                    break
                remaining = None if limit_rows is None else max(0, limit_rows - read_rows)
                if remaining is not None and len(chunk) > remaining:
                    chunk = chunk.iloc[:remaining]
                local_count = 0
                for _, row in chunk.iterrows():
                    csv_row_index = read_rows + local_count
                    for pair in build_texts_from_row(row, tokenizer, csv_row_index):
                        yield pair
                    local_count += 1
                read_rows += len(chunk)
        else:
            df = pd.read_csv(csv_path, nrows=limit_rows, **local_kwargs)
            for i, (_, row) in enumerate(df.iterrows()):
                for pair in build_texts_from_row(row, tokenizer, i):
                    yield pair

    # Try fast C engine first (no field_size_limit), then fallback to python with raised limit
    try:
        yield from stream_rows(engine=None)
        return
    except Exception:
        pass
    try:
        import csv as _csv
        try:
            _csv.field_size_limit(sys.maxsize)
        except OverflowError:
            _csv.field_size_limit(10 ** 9)
        yield from stream_rows(engine="python")
        return
    except Exception as e:
        raise e


@torch.no_grad()
def embed_batch(texts: List[str], tokenizer: AutoTokenizer, model: AutoAdapterModel, device: torch.device, batch_size: int) -> torch.Tensor:
    """Embed a small batch of texts and return a tensor [B, hidden]."""
    if not texts:
        hidden = getattr(model.config, "hidden_size", 768)
        return torch.zeros((0, hidden))
    chunk_embs: List[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False,
            max_length=512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :].detach().cpu()
        chunk_embs.append(emb)
    return torch.cat(chunk_embs, dim=0)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/shared/hdd/nuochen/arxivdata/index/specter2_merged/final_combined_deduplicated.jsonl", help="Input file path")
    parser.add_argument("--output", default="/shared/hdd/nuochen/arxivdata/embeddings/specter2", help="Output directory")
    parser.add_argument("--limit_rows", type=int, default=None, help="Max rows to read (for debugging)")
    parser.add_argument("--batch_size", type=int, default=16, help="Embedding batch size")
    parser.add_argument("--accumulate", type=int, default=2048, help="Number of texts to buffer before sending to model")
    parser.add_argument("--gpu_id", type=int, default=1, help="GPU ID to use (default 1, i.e., second card)")
    parser.add_argument("--log_dir", default=None, help="Log directory (print to console if not set)")
    args = parser.parse_args()

    # logging setup
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    log_file_path = None
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file_path = os.path.join(args.log_dir, f"run_{ts}.log")
        handlers.append(logging.FileHandler(log_file_path, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )

    # Configure selected GPU
    if torch.cuda.is_available() and args.gpu_id < torch.cuda.device_count():
        device = torch.device(f"cuda:{args.gpu_id}")
        torch.cuda.set_device(args.gpu_id)
        logging.info(f"Using GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")
    else:
        device = torch.device("cpu")
        logging.warning(f"GPU {args.gpu_id} unavailable, using CPU")
    
    logging.info(f"device: {device}")

    logging.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True)
    model.to(device)
    model.eval()

    ensure_dir(args.output)

    # Process single input file
    input_path = args.input
    logging.info("Processing input file: %s", input_path)
    if log_file_path:
        logging.info("日志文件: %s", log_file_path)
    
    try:
        base = os.path.basename(input_path)
        stem = os.path.splitext(base)[0]
        out_tensor = os.path.join(args.output, f"{stem}.pt")
        out_meta_summary = os.path.join(args.output, f"{stem}.json")
        out_meta_jsonl = os.path.join(args.output, f"{stem}.jsonl")
        out_npy = os.path.join(args.output, f"{stem}.npy")
        shards_dir = os.path.join(args.output, "shards", stem)
        ensure_dir(shards_dir)

        # Resume state: count already processed items from jsonl (lines)
        processed_count = 0
        if os.path.exists(out_meta_jsonl):
            try:
                with open(out_meta_jsonl, "r", encoding="utf-8") as jf:
                    for processed_count, _ in enumerate(jf, start=1):
                        pass
            except Exception:
                processed_count = 0
        logging.info("Existing progress: %d items (resuming from here)", processed_count)

        # Stream texts and metas; embed by chunks; write JSONL progressively (append)
        texts_and_metas = iter_texts_from_jsonl(
            input_path,
            tokenizer,
            limit_rows=args.limit_rows,
        )
        buffer_texts: List[str] = []
        buffer_metas: List[Dict[str, object]] = []
        total_items = processed_count
        seen = 0

        # prepare jsonl file
        jf_mode = "a" if processed_count > 0 else "w"
        with open(out_meta_jsonl, jf_mode, encoding="utf-8") as jf:
            for text, meta_item in texts_and_metas:
                # skip already processed items to resume
                if seen < processed_count:
                    seen += 1
                    continue
                buffer_texts.append(text)
                buffer_metas.append(meta_item)
                if len(buffer_texts) >= args.accumulate:
                    logging.info("flush buffer: %d items", len(buffer_texts))
                    emb = embed_batch(buffer_texts, tokenizer, model, device, args.batch_size)
                    # save shard
                    start_idx = total_items
                    end_idx = total_items + len(buffer_texts) - 1
                    shard_path = os.path.join(shards_dir, f"{stem}.shard_{start_idx}_{end_idx}.npy")
                    np.save(shard_path, emb.numpy())
                    logging.info("Saved shard: %s", shard_path)
                    # write metas in the same order
                    for m in buffer_metas:
                        jf.write(json.dumps(m, ensure_ascii=False) + "\n")
                    total_items += len(buffer_texts)
                    buffer_texts.clear()
                    buffer_metas.clear()
            # flush remaining
            if buffer_texts:
                logging.info("flush last buffer: %d items", len(buffer_texts))
                emb = embed_batch(buffer_texts, tokenizer, model, device, args.batch_size)
                start_idx = total_items
                end_idx = total_items + len(buffer_texts) - 1
                shard_path = os.path.join(shards_dir, f"{stem}.shard_{start_idx}_{end_idx}.npy")
                np.save(shard_path, emb.numpy())
                logging.info("Saved shard: %s", shard_path)
                for m in buffer_metas:
                    jf.write(json.dumps(m, ensure_ascii=False) + "\n")
                total_items += len(buffer_texts)
                buffer_texts.clear()
                buffer_metas.clear()

        # Merge shards into final npy (resume-safe)
        logging.info("Start merging shards, total items: %d", total_items)
        shard_files = sorted(
            [os.path.join(shards_dir, f) for f in os.listdir(shards_dir) if f.startswith(stem + ".shard_") and f.endswith(".npy")],
            key=lambda p: (int(os.path.basename(p).split("_")[-2]), int(os.path.basename(p).split("_")[-1].split(".")[0]))
        )
        
        # Check if shard files exist
        if not shard_files:
            logging.warning("No shard files found, creating empty embedding file")
            np.save(out_npy, np.zeros((0, getattr(model.config, "hidden_size", 768)), dtype=np.float32))
        else:
            # Compute total rows and hidden size
            total_rows = 0
            hidden_size = None
            valid_shards = []
            
            for sf in shard_files:
                try:
                    arr = np.load(sf, mmap_mode='r')
                    if arr.shape[0] > 0:  # Only process non-empty shards
                        valid_shards.append((sf, arr.shape))
                        total_rows += arr.shape[0]
                        if hidden_size is None:
                            hidden_size = arr.shape[1]
                        logging.info("Shard %s: shape %s", os.path.basename(sf), arr.shape)
                    else:
                        logging.warning("Skipping empty shard: %s", os.path.basename(sf))
                except Exception as e:
                    logging.error("Failed to read shard %s: %s", os.path.basename(sf), repr(e))
            
            if total_rows == 0:
                logging.warning("All shards are empty, creating empty embedding file")
                np.save(out_npy, np.zeros((0, getattr(model.config, "hidden_size", 768)), dtype=np.float32))
            else:
                logging.info("Valid shards: %d, total rows: %d, hidden size: %d", len(valid_shards), total_rows, hidden_size)
                
                # Create final file
                mm = open_memmap(out_npy, mode='w+', dtype=np.float32, shape=(total_rows, hidden_size))
                cursor = 0
                
                for sf, shp in valid_shards:
                    try:
                        arr = np.load(sf, mmap_mode='r')
                        if arr.shape[0] > 0:  # Double-check
                            mm[cursor:cursor+arr.shape[0], :] = arr
                            cursor += arr.shape[0]
                            logging.info("Merged shard %s: position %d:%d", os.path.basename(sf), cursor-arr.shape[0], cursor)
                    except Exception as e:
                        logging.error("Failed to merge shard %s: %s", os.path.basename(sf), repr(e))
                
                del mm  # Flush to disk
                logging.info("Finished merge to: %s, actual rows: %d", out_npy, cursor)

        # Also save .pt for compatibility (loads npy then saves tensor)
        final_arr = np.load(out_npy, mmap_mode=None)
        embeddings = torch.from_numpy(final_arr.copy())  # ensure owning memory
        torch.save(embeddings, out_tensor)

        meta_summary = {
            "input_path": input_path,
            "num_embeddings": int(embeddings.shape[0]),
            "hidden_size": int(embeddings.shape[1]) if embeddings.numel() > 0 else 0,
            "model": "allenai/specter2_base",
            "adapter": "allenai/specter2:proximity",
            "jsonl": out_meta_jsonl,
            "npy": out_npy,
        }
        with open(out_meta_summary, "w", encoding="utf-8") as f:
            json.dump(meta_summary, f, ensure_ascii=False, indent=2)
        logging.info("Completed: %s", out_tensor)
        logging.info("Count check: embeddings=%d, jsonl_lines(expected)=%d", embeddings.shape[0], total_items)
        if embeddings.shape[0] != total_items:
            logging.warning("Count mismatch: tensor=%d vs jsonl_accum=%d", embeddings.shape[0], total_items)
    except Exception as e:
        logging.exception("Failed: %s", input_path)


if __name__ == "__main__":
    main()



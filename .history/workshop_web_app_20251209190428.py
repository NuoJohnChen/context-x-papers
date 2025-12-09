#!/usr/bin/env python3
"""
Workshop Recommendation Web Application
"""

from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
try:
    from sentence_transformers import CrossEncoder  # optional fallback reranker
except Exception:
    CrossEncoder = None
from sklearn.metrics.pairwise import cosine_similarity
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass
import os
import time
import json
import re
import threading
import uuid
import http.client
import io
from datetime import datetime
import zipfile
import requests
import paper2poster_app
import shutil
from collections import Counter
from hdbscan import HDBSCAN
import umap
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from concurrent.futures import ThreadPoolExecutor

try:
    import lancedb  # type: ignore
except Exception:
    lancedb = None

app = Flask(__name__)

# Configuration for remote deployment
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour session timeout

# Enable CORS for remote access
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

def _get_client_ip(req) -> str:
    """Best-effort client IP extraction behind proxies.
    Checks common headers then falls back to Flask's remote_addr.
    """
    try:
        # X-Forwarded-For may contain a list like: client, proxy1, proxy2
        xff = (req.headers.get('X-Forwarded-For') or req.headers.get('X-Forwarded-for') or '').strip()
        if xff:
            first = xff.split(',')[0].strip()
            if first:
                return first
        xri = (req.headers.get('X-Real-IP') or '').strip()
        if xri:
            return xri
    except Exception:
        pass
    return req.remote_addr

def convert_to_json_serializable(obj):
    """Convert Python objects to JSON-serializable values"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return ""
    else:
        return obj

# class WorkshopRecommender:
#     def __init__(self, 
#                  data_path: str = "/home/nuochen/workshop_reco/index/specter2_merged/embedding_atlas_export_20250829_101535/data/dataset_with_published_with_published_date_update_with_published_date_update.parquet",
#                  embeddings_path: str = "/home/nuochen/workshop_reco/outputs/minilm_filtered/embeddings.npy"):
# class WorkshopRecommender:
#     def __init__(self, 
#                  data_path: str = "/home/nuochen/arxiv_merged_new.parquet",
#                  embeddings_path: str = "/home/nuochen/arxiv_merged.npy"):
class WorkshopRecommender:
    def __init__(self, 
                 data_path: str = "/home/nuochen/dataset.parquet",
                 embeddings_path: str = "/home/nuochen/embeddings.npy"):
        """Initialize the recommender"""
        self.data_path = data_path
        self.embeddings_path = embeddings_path
        
        # Initialize SPECTER2 model
        print("Loading SPECTER2 model...")
        try:
            #self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            self.model = SentenceTransformer("/home/nuochen/models/specter2_base")#'allenai/specter2_base')
            # mixedbread-ai/mxbai-embed-large-v1
            # self.model = SentenceTransformer("/home/nuochen/models/mxbai-embed-large-v1")
            print("✓ Model loaded.")
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            return
        
        # Load data
        print("Loading paper data...")
        try:
            print(f"  Reading {data_path}...")
            self.df = pd.read_parquet(data_path)
            print(f"✓ Loaded {len(self.df)} papers")
            print(f"  Data size: {self.df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        except Exception as e:
            print(f"✗ Failed to load data: {e}")
            return
        
        # Preprocess the year column to create a numeric year for filtering
        try:
            if 'year' in self.df.columns:
                # Convert to numeric first
                year_numeric = pd.to_numeric(self.df['year'], errors='coerce')

                # For non-numeric values, try to extract a 4-digit year (e.g., 2025-08-23 or 'Jan 2025')
                need_parse = year_numeric.isna()
                if need_parse.any():
                    year_str = self.df.loc[need_parse, 'year'].astype(str)
                    extracted = year_str.str.extract(r'(\d{4})', expand=False)
                    extracted_numeric = pd.to_numeric(extracted, errors='coerce')

                    # Merge back
                    year_numeric.loc[need_parse] = extracted_numeric

                # Filter unreasonable years
                year_numeric = year_numeric.where((year_numeric >= 1900) & (year_numeric <= 2100))

                self.df['year_numeric'] = year_numeric
            else:
                # If there is no year column, create an empty column
                self.df['year_numeric'] = pd.Series([np.nan] * len(self.df))
        except Exception as e:
            print(f"⚠️ Failed to preprocess year; skipping year filter: {e}")
            self.df['year_numeric'] = pd.Series([np.nan] * len(self.df))
        
        # Parse dates (aim to include month/day where possible)
        try:
            candidate_columns = [
                'published_date'
            ]
            date_series = pd.Series([pd.NaT] * len(self.df))
            for col in candidate_columns:
                if col in self.df.columns:
                    parsed = pd.to_datetime(self.df[col], errors='coerce', utc=True)
                    date_series = date_series.fillna(parsed)
            if date_series.isna().all() and 'year' in self.df.columns:
                year_str_series = self.df['year'].astype(str)
                # ISO-style YYYY-MM-DD / YYYY/MM/DD
                match_iso = year_str_series.str.extract(r'((?:19|20)\\d{2}[-/](?:0?[1-9]|1[0-2])[-/](?:0?[1-9]|[12]\\d|3[01]))', expand=False)
                parsed_iso = pd.to_datetime(match_iso, errors='coerce', utc=True)
                date_series = date_series.fillna(parsed_iso)
                # English month strings such as "Jan 2025"
                match_mon = year_str_series.str.extract(r'([A-Za-z]{3,9}\\s+(?:19|20)\\d{2})', expand=False)
                parsed_mon = pd.to_datetime(match_mon, errors='coerce', utc=True)
                date_series = date_series.fillna(parsed_mon)
            # Drop timezone to keep local date
            self.df['date_parsed'] = date_series.dt.tz_localize(None)
        except Exception as e:
            print(f"⚠️ Failed to preprocess dates: {e}")
            self.df['date_parsed'] = pd.Series([pd.NaT] * len(self.df))
        
        # Load precomputed embeddings
        print("Loading precomputed embeddings...")
        try:
            if os.path.exists(embeddings_path):
                print(f"  Reading {embeddings_path}...")
                self.embeddings = np.load(embeddings_path)#,mmap_mode='r'
                print(f"✓ Loaded embeddings: {self.embeddings.shape}")
                print(f"  Embeddings size: {self.embeddings.nbytes / 1024**2:.1f} MB")
                self.has_embeddings = True
            else:
                print(f"⚠️  Embeddings file not found: {embeddings_path}")
                self.has_embeddings = False
        except Exception as e:
            print(f"✗ Failed to load embeddings: {e}")
            self.has_embeddings = False
        
        # If data count and embedding count differ, use the smaller value
        if self.has_embeddings and hasattr(self, 'df'):
            self.num_items = int(min(len(self.df), self.embeddings.shape[0]))
            if len(self.df) != self.embeddings.shape[0]:
                print(f"⚠️ Data and embeddings count mismatch: df={len(self.df)}, emb={self.embeddings.shape[0]} -> using first {self.num_items}")
        else:
            self.num_items = 0

        self.lance_db = None
        self.lance_table = None
        self._lance_enabled = os.getenv('ENABLE_LANCE_DB', '1').lower() in ('1', 'true', 'yes', 'on')
        self._lance_uri = os.getenv('LANCE_DB_URI', os.path.join(os.getcwd(), 'lancedb_store'))
        self._lance_table_name = os.getenv('LANCE_TABLE_NAME', 'papers')
        self._lance_search_k = int(os.getenv('LANCE_SEARCH_K', '800'))
        self._lance_build_chunk = int(os.getenv('LANCE_BUILD_CHUNK', '2000'))
        if self._lance_enabled:
            self._init_lancedb()
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get the embedding vector for a given text"""
        return self.model.encode(text, convert_to_tensor=False)
    
    def recommend_papers(self, workshop_description: str, top_k: int = 20, date_min: str | None = None, date_max: str | None = None, countries: list[str] | None = None, categories: list[str] | None = None, min_similarity: float | None = None):
        """Recommend papers for the given workshop description"""
        if not self.has_embeddings or self.num_items <= 0:
            return []
        
        debug_timing = os.getenv('RECO_TIMING_DEBUG', '0') == '1'
        stage_start = time.time()
        if debug_timing:
            print(f"[rec-debug] Stage1 recall start | len_df={len(getattr(self, 'df', []))} | num_items={self.num_items} | desc_len={len(workshop_description or '')} | top_k={top_k}")
        
        encode_start = time.time()
        workshop_embedding = self.get_embedding(workshop_description)
        encode_end = time.time()
        if debug_timing:
            print(f"[rec-debug] encoder time={encode_end - encode_start:.3f}s device={getattr(self.model, 'device', 'cpu')}")
        
        if workshop_embedding.shape[0] != self.embeddings.shape[1]:
            return []
        
        query_vec = np.asarray(workshop_embedding, dtype=np.float32)
        candidate_indices = np.array([], dtype=np.int64)
        candidate_scores = np.array([], dtype=np.float32)
        use_lance = self.lance_table is not None
        
        if use_lance:
            candidate_indices, candidate_scores = self._lance_search(
                query_vec,
                top_k,
                date_min,
                date_max,
                countries,
                categories,
                debug_timing,
            )
        
        if candidate_indices.size == 0:
            candidate_indices, candidate_scores = self._exact_search(
                query_vec,
                top_k,
                date_min,
                date_max,
                countries,
                categories,
                debug_timing,
            )
        
        if candidate_indices.size == 0:
            return []
        
        if min_similarity is not None:
            try:
                thr = float(min_similarity)
                mask_sim = candidate_scores >= thr
                candidate_indices = candidate_indices[mask_sim]
                candidate_scores = candidate_scores[mask_sim]
            except Exception:
                pass
        
        limit = min(top_k, candidate_indices.size)
        candidate_indices = candidate_indices[:limit]
        candidate_scores = candidate_scores[:limit]
        
        recommendations = []
        for idx, score in zip(candidate_indices, candidate_scores):
            if idx < len(self.df):
                paper = self.df.iloc[idx]
                date_display = ""
                try:
                    date_val = paper.get('date_parsed', None)
                    if pd.notna(date_val):
                        date_display = pd.to_datetime(date_val).strftime('%Y-%m-%d')
                except Exception:
                    date_display = ""
                arxiv_id_val = str(convert_to_json_serializable(paper.get("arxiv_id", "")))
                year_month = parse_arxiv_year_month(arxiv_id_val)
                rec = {
                    "title": str(convert_to_json_serializable(paper.get("title", "Unknown title"))),
                    "authors": str(convert_to_json_serializable(paper.get("authors", paper.get("author", "Unknown author")))),
                    "abstract": str(convert_to_json_serializable(paper.get("abstract", "No abstract"))),
                    "similarity": float(score),
                    "arxiv_id": arxiv_id_val,
                    "year": str(convert_to_json_serializable(paper.get("year", ""))),
                    "year_month": year_month,
                    "date": date_display,
                    "categories": str(convert_to_json_serializable(paper.get("categories", ""))),
                    "affiliation": str(convert_to_json_serializable(paper.get("affiliation", ""))),
                    "country": str(convert_to_json_serializable(paper.get("affiliation_country", ""))),
                    "department": str(convert_to_json_serializable(paper.get("department", ""))),
                    "corresponding_email": str(convert_to_json_serializable(paper.get("corresponding_email", ""))),
                }
                try:
                    country_val = str(convert_to_json_serializable(paper.get("affiliation_country", "")).strip())
                except Exception:
                    country_val = ""
                if country_val and country_val.upper() != 'N/A':
                    rec["country"] = country_val
                try:
                    dept_val = str(convert_to_json_serializable(paper.get("department", "")).strip())
                except Exception:
                    dept_val = ""
                if dept_val and dept_val.upper() != 'N/A':
                    rec["department"] = dept_val
                recommendations.append(rec)
        
        if debug_timing:
            print(f"[rec-debug] total Stage1 time={time.time() - stage_start:.3f}s | returned={len(recommendations)} | use_lance={use_lance}")
        return recommendations

    def _prepare_filter_params(self, date_min, date_max, countries, categories):
        def _parse_date(value):
            try:
                if value is None or str(value).strip() == "":
                    return None
                return pd.to_datetime(str(value)).to_pydatetime().date()
            except Exception:
                return None

        start_dt = _parse_date(date_min)
        end_dt = _parse_date(date_max)
        country_set = {str(c).strip().upper() for c in (countries or []) if str(c).strip()}
        country_set = {c for c in country_set if re.fullmatch(r"[A-Z]{3}", c)}
        category_tokens = [str(c).strip().upper() for c in (categories or []) if str(c).strip()]
        category_tokens = [c for c in category_tokens if re.fullmatch(r"[A-Z]{2}", c)]
        return {
            "start_dt": start_dt,
            "end_dt": end_dt,
            "need_date": bool(start_dt or end_dt),
            "country_set": country_set,
            "need_country": bool(country_set),
            "category_tokens": category_tokens,
            "need_category": bool(category_tokens),
        }

    def _exact_search(self, query_vec, top_k, date_min, date_max, countries, categories, debug_timing=False):
        params = self._prepare_filter_params(date_min, date_max, countries, categories)
        need_filters = params["need_date"] or params["need_country"] or params["need_category"]
        valid_indices = None
        if need_filters:
            mask = pd.Series([True] * self.num_items)
            if params["need_date"]:
                dates = self.df.get('date_parsed')
                start_dt = params["start_dt"]
                end_dt = params["end_dt"]
                if dates is not None:
                    dates_clip = dates.iloc[:self.num_items]
                    if start_dt is not None:
                        mask &= dates_clip.notna() & (dates_clip.dt.date >= start_dt)
                    if end_dt is not None:
                        mask &= dates_clip.notna() & (dates_clip.dt.date <= end_dt)
                else:
                    mask &= False
            if params["need_country"]:
                col_country = self.df.get('affiliation_country')
                if col_country is not None:
                    col_clip = col_country.iloc[:self.num_items].astype(str).str.strip().str.upper()
                    mask &= col_clip.isin(params["country_set"])
                else:
                    mask &= False
            if params["need_category"]:
                col_cats = self.df.get('categories')
                if col_cats is not None:
                    tokens = params["category_tokens"]
                    col_clip = col_cats.iloc[:self.num_items].astype(str).fillna("")
                    def _row_matches_any(v: str) -> bool:
                        s = (v or "").upper()
                        if not s or s == 'NAN':
                            return False
                        for code in tokens:
                            if f"CS.{code}" in s:
                                return True
                        return False
                    mask &= col_clip.map(_row_matches_any)
                else:
                    mask &= False
            valid_indices = np.where(mask.values)[0]
            if debug_timing:
                print(f"[rec-debug] exact-filter subset size={0 if valid_indices is None else valid_indices.size}")
            if valid_indices.size == 0:
                return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        search_embeddings = self.embeddings if valid_indices is None else self.embeddings[valid_indices]
        if search_embeddings.size == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        sim_start = time.time()
        similarities = cosine_similarity([query_vec], search_embeddings)[0]
        sim_end = time.time()
        if debug_timing:
            print(f"[rec-debug] cosine_similarity over subset={similarities.size} took {sim_end - sim_start:.3f}s")
        order = np.argsort(similarities)[::-1]
        order = order[:min(top_k, order.size)]
        scores = similarities[order]
        if valid_indices is not None:
            indices = valid_indices[order]
        else:
            indices = order
        return indices.astype(np.int64, copy=False), scores.astype(np.float32, copy=False)

    def _lance_search(self, query_vec, top_k, date_min, date_max, countries, categories, debug_timing=False):
        if self.lance_table is None:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        params = self._prepare_filter_params(date_min, date_max, countries, categories)
        search_k = min(self.num_items, max(top_k, self._lance_search_k))
        where_expr = self._build_lance_where(params)
        try:
            query = self.lance_table.search(query_vec.tolist()).metric("cosine")
            if where_expr:
                query = query.where(where_expr)
            search_start = time.time()
            result_df = query.limit(search_k).to_pandas()
            search_end = time.time()
            if debug_timing:
                print(f"[rec-debug] LanceDB search k={search_k} returned={len(result_df)} in {search_end - search_start:.3f}s where={where_expr or 'none'}")
        except Exception as e:
            print(f"[lance] search error: {e}")
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        if result_df.empty:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        if '_distance' in result_df.columns:
            scores = 1.0 - result_df['_distance'].to_numpy(dtype=np.float32)
        elif '_score' in result_df.columns:
            scores = result_df['_score'].to_numpy(dtype=np.float32)
        else:
            scores = result_df.get('score', pd.Series([0.0] * len(result_df))).to_numpy(dtype=np.float32)
        indices = result_df['row_id'].to_numpy(dtype=np.int64)
        if params["need_date"] or params["need_country"] or params["need_category"]:
            keep_idx = []
            keep_scores = []
            for idx, score in zip(indices, scores):
                if self._passes_filters(idx, params):
                    keep_idx.append(idx)
                    keep_scores.append(score)
            indices = np.array(keep_idx, dtype=np.int64)
            scores = np.array(keep_scores, dtype=np.float32)
        return indices, scores

    def _build_lance_where(self, params):
        clauses = []
        if params["need_country"]:
            country_clause = " OR ".join([f"country = '{c}'" for c in sorted(params["country_set"])])
            if country_clause:
                clauses.append(f"({country_clause})")
        if params["need_date"]:
            if params["start_dt"] is not None:
                clauses.append(f"(date_str != '' AND date_str >= '{params['start_dt'].isoformat()}')")
            if params["end_dt"] is not None:
                clauses.append(f"(date_str != '' AND date_str <= '{params['end_dt'].isoformat()}')")
        if params["need_category"]:
            cat_clauses = [f"categories ILIKE '%CS.{code}%'" for code in params["category_tokens"]]
            if cat_clauses:
                clauses.append("(" + " OR ".join(cat_clauses) + ")")
        return " AND ".join(clauses)

    def _passes_filters(self, row_idx: int, params: dict) -> bool:
        try:
            paper = self.df.iloc[row_idx]
        except Exception:
            return False
        if params["need_country"]:
            try:
                country_val = str(convert_to_json_serializable(paper.get("affiliation_country", ""))).strip().upper()
            except Exception:
                country_val = ""
            if country_val not in params["country_set"]:
                return False
        if params["need_date"]:
            date_val = paper.get('date_parsed', None)
            if pd.isna(date_val):
                return False
            date_only = pd.to_datetime(date_val).date()
            if params["start_dt"] is not None and date_only < params["start_dt"]:
                return False
            if params["end_dt"] is not None and date_only > params["end_dt"]:
                return False
        if params["need_category"]:
            cats = str(convert_to_json_serializable(paper.get("categories", "")) or "").upper()
            if not cats or cats == 'NAN':
                return False
            hit = False
            for code in params["category_tokens"]:
                if f"CS.{code}" in cats:
                    hit = True
                    break
            if not hit:
                return False
        return True

    def _init_lancedb(self):
        if not self.has_embeddings or self.num_items <= 0:
            return
        if lancedb is None:
            print("[lance] LanceDB not installed; skipping Lance acceleration.")
            return
        try:
            os.makedirs(self._lance_uri, exist_ok=True)
            self.lance_db = lancedb.connect(self._lance_uri)
        except Exception as e:
            print(f"[lance] Failed to connect DB: {e}")
            self.lance_db = None
            return
        try:
            existing = []
            try:
                existing = self.lance_db.table_names()
            except Exception:
                existing = []
            if self._lance_table_name in existing:
                table = self.lance_db.open_table(self._lance_table_name)
                if table.count() >= self.num_items:
                    self.lance_table = table
                    print(f"✓ LanceDB table loaded: {table.count()} rows")
                    return
                else:
                    try:
                        self.lance_db.drop_table(self._lance_table_name)
                    except Exception:
                        pass
            self._build_lance_table()
        except Exception as e:
            print(f"[lance] initialization error: {e}")
            self.lance_table = None

    def _build_lance_table(self):
        if self.lance_db is None or self.num_items <= 0:
            return
        print(f"[lance] Building table '{self._lance_table_name}' at {self._lance_uri} (records={self.num_items}) ...")
        batch_iter = self._yield_lance_batches(self._lance_build_chunk)
        first_batch = next(batch_iter, None)
        if not first_batch:
            print("[lance] No data available for LanceDB table.")
            return
        table = self.lance_db.create_table(self._lance_table_name, data=first_batch, mode="overwrite")
        added = len(first_batch)
        for batch in batch_iter:
            if not batch:
                continue
            table.add(batch)
            added += len(batch)
            if added % max(self._lance_build_chunk * 10, 50000) == 0:
                print(f"[lance] added {added}/{self.num_items} records...")
        print(f"✓ LanceDB table built with {added} rows")
        self.lance_table = table

    def _yield_lance_batches(self, batch_size: int):
        if batch_size <= 0:
            batch_size = 1000
        for start in range(0, self.num_items, batch_size):
            end = min(self.num_items, start + batch_size)
            vec_chunk = np.asarray(self.embeddings[start:end], dtype=np.float32)
            chunk_df = self.df.iloc[start:end]
            records = []
            for idx_offset, (df_index, row) in enumerate(chunk_df.iterrows()):
                vector = vec_chunk[idx_offset].tolist()
                date_val = row.get('date_parsed', None)
                date_str = ""
                try:
                    if pd.notna(date_val):
                        date_str = pd.to_datetime(date_val).strftime('%Y-%m-%d')
                except Exception:
                    date_str = ""
                try:
                    year_numeric = convert_to_json_serializable(row.get('year', ''))
                except Exception:
                    year_numeric = ""
                try:
                    country_val = str(convert_to_json_serializable(row.get('affiliation_country', ""))).strip().upper()
                except Exception:
                    country_val = ""
                record = {
                    "row_id": int(df_index),
                    "vector": vector,
                    "title": str(convert_to_json_serializable(row.get('title', ''))),
                    "abstract": str(convert_to_json_serializable(row.get('abstract', ''))),
                    "authors": str(convert_to_json_serializable(row.get('authors', row.get('author', '')))),
                    "categories": str(convert_to_json_serializable(row.get('categories', ''))),
                    "country": country_val,
                    "date_str": date_str,
                    "year": str(year_numeric),
                }
                records.append(record)
            if records:
                yield records

# 全局推荐器实例
recommender = None
JOBS = {}
RESULTS = {}
CROSS_ENCODER = None

def parse_arxiv_year_month(arxiv_id: str) -> str:
    """Infer YYYY-MM from arXiv ID (new and old formats). Return empty string if not parseable."""
    try:
        if not arxiv_id or not isinstance(arxiv_id, str):
            return ""
        base = arxiv_id.strip()
        # Remove version suffix like v2/v10 (only trim trailing version marker)
        m_ver = re.match(r"^(.*?)(v\d+)$", base)
        if m_ver:
            base = m_ver.group(1)
        # New format: YYMM.NNNNN
        m_new = re.match(r"^(\d{2})(\d{2})\.\d+", base)
        if m_new:
            yy = int(m_new.group(1))
            mm = int(m_new.group(2))
            year = 1900 + yy if yy >= 90 else 2000 + yy
            return f"{year}-{mm:02d}"
        # Old format: prefix/YYMMNNN
        m_old = re.match(r"^[a-zA-Z\-]+\/(\d{2})(\d{2})\d+", base)
        if m_old:
            yy = int(m_old.group(1))
            mm = int(m_old.group(2))
            year = 1900 + yy if yy >= 90 else 2000 + yy
            return f"{year}-{mm:02d}"
    except Exception:
        pass
    return ""

def split_into_topics(text: str, max_topics: int = 6) -> list[str]:
    """Heuristically split input into multiple topics (bullets/numbers supported)."""
    if not text:
        return []
    # Normalize whitespace and newlines (handles NBSP and full-width spaces)
    normalized = (text.replace("\u00A0", " ").replace("\u3000", " "))
    normalized = re.sub(r"[\t]", " ", normalized)
    # Keep newlines while trimming extra carriage returns
    normalized = normalized.replace("\r", "\n").strip()
    # If bullet/number markers appear (including common Unicode bullets in English/Chinese)
    bullet_chars = "\\-\\*•●○◦▪‣·–—"  # hyphen/asterisk/bullet/black circle/white circle/small circle/square/triangle/dot/en dash/em dash
    # Prefer segment extraction: text between one bullet and the next
    segment_regex = rf"(?m)(?:^|[\n])\s*(?:\d+[\.\)]|[（(]?\d+[）)]|[一二三四五六七八九十]+、|[{bullet_chars}])\s*(.+?)(?=(?:[\n]\s*(?:\d+[\.\)]|[（(]?\d+[）)]|[一二三四五六七八九十]+、|[{bullet_chars}])\s*)|\Z)"
    bullet_lines = re.findall(segment_regex, normalized)
    if bullet_lines:
        topics = [ln.strip() for ln in bullet_lines if len(ln.strip()) >= 5]
    else:
        # Try line-by-line prefix bullet detection (covers ● / • etc.)
        line_regex = rf"^\s*(?:\d+[\.\)]|[（(]?\d+[）)]|[一二三四五六七八九十]+、|[{bullet_chars}])\s+(.*\S)\s*$"
        lines = normalized.split("\n")
        line_hits = []
        for ln in lines:
            m = re.match(line_regex, ln)
            if m:
                val = m.group(1).strip()
                if len(val) >= 3:
                    line_hits.append(val)
        if len(line_hits) >= 2:
            topics = line_hits
        else:
            # Conservatively split by explicit delimiters (excluding newlines and periods)
            parts = re.split(r"[;；、]+", normalized)
            non_empty = [p for p in parts if p.strip()]
            # If still a single segment, consider conjunctions (English and Chinese) when they appear at least twice
            if len(non_empty) <= 1:
                connector_matches = re.findall(r"\s+(?:and|以及|和)\s+", normalized)
                if len(connector_matches) >= 2:
                    parts = re.split(r"\s+(?:and|以及|和)\s+", normalized)
                else:
                    parts = [normalized]
            # If still too few, try splitting again by inline bullet markers (e.g., consecutive "● topic")
            expanded = []
            for p in parts:
                p_stripped = p.strip()
                if not p_stripped:
                    continue
                sub_parts = re.split(rf"\s*(?:[{bullet_chars}]+)\s+", p_stripped)
                sub_parts = [sp.strip() for sp in sub_parts if len(sp.strip()) >= 5]
                if len(sub_parts) > 1:
                    expanded.extend(sub_parts)
                else:
                    expanded.append(p_stripped)
            topics = [p for p in expanded if len(p) >= 5]
    # Deduplicate and cap the number of topics
    seen = set()
    dedup = []
    for t in topics:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            dedup.append(t)
        if len(dedup) >= max_topics:
            break
    return dedup if dedup else [normalized]

def _build_doc_text(paper: dict) -> str:
    title = str(paper.get('title', '') or '').strip()
    abstract = str(paper.get('abstract', '') or '').strip()
    if title and abstract:
        return f"{title}\n\n{abstract}"
    return title or abstract or ""

def _qwen3_rerank(query: str, papers: list[dict], batch_size: int = 25, on_batch=None, top_n_limit: int | None = None) -> list[float]:
    """Call Qwen/Qwen3-Reranker-8B in batches, return scores.
    Requires env QWEN_RERANK_API_KEY; optional QWEN_RERANK_HOST/QWEN_RERANK_PATH/QWEN_RERANK_MODEL.
    """
    api_key = (os.getenv('CHATFIRE_API_KEY') or os.getenv('QWEN_RERANK_API_KEY') or '').strip()
    if not api_key:
        try:
            if os.path.exists('secrets.json'):
                with open('secrets.json', 'r', encoding='utf-8') as f:
                    secrets = json.load(f)
                    api_key = str(secrets.get('CHATFIRE_API_KEY') or secrets.get('QWEN_RERANK_API_KEY') or '').strip()
        except Exception:
            pass
    host = (os.getenv('QWEN_RERANK_HOST') or 'api.chatfire.cn').strip()
    path = (os.getenv('QWEN_RERANK_PATH') or '/v1/rerank').strip()
    model = os.getenv('QWEN_RERANK_MODEL', 'qwen3-reranker-8b')
    if os.getenv('RERANK_DEBUG', '0') == '1':
        try:
            print(f"[rerank] using remote model={model} via {host}{path} key_present={bool(api_key)}")
        except Exception:
            pass
    if not api_key or not papers:
        return []
    scores: list[float] = [0.0] * len(papers)
    total = len(papers)
    batches = (total + batch_size - 1) // batch_size
    try:
        for b in range(batches):
            s = b * batch_size
            e = min(s + batch_size, total)
            sub_docs = [_build_doc_text(p) for p in papers[s:e]]
            dbg_top_n = int(min(len(sub_docs), top_n_limit)) if top_n_limit else len(sub_docs)
            if os.getenv('RERANK_DEBUG', '0') == '1':
                print(f"[rerank] preparing request: host={host} path={path} model={model} batch={b+1}/{batches} docs={len(sub_docs)} top_n={dbg_top_n} key_present={bool(api_key)}")
            payload = json.dumps({
                "model": model,
                "query": query,
                "top_n": dbg_top_n,
                "documents": sub_docs
            })
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            # Retry up to 3 times
            for attempt in range(3):
                try:
                    conn = http.client.HTTPSConnection(host, timeout=15)
                    conn.request("POST", path, payload, headers)
                    resp = conn.getresponse()
                    data = resp.read()
                    conn.close()
                    if os.getenv('RERANK_DEBUG', '0') == '1':
                        print(f"[rerank] response status={resp.status} bytes={len(data) if data else 0}")
                    if resp.status != 200:
                        snippet = (data[:200] if data else b'').decode('utf-8', errors='ignore')
                        raise RuntimeError(f"HTTP {resp.status}: {snippet}")
                    result = json.loads(data.decode('utf-8')) if data else {}
                    if isinstance(result, dict) and 'results' in result:
                        for item in result['results']:
                            idx = int(item.get('index', 0))
                            score = float(item.get('relevance_score', 0.0))
                            if 0 <= idx < (e - s):
                                scores[s + idx] = score
                    if on_batch is not None:
                        try:
                            on_batch()
                        except Exception:
                            pass
                    break
                except Exception as ex:
                    if os.getenv('RERANK_DEBUG', '0') == '1':
                        print(f"[rerank] attempt {attempt+1}/3 failed: {ex}")
                    if attempt == 2:
                        print(f"[rerank] giving up after 3 attempts: {ex}")
                        return []
                    time.sleep(0.2 * (2 ** attempt))
            time.sleep(0.05)
        return scores
    except Exception as e:
        print(f"[rerank] exception: {e}")
        return []

def _get_cross_encoder():
    global CROSS_ENCODER
    if CROSS_ENCODER is not None:
        return CROSS_ENCODER
    if CrossEncoder is None:
        return None
    model_name = os.getenv('CROSS_ENCODER_MODEL', 'cross-encoder/ms-marco-MiniLM-L-12-v2')
    try:
        CROSS_ENCODER = CrossEncoder(model_name)
        return CROSS_ENCODER
    except Exception:
        return None

def _cross_encoder_rerank(query: str, papers: list[dict], batch_size: int = 32, on_batch=None) -> list[float]:
    ce = _get_cross_encoder()
    if ce is None or not papers:
        return []
    if os.getenv('RERANK_DEBUG', '0') == '1':
        try:
            model_name = os.getenv('CROSS_ENCODER_MODEL', 'cross-encoder/ms-marco-MiniLM-L-12-v2')
            print(f"[rerank] using local model={model_name} (Cross-Encoder) for rerank")
        except Exception:
            pass
    pairs = []
    for p in papers:
        text = _build_doc_text(p)
        pairs.append((query, text))
    scores: list[float] = []
    try:
        for i in range(0, len(pairs), batch_size):
            sub = pairs[i:i+batch_size]
            sub_scores = ce.predict(sub)
            scores.extend([float(s) for s in sub_scores])
            if on_batch is not None:
                try:
                    on_batch()
                except Exception:
                    pass
        return scores
    except Exception:
        return []

def run_recommend_job(job_id: str, payload: dict):
    try:
        JOBS[job_id] = {"percent": 0, "stage": 1, "message": "Stage 1/2 (Recall): Using embedding model for scientific tasks (allenai/specter2_base) to compute embeddings and retrieve top-200 candidates by cosine similarity...", "done": False}
        query = payload.get('workshop_description', '')
        top_k = int(payload.get('top_k', 20))
        date_min = payload.get('date_min')
        date_max = payload.get('date_max')
        countries = payload.get('countries') or []
        categories = payload.get('categories') or []
        min_similarity = payload.get('min_similarity')
        auto_split = bool(payload.get('auto_split', True))
        recall_top = int(payload.get('recall_top', 200))

        start_time = time.time()
        topics = split_into_topics(query, max_topics=6) if auto_split else [query]
        grouped = []

        JOBS[job_id].update({"percent": 5, "stage": 1, "message": "Stage 1/2 (Recall): Computing embeddings and recalling top-200 by cosine similarity..."})
        for idx_t, t in enumerate(topics, start=1):
            # Stage 1 performs recall only; min threshold is applied in Stage 2 via reranker scores
            recs = recommender.recommend_papers(
                t,
                top_k=recall_top,
                date_min=date_min,
                date_max=date_max,
                countries=countries,
                categories=categories,
                min_similarity=None,
            )
            grouped.append({"topic_index": idx_t, "topic": t, "recommendations": recs, "count": len(recs)})
        # Prepare active topics (non-empty) for clearer progress messages
        active_groups = [g for g in grouped if g.get('recommendations')]
        topics_total = len(active_groups)
        total_batches = sum(((len(g.get('recommendations', [])) + 25 - 1) // 25) for g in active_groups)
        # Initial Stage-2 message with first topic hint if any
        topic_hint = ""
        if topics_total >= 1:
            first_topic = active_groups[0]['topic']
            short_first = (first_topic[:80] + '...') if len(first_topic) > 80 else first_topic
            topic_hint = f" Topic 1/{topics_total}: {short_first}"
        JOBS[job_id].update({"percent": 20, "stage": 2, "message": f"Stage 2/2 (Rerank): Reranking with Qwen/Qwen3-Reranker-8B. Processing in 0/{max(1, total_batches)} batches...{topic_hint}"})

        processed_batches = 0
        all_recs = []

        # Process topics sequentially, but use thread pool for batch-level parallelism within each topic
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        progress_lock = threading.Lock()
        max_workers = 1
        try:
            max_workers = max(1, int(os.getenv('RERANK_MAX_CONCURRENCY', '3')))
        except Exception:
            max_workers = 3

        def process_topic(topic_pos: int, g: dict):
            nonlocal processed_batches
            t = g['topic']
            recs = g.get('recommendations', [])
            short_topic = (t[:80] + '...') if len(t) > 80 else t

            def _on_batch_local():
                nonlocal processed_batches
                with progress_lock:
                    processed_batches += 1
                    pct = 20 + int(80 * processed_batches / max(1, total_batches))
                    JOBS[job_id].update({
                        "percent": min(99, pct),
                        "stage": 2,
                        "message": f"Stage 2/2 (Rerank): Reranking with Qwen/Qwen3-Reranker-8B. Processing in {processed_batches}/{total_batches} batches... Topic {topic_pos}/{topics_total}: {short_topic}"
                    })

            # Use thread pool for batch-level parallelism within this topic
            if max_workers > 1 and len(recs) > 25:  # Only use parallel processing for larger datasets
                # Split recs into batches and process them in parallel
                batch_size = 25
                batches = [recs[i:i + batch_size] for i in range(0, len(recs), batch_size)]
                
                def process_single_batch(batch_recs, batch_idx):
                    """Process a single batch with direct API call"""
                    try:
                        # Build documents for this batch
                        sub_docs = [_build_doc_text(p) for p in batch_recs]
                        
                        # Direct API call for this batch (similar to _qwen3_rerank but for single batch)
                        api_key = (os.getenv('CHATFIRE_API_KEY') or os.getenv('QWEN_RERANK_API_KEY') or '').strip()
                        if not api_key:
                            try:
                                if os.path.exists('secrets.json'):
                                    with open('secrets.json', 'r', encoding='utf-8') as f:
                                        secrets = json.load(f)
                                        api_key = str(secrets.get('CHATFIRE_API_KEY') or secrets.get('QWEN_RERANK_API_KEY') or '').strip()
                            except Exception:
                                pass
                        
                        if not api_key:
                            return batch_recs, batch_idx
                        
                        host = (os.getenv('QWEN_RERANK_HOST') or 'api.chatfire.cn').strip()
                        path = (os.getenv('QWEN_RERANK_PATH') or '/v1/rerank').strip()
                        model = os.getenv('QWEN_RERANK_MODEL', 'qwen3-reranker-8b')
                        
                        payload = json.dumps({
                            "model": model,
                            "query": t,
                            "top_n": min(len(sub_docs), top_k) if top_k else len(sub_docs),
                            "documents": sub_docs
                        })
                        headers = {
                            'Authorization': f'Bearer {api_key}',
                            'Content-Type': 'application/json'
                        }
                        
                        # Make API call
                        conn = http.client.HTTPSConnection(host, timeout=15)
                        conn.request("POST", path, payload, headers)
                        resp = conn.getresponse()
                        data = resp.read()
                        conn.close()
                        
                        if resp.status == 200:
                            result = json.loads(data.decode('utf-8')) if data else {}
                            if isinstance(result, dict) and 'results' in result:
                                for item in result['results']:
                                    idx = int(item.get('index', 0))
                                    score = float(item.get('relevance_score', 0.0))
                                    if 0 <= idx < len(batch_recs):
                                        batch_recs[idx]['rerank_score'] = score
                        
                        return batch_recs, batch_idx
                    except Exception as e:
                        print(f"Batch {batch_idx} failed: {e}")
                        return batch_recs, batch_idx
                
                # Process batches in parallel
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(process_single_batch, batch, idx): idx for idx, batch in enumerate(batches)}
                    
                    # Collect results in order
                    batch_results = [None] * len(batches)
                    for fut in as_completed(futures):
                        try:
                            batch_recs, batch_idx = fut.result()
                            batch_results[batch_idx] = batch_recs
                            _on_batch_local()  # Update progress for each completed batch
                        except Exception:
                            _on_batch_local()  # Still update progress even if batch failed
                    
                    # Flatten results back to original order
                    recs = []
                    for batch_result in batch_results:
                        if batch_result:
                            recs.extend(batch_result)
            else:
                # Sequential processing for smaller datasets or when parallel processing is disabled
                scores = _qwen3_rerank(t, recs, batch_size=25, on_batch=_on_batch_local, top_n_limit=top_k)
                if not scores:
                    with progress_lock:
                        JOBS[job_id].update({"stage": 2, "message": f"Stage 2/2 (Rerank): Qwen API error, falling back to Cross-Encoder reranker... Topic {topic_pos}/{topics_total}: {short_topic}"})
                    scores = _cross_encoder_rerank(t, recs, batch_size=32, on_batch=_on_batch_local)
                if not scores:
                    # Fallback: count one batch to keep progress moving
                    _on_batch_local()

            # Sort and process results
            if any('rerank_score' in r for r in recs):
                recs_sorted_local = sorted(recs, key=lambda r: r.get('rerank_score', 0.0), reverse=True)
            else:
                recs_sorted_local = sorted(recs, key=lambda r: r.get('similarity', 0.0), reverse=True)
            
            for r in recs_sorted_local:
                if 'rerank_score' in r:
                    r['similarity'] = float(r['rerank_score'])
                r['_topic_index'] = g['topic_index']
                r['_topic'] = t
            
            if min_similarity is not None:
                try:
                    thr = float(min_similarity)
                    recs_sorted_local = [r for r in recs_sorted_local if isinstance(r.get('similarity'), (int, float)) and r.get('similarity', 0.0) >= thr]
                except Exception:
                    pass
            return recs_sorted_local

        # Process topics sequentially
        for topic_pos, g in enumerate(active_groups, start=1):
            try:
                all_recs.extend(process_topic(topic_pos, g))
            except Exception:
                pass

        end_time = time.time()
        # After global reranking, trim to top_k to keep output size consistent
        all_recs_sorted = sorted(all_recs, key=lambda r: r.get('similarity', 0.0), reverse=True)
        final_recs = all_recs_sorted[:top_k]
        RESULTS[job_id] = {
            "recommendations": final_recs,
            "count": len(final_recs),
            "search_time": round(end_time - start_time, 2)
        }
        JOBS[job_id].update({"percent": 100, "done": True, "message": "Completed."})
    except Exception as e:
        JOBS[job_id].update({"done": True, "error": str(e), "message": "Error."})

@app.route('/')
def index():
    """Homepage"""
    return render_template('index.html')

@app.route('/api/recommend', methods=['POST'])
def recommend():
    """Recommendation API"""
    global recommender
    
    if recommender is None:
        return jsonify({"error": "Recommender not initialized"}), 500
    
    try:
        data = request.get_json()
        workshop_description = data.get('workshop_description', '')
        top_k = data.get('top_k', 20)
        date_min = data.get('date_min')
        date_max = data.get('date_max')
        countries = data.get('countries') or []
        categories = data.get('categories') or []
        min_similarity = data.get('min_similarity')
        
        if not workshop_description:
            return jsonify({"error": "Please provide a workshop description"}), 400
        
        # Normalize date input (YYYY-MM-DD or YYYY-MM)
        def _to_date_or_none(value):
            try:
                if value is None or str(value).strip() == "":
                    return None
                v = str(value).strip()
                # Basic format validation
                _ = pd.to_datetime(v, errors='raise')
                return v
            except Exception:
                return None
        date_min = _to_date_or_none(date_min)
        date_max = _to_date_or_none(date_max)
        if not isinstance(countries, list):
            countries = []
        if not isinstance(categories, list):
            categories = []

        # Normalize minimum similarity
        def _to_float_or_none(value):
            try:
                if value is None or value == "":
                    return None
                v = float(value)
                if v < 0:
                    v = 0.0
                if v > 1:
                    v = 1.0
                return v
            except Exception:
                return None
        min_similarity = _to_float_or_none(min_similarity)
        
        # two-stage async flow: create job and return job_id
        job_payload = {
            'workshop_description': workshop_description,
            'top_k': top_k,
            'date_min': date_min,
            'date_max': date_max,
            'countries': countries,
            'categories': categories,
            'min_similarity': min_similarity,
            'auto_split': True,
            'recall_top': 200,
        }
        job_id = uuid.uuid4().hex
        JOBS[job_id] = {"percent": 0, "stage": 0, "message": "Queued...", "done": False}
        thread = threading.Thread(target=run_recommend_job, args=(job_id, job_payload), daemon=True)
        thread.start()
        return jsonify({"job_id": job_id})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/health')
def health():
    """Health check"""
    global recommender
    if recommender is None:
        return jsonify({"status": "error", "message": "Recommender not initialized"})
    return jsonify({"status": "ok", "message": "Recommender is running"})

@app.route('/api/discover_topics', methods=['POST'])
def discover_topics():
    """Topic discovery API - Paper2Topic"""
    global recommender
    
    if recommender is None:
        return jsonify({"error": "Recommender not initialized"}), 500
    
    try:
        data = request.get_json()
        date_min = data.get('date_min')
        date_max = data.get('date_max')
        countries = data.get('countries') or []
        categories = data.get('categories') or []
        n_topics = data.get('n_topics', 10)
        
        # Validate inputs
        try:
            n_topics = int(n_topics)
            if n_topics < 2 or n_topics > 50:
                return jsonify({"error": "Topic count must be between 2 and 50"}), 400
        except Exception:
            return jsonify({"error": "Invalid topic count"}), 400
        
        # Create background job
        job_payload = {
            'date_min': date_min,
            'date_max': date_max,
            'countries': countries,
            'categories': categories,
            'n_topics': n_topics,
        }
        job_id = uuid.uuid4().hex
        JOBS[job_id] = {"percent": 0, "stage": 0, "message": "Queued...", "done": False}
        thread = threading.Thread(target=run_discover_topics_job, args=(job_id, job_payload), daemon=True)
        thread.start()
        return jsonify({"job_id": job_id})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/feedback', methods=['POST'])
def feedback():
    """Receive user feedback (upvote/downvote) and append to JSONL"""
    try:
        data = request.get_json() or {}
        record = {
            "action": data.get('action'),
            "workshop_description": data.get('workshop_description'),
            "rank": data.get('rank'),
            "similarity": data.get('similarity'),
            "paper": data.get('paper'),
            "arxiv_id": data.get('arxiv_id'),
            "title": data.get('title'),
            "authors": data.get('authors'),
            "recommendation_params": data.get('recommendation_params'),
            "recommendation_meta": data.get('recommendation_meta'),
            "page_url": data.get('page_url'),
            "timestamp": data.get('timestamp') or time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            "client_ip": _get_client_ip(request),
            "user_agent": request.headers.get('User-Agent', ''),
        }
        os.makedirs('logs', exist_ok=True)
        with open(os.path.join('logs', 'feedback.jsonl'), 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/progress')
def progress():
    job_id = request.args.get('job_id', '')
    if not job_id or job_id not in JOBS:
        return jsonify({"error": "job_id not found"}), 404
    return jsonify(JOBS[job_id])

@app.route('/api/result')
def result():
    job_id = request.args.get('job_id', '')
    if not job_id:
        return jsonify({"error": "job_id required"}), 400
    if job_id not in RESULTS:
        return jsonify({"error": "result not ready"}), 404
    return jsonify(RESULTS[job_id])
    # payload = RESULTS.get(job_id)
    # response = jsonify(payload)
    # # 清理已经完成的任务，防止 JOBS/RESULTS 堆积
    # try:
    #     RESULTS.pop(job_id, None)
    #     JOBS.pop(job_id, None)
    # except Exception:
    #     pass
    # return response

def create_templates():
    """Create the HTML template directory structure (templates live externally)"""
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    # Templates and static assets are created outside this module

# ===== Topic Discovery utilities =====
def clean_text_for_clustering(text: str) -> str:
    """Clean text for topic clustering"""
    if not isinstance(text, str):
        return ""
    t = text
    # Remove LaTeX
    t = re.sub(r"\$[^$]*\$", " ", t)
    t = re.sub(r"\\\(|\\\)|\\\[|\\\]", " ", t)
    t = re.sub(r"\\begin\{[^}]+\}.*?\\end\{[^}]+\}", " ", t, flags=re.S)
    t = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", " ", t)
    t = re.sub(r"\\[a-zA-Z]+", " ", t)
    t = t.lower()
    # Remove tokens with digits
    t = re.sub(r"\b\w*\d\w*\b", " ", t)
    t = re.sub(r"[^a-z\-\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def perform_topic_clustering(papers: list[dict], embeddings: np.ndarray, n_topics: int = 10) -> tuple:
    """Perform BERTopic clustering on filtered papers"""
    print(f"Performing topic clustering on {len(papers)} papers...")
    
    # Stopwords
    stopwords = {
        'the','of','and','to','in','for','is','that','with','a','an','as','are','was','were','be','been','have','has','had','do','does','did','will','would','could','should','may','might','can','this','these','those','we','you','they','it','he','she','him','her','his','hers','its','our','your','their','or','but','not','at','by','from','up','about','into','through','during','before','after','above','below','between','among','under','over','on','off','out','down','away','back','here','there','where','when','why','how','all','any','both','each','few','more','most','other','some','such','no','nor','only','own','same','so','than','too','very','just','now','which',
        'using','based','approach','method','methods','system','systems','algorithm','algorithms','framework','analysis','study','research','paper','work','results','proposed','novel','new','model','models','learning','deep','neural','network','networks','data','dataset','datasets','classification','detection','recognition','optimization','performance','accuracy','efficient','robust','effective','task','tasks','problem','problems','via','towards','toward','across','general','generalization','stateoftheart','baseline','benchmarks','improving','improved'
    }
    
    # Prepare documents
    documents = []
    for paper in papers:
        title = paper.get('title', '')
        abstract = paper.get('abstract', '')
        raw = f"{title} {abstract}".strip()
        cleaned = clean_text_for_clustering(raw)
        words = cleaned.split()
        filtered_words = [w for w in words if w not in stopwords and len(w) > 2]
        documents.append(' '.join(filtered_words))
    
    # UMAP configuration
    umap_model = umap.UMAP(
        n_neighbors=min(50, len(papers) // 10),
        n_components=min(50, len(papers) // 5),
        min_dist=0.01,
        metric='cosine',
        random_state=42
    )
    
    # HDBSCAN configuration
    data_size = len(papers)
    if data_size < 300:
        min_cluster_size = max(5, data_size // 60)
        min_samples = 2
    elif data_size < 800:
        min_cluster_size = max(8, data_size // 100)
        min_samples = 3
    else:
        min_cluster_size = max(10, data_size // 150)
        min_samples = 4
    
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )
    
    # Vectorizer
    vectorizer_model = CountVectorizer(
        ngram_range=(1, 3),
        stop_words=list(stopwords),
        min_df=2,
        max_df=0.7,
        max_features=5000
    )
    
    # BERTopic
    topic_model = BERTopic(
        embedding_model=None,
        vectorizer_model=vectorizer_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        nr_topics=None,
        calculate_probabilities=False,
        verbose=False,
        low_memory=True
    )
    
    # Fit model
    topic_labels, _ = topic_model.fit_transform(documents, embeddings=embeddings)
    
    # Reduce to n_topics if needed
    if n_topics and len(set(topic_labels)) > n_topics:
        topic_model.reduce_topics(documents, nr_topics=n_topics)
        topic_labels, _ = topic_model.transform(documents, embeddings=embeddings)
    
    return topic_labels, topic_model

def generate_topic_summary(topic_model: BERTopic, topic_labels: np.ndarray, papers: list[dict], embeddings: np.ndarray) -> list[dict]:
    """Generate summary for each discovered topic"""
    topic_info = topic_model.get_topic_info()
    results = []
    
    for _, row in topic_info.iterrows():
        topic_id = row['Topic']
        if topic_id == -1:  # Skip noise
            continue
        
        # Get papers in this topic
        topic_indices = [i for i, label in enumerate(topic_labels) if label == topic_id]
        topic_papers = [papers[i] for i in topic_indices]
        
        # Get top keywords
        keywords = topic_model.get_topic(topic_id) or []
        top_keywords = [w for w, _ in keywords[:10]]
        
        # Get centroid
        topic_embeddings = embeddings[topic_indices]
        centroid = np.mean(topic_embeddings, axis=0)
        
        # Find papers closest to centroid
        distances = np.linalg.norm(topic_embeddings - centroid, axis=1)
        closest_indices = np.argsort(distances)[:5]
        representative_papers = [topic_papers[i] for i in closest_indices]
        
        results.append({
            'topic_id': int(topic_id),
            'name': row.get('Name', f'Topic {topic_id}'),
            'count': int(row['Count']),
            'keywords': top_keywords,
            'representative_papers': [
                {
                    'title': p.get('title', ''),
                    'authors': p.get('authors', ''),
                    'arxiv_id': p.get('arxiv_id', ''),
                    'abstract': (p.get('abstract', '') or '')[:300]
                }
                for p in representative_papers
            ]
        })
    
    # Sort by count
    results.sort(key=lambda x: x['count'], reverse=True)
    return results

def run_discover_topics_job(job_id: str, payload: dict):
    """Background job for topic discovery"""
    try:
        JOBS[job_id] = {"percent": 10, "stage": 1, "message": "Filtering papers based on your criteria...", "done": False}
        
        date_min = payload.get('date_min')
        date_max = payload.get('date_max')
        countries = payload.get('countries') or []
        categories = payload.get('categories') or []
        n_topics = int(payload.get('n_topics', 10))
        
        # Apply filters to get subset of papers
        valid_indices = np.arange(recommender.num_items)
        need_date = (date_min is not None and str(date_min).strip() != "") or (date_max is not None and str(date_max).strip() != "")
        need_country = bool(countries)
        need_category = bool(categories)
        
        if need_date or need_country or need_category:
            mask = pd.Series([True] * recommender.num_items)
            
            # Date filter
            if need_date:
                try:
                    start_dt = pd.to_datetime(str(date_min)) if (date_min is not None and str(date_min).strip() != "") else None
                    end_dt = pd.to_datetime(str(date_max)) if (date_max is not None and str(date_max).strip() != "") else None
                except Exception:
                    start_dt, end_dt = None, None
                dates = recommender.df.get('date_parsed')
                if dates is not None:
                    dates_clip = dates.iloc[:recommender.num_items]
                    if start_dt is not None:
                        mask &= dates_clip.notna() & (dates_clip >= start_dt)
                    if end_dt is not None:
                        mask &= dates_clip.notna() & (dates_clip <= end_dt)
            
            # Country filter
            if need_country:
                selected_iso3 = {str(c).strip().upper() for c in countries if str(c).strip()}
                col_country = recommender.df.get('affiliation_country')
                if col_country is not None:
                    col_clip = col_country.iloc[:recommender.num_items].astype(str).str.strip().str.upper()
                    mask &= col_clip.isin(selected_iso3)
            
            # Category filter
            if need_category:
                selected_cats = [str(c).strip().upper() for c in categories if str(c).strip()]
                selected_cats = [c for c in selected_cats if re.fullmatch(r"[A-Z]{2}", c) is not None]
                col_cats = recommender.df.get('categories')
                if col_cats is not None and len(selected_cats) > 0:
                    col_clip = col_cats.iloc[:recommender.num_items].astype(str).fillna("")
                    def _row_matches_any(v: str) -> bool:
                        s = (v or "").upper()
                        if not s or s == 'NAN':
                            return False
                        for code in selected_cats:
                            token = f"CS.{code}"
                            if token in s:
                                return True
                        return False
                    match_series = col_clip.map(_row_matches_any)
                    mask &= match_series
            
            valid_indices = np.where(mask.values)[0]
        
        if valid_indices.size < 50:
            JOBS[job_id].update({"done": True, "error": "Not enough papers after filtering (minimum 50 required)", "message": "Error."})
            return
        
        JOBS[job_id].update({"percent": 30, "stage": 2, "message": f"Found {len(valid_indices)} papers. Performing topic clustering..."})
        
        # Get filtered papers and embeddings
        filtered_papers = []
        for idx in valid_indices[:5000]:  # Limit to 5000 for performance
            paper = recommender.df.iloc[idx]
            filtered_papers.append({
                'title': str(convert_to_json_serializable(paper.get('title', ''))),
                'authors': str(convert_to_json_serializable(paper.get('authors', ''))),
                'abstract': str(convert_to_json_serializable(paper.get('abstract', ''))),
                'arxiv_id': str(convert_to_json_serializable(paper.get('arxiv_id', ''))),
                'year': str(convert_to_json_serializable(paper.get('year', ''))),
                'categories': str(convert_to_json_serializable(paper.get('categories', ''))),
            })
        
        filtered_embeddings = recommender.embeddings[valid_indices[:5000]]
        
        JOBS[job_id].update({"percent": 50, "stage": 3, "message": f"Clustering {len(filtered_papers)} papers into {n_topics} topics..."})
        
        # Perform clustering
        topic_labels, topic_model = perform_topic_clustering(filtered_papers, filtered_embeddings, n_topics)
        
        JOBS[job_id].update({"percent": 80, "stage": 4, "message": "Generating topic summaries..."})
        
        # Generate summaries
        topics_summary = generate_topic_summary(topic_model, topic_labels, filtered_papers, filtered_embeddings)
        
        RESULTS[job_id] = {
            'topics': topics_summary,
            'total_papers': len(filtered_papers),
            'n_topics': len(topics_summary)
        }
        
        JOBS[job_id].update({"percent": 100, "done": True, "message": "Completed."})
        
    except Exception as e:
        print(f"Topic discovery error: {e}")
        import traceback
        traceback.print_exc()
        JOBS[job_id].update({"done": True, "error": str(e), "message": "Error."})

# ===== Organization extraction utilities and API =====
from collections import defaultdict

def _split_multi_values(value: str) -> list[str]:
    if not value or not isinstance(value, str):
        return []
    parts = re.split(r"[\|;,/\\、；\n]+", value)
    return [p.strip() for p in parts if p and p.strip()]

def _collect_all_orgs(df: pd.DataFrame) -> list[str]:
    org_cols = ['affiliation', 'organizations', 'institution', 'institutions']
    present = [c for c in org_cols if c in df.columns]
    orgs: set[str] = set()
    for col in present:
        col_vals = df[col].fillna("").astype(str)
        for v in col_vals:
            for item in _split_multi_values(v):
                if item:
                    orgs.add(item)
    # return sorted list for readability
    return sorted(orgs)

@app.route('/api/list_organizations')
def list_organizations():
    global recommender
    if recommender is None or not hasattr(recommender, 'df'):
        return jsonify({"error": "Recommender not initialized or no data available"}), 500
    try:
        orgs = _collect_all_orgs(recommender.df)
        return jsonify({
            "count": len(orgs),
            "organizations": orgs
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/export_excel', methods=['POST'])
def export_excel():
    """Export recommendation results to an Excel file"""
    try:
        data = request.get_json()
        recommendations = data.get('recommendations', [])
        query_info = data.get('query_info', {})
        
        if not recommendations:
            return jsonify({"error": "No recommendations to export"}), 400
        
        # Prepare Excel data
        excel_data = []
        for i, paper in enumerate(recommendations, 1):
            excel_data.append({
                'Rank': i,
                'Title': paper.get('title', ''),
                'Authors': paper.get('authors', ''),
                'Abstract': paper.get('abstract', ''),
                'arXiv ID': paper.get('arxiv_id', ''),
                'arXiv URL': f"https://arxiv.org/abs/{paper.get('arxiv_id', '')}" if paper.get('arxiv_id') else '',
                'Similarity Score': paper.get('similarity', 0),
                'Categories': paper.get('categories', ''),
                'Affiliations': paper.get('affiliation', ''),
                'Date': paper.get('date', ''),
                'Year': paper.get('year', ''),
                'Year-Month': paper.get('year_month', ''),
                'Corresponding Email': paper.get('corresponding_email', ''),
                'Country': paper.get('country', ''),
                'Department': paper.get('department', '')
            })
        
        # Create DataFrame
        df = pd.DataFrame(excel_data)
        
        # Create Excel file in memory
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Write recommendation results
            df.to_excel(writer, sheet_name='Recommendations', index=False)
            
            # Write query info sheet
            query_df = pd.DataFrame([
                {'Parameter': 'Workshop Description', 'Value': query_info.get('workshop_description', '')},
                {'Parameter': 'Number of Results', 'Value': query_info.get('top_k', '')},
                {'Parameter': 'Date Range', 'Value': f"{query_info.get('date_min', '')} to {query_info.get('date_max', '')}"},
                {'Parameter': 'Countries', 'Value': ', '.join(query_info.get('countries', []))},
                {'Parameter': 'Categories', 'Value': ', '.join(query_info.get('categories', []))},
                {'Parameter': 'Min Similarity', 'Value': query_info.get('min_similarity', '')},
                {'Parameter': 'Export Time', 'Value': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            ])
            query_df.to_excel(writer, sheet_name='Query Info', index=False)
        
        output.seek(0)
        
        # Build filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"CFC_Paper_Recommendations_{timestamp}.xlsx"
        
        return send_file(
            output,
            as_attachment=True,
            download_name=filename,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
    except Exception as e:
        return jsonify({"error": f"Export failed: {str(e)}"}), 500

@app.route('/api/export_pdf', methods=['POST'])
def export_pdf():
    """Export recommended papers' PDFs as a ZIP archive"""
    try:
        data = request.get_json()
        papers = data.get('papers', [])
        filter_type = data.get('filter', 'all')
        query_info = data.get('query_info', {})
        
        if not papers:
            return jsonify({"error": "No papers to export"}), 400
        
        # Create ZIP file in memory
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add query info file
            query_info_text = f"""CFC Paper PDF Export
Export Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Filter: {filter_type}
Query: {query_info.get('workshop_description', '')}
Total Papers: {len(papers)}

Papers List:
"""
            for i, paper in enumerate(papers, 1):
                query_info_text += f"{i}. {paper.get('title', '')} - {paper.get('arxiv_id', '')}\n"
            
            zip_file.writestr("query_info.txt", query_info_text)
            
            # Collect download errors
            download_errors = []
            
            # Download and add PDF files
            for i, paper in enumerate(papers, 1):
                arxiv_id = paper.get('arxiv_id', '')
                if not arxiv_id:
                    continue
                
                try:
                    # Build arXiv PDF URL
                    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                    
                    # Download PDF
                    response = requests.get(pdf_url, timeout=30)
                    response.raise_for_status()
                    
                    # Build a safe filename
                    safe_title = re.sub(r'[^\w\s-]', '', paper.get('title', f'paper_{i}'))[:50]
                    safe_title = re.sub(r'[-\s]+', '-', safe_title)
                    filename = f"{i:02d}_{safe_title}_{arxiv_id}.pdf"
                    
                    # Add to ZIP
                    zip_file.writestr(filename, response.content)
                    
                except Exception as e:
                    # If a PDF fails to download, record the error and continue
                    error_msg = f"Failed to download {arxiv_id}: {str(e)}\n"
                    download_errors.append(error_msg)
                    continue
            
            # If there are download errors, include an error file
            if download_errors:
                error_content = "Download Errors:\n" + "".join(download_errors)
                zip_file.writestr("download_errors.txt", error_content)
        
        zip_buffer.seek(0)
        
        # Build filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"CFC_Paper_PDFs_{filter_type}_{timestamp}.zip"
        
        return send_file(
            zip_buffer,
            as_attachment=True,
            download_name=filename,
            mimetype='application/zip'
        )
        
    except Exception as e:
        return jsonify({"error": f"PDF export failed: {str(e)}"}), 500

@app.route('/paper2poster', defaults={'path': ''})
@app.route('/paper2poster/<path:path>')
def paper2poster_proxy(path):
    """Proxy paper2poster requests to paper2poster_app"""
    from flask import request as flask_request
    
    # Delegate to the matching handler in paper2poster_app
    if path == '':
        return paper2poster_app.paper2poster_index()
    elif path == 'upload' and flask_request.method == 'POST':
        return paper2poster_app.paper2poster_upload()
    elif path == 'progress':
        return paper2poster_app.paper2poster_progress()
    elif path == 'result':
        return paper2poster_app.paper2poster_result()
    elif path.startswith('download/'):
        filename = path.split('/')[-1]
        return paper2poster_app.paper2poster_download(filename)
    else:
        return "Not Found", 404

# API route proxies
@app.route('/api/paper2poster/upload', methods=['POST'])
def api_paper2poster_upload():
    """Proxy API requests to paper2poster_app"""
    # Copy request data and call paper2poster_app functions directly
    try:
        # Ensure a file was uploaded
        if 'pdf_file' not in request.files:
            return jsonify({"error": "No PDF file uploaded"}), 400
        
        pdf_file = request.files['pdf_file']
        if pdf_file.filename == '':
            return jsonify({"error": "No PDF file selected"}), 400
        
        # Validate file type
        if not pdf_file.filename.lower().endswith('.pdf'):
            return jsonify({"error": "Only PDF files are allowed"}), 400
        
        # Retrieve OpenAI API key
        openai_api_key = request.form.get('openai_api_key', '').strip()
        if not openai_api_key:
            return jsonify({"error": "OpenAI API Key is required"}), 400
        
        # Create job
        job_id = paper2poster_app.uuid.uuid4().hex
        paper2poster_app.JOBS[job_id] = {"percent": 0, "stage": 0, "message": "Queued...", "done": False}
        
        # Save the PDF file first
        pdfs_dir = '/home/nuochen/paper2poster/pdfs'
        os.makedirs(pdfs_dir, exist_ok=True)
        
        pdf_filename = paper2poster_app.secure_filename(pdf_file.filename)
        pdf_path = os.path.join(pdfs_dir, pdf_filename)
        pdf_file.save(pdf_path)
        
        # Prepare job data
        payload = {
            'openai_api_key': openai_api_key,
            'pdf_path': pdf_path,
            'pdf_filename': pdf_filename
        }
        
        # Start background task
        thread = paper2poster_app.threading.Thread(target=paper2poster_app.run_paper2poster_job, args=(job_id, payload), daemon=True)
        thread.start()
        
        return jsonify({"job_id": job_id})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/paper2poster/progress')
def api_paper2poster_progress():
    """Proxy API requests to paper2poster_app"""
    job_id = request.args.get('job_id', '')
    if not job_id or job_id not in paper2poster_app.JOBS:
        return jsonify({"error": "job_id not found"}), 404
    return jsonify(paper2poster_app.JOBS[job_id])

@app.route('/api/paper2poster/result')
def api_paper2poster_result():
    """Proxy API requests to paper2poster_app"""
    job_id = request.args.get('job_id', '')
    if not job_id:
        return jsonify({"error": "job_id required"}), 400
    if job_id not in paper2poster_app.RESULTS:
        return jsonify({"error": "result not ready"}), 404
    return jsonify(paper2poster_app.RESULTS[job_id])

@app.route('/api/paper2poster/download/<filename>')
def api_paper2poster_download(filename):
    """Proxy API requests to paper2poster_app"""
    try:
        # Retrieve actual file info from job_id
        job_id = request.args.get('job_id', '')
        if not job_id or job_id not in paper2poster_app.RESULTS:
            return jsonify({"error": "Job not found"}), 404
        
        result = paper2poster_app.RESULTS[job_id]
        actual_filename = result.get('actual_filename', filename)
        file_path = result.get('path')
        
        if not file_path or not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
        
        # Use the client-facing filename as the download name
        return send_file(file_path, as_attachment=True, download_name=filename)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Initialize paper2poster module
    paper2poster_app.init_paper2poster()
    
    # Create templates
    create_templates()
    
    # Initialize recommender
    print("Initializing Community-Federated Conference Curation...")
    recommender = WorkshopRecommender()
    
    if recommender is None or not hasattr(recommender, 'has_embeddings'):
        print("❌ Initialization failed")
        exit(1)
    
    print("✓ Initialization complete")
    print("🌐 Starting web server... ")
    print("📱 Visit: http://xtraweb1.ddns.comp.nus.edu.sg:4095")
    import logging
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    # Start Flask app
    app.run(host='0.0.0.0', port=4095, debug=True,use_reloader=False)

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
import faiss  # faiss-cpu
from openai import OpenAI


# ----------------------------- Config ---------------------------------

DEFAULT_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
DEFAULT_DB_PATH = os.getenv("EQUICHAT_DB_PATH", "./data/equichat.duckdb")

# Index layout on disk
@dataclass
class IndexPaths:
    root: Path
    faiss_index: Path
    metadata_parquet: Path
    manifest_json: Path

    @staticmethod
    def at(path: str | Path) -> "IndexPaths":
        root = Path(path).expanduser().resolve()
        return IndexPaths(
            root=root,
            faiss_index=root / "index.faiss",
            metadata_parquet=root / "meta.parquet",
            manifest_json=root / "manifest.json",
        )


# ----------------------------- Utilities --------------------------------

def _clean_text(s: str) -> str:
    s = s.replace("\x00", " ")  # null bytes
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _chunk_text(text: str, *, max_chars: int = 800, overlap: int = 150) -> List[str]:
    """
    Simple char-based chunker that respects sentence-ish boundaries.
    """
    text = _clean_text(text)
    if len(text) <= max_chars:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        # try to break on punctuation near the end
        cut = text.rfind(".", start, end)
        if cut == -1 or cut < start + max_chars * 0.6:
            cut = end
        chunk = text[start:cut].strip()
        if chunk:
            chunks.append(chunk)
        start = max(cut - overlap, start + 1)
    return chunks


def _embed_texts(client: OpenAI, texts: List[str], model: str) -> np.ndarray:
    """
    Batch embed texts with OpenAI. Returns float32 np.ndarray shape (N, D).
    """
    # OpenAI Python SDK will chunk internally if needed; keep batches modest
    embs: List[List[float]] = []
    B = 96
    for i in range(0, len(texts), B):
        batch = texts[i : i + B]
        resp = client.embeddings.create(model=model, input=batch)
        # SDK returns list in the same order
        for item in resp.data:
            embs.append(item.embedding)
    arr = np.asarray(embs, dtype=np.float32)
    # normalize for cosine similarity (via inner product on normalized vectors)
    faiss.normalize_L2(arr)
    return arr


# ----------------------------- Build index --------------------------------

def build_faiss_from_duckdb(
    *,
    db_path: str = DEFAULT_DB_PATH,
    out_dir: str | Path = "./data/vec_index",
    embed_model: str = DEFAULT_EMBED_MODEL,
    company_filter: Optional[str] = None,
    ignore_last_page: bool = False,
    min_chars: int = 40,
) -> Dict[str, str]:
    """
    Build a FAISS cosine index from DuckDB raw_pages:
      raw_pages(doc_id, page_num, text, company, source_path)

    - Chunks each page to ~800 chars (150 overlap)
    - Stores FAISS index + a Parquet metadata file (row-aligned)
    """
    paths = IndexPaths.at(out_dir)
    paths.root.mkdir(parents=True, exist_ok=True)

    # 1) Load pages
    conn = duckdb.connect(db_path, read_only=True)
    where = []
    params: List[str] = []
    if company_filter:
        where.append("company ILIKE ?")
        params.append(f"%{company_filter}%")
    sql = f"""
        SELECT doc_id, page_num, company, source_path, text
        FROM raw_pages
        {"WHERE " + " AND ".join(where) if where else ""}
        ORDER BY doc_id, page_num
    """
    df = conn.execute(sql, params).df()
    if ignore_last_page and not df.empty:
        # drop last page per doc
        df["max_page"] = df.groupby("doc_id")["page_num"].transform("max")
        df = df[df["page_num"] < df["max_page"]].drop(columns=["max_page"])

    if df.empty:
        raise RuntimeError("No rows in raw_pages. Ingest text first.")

    # 2) Chunk + build metadata rows
    rows: List[Dict[str, str | int]] = []
    texts: List[str] = []
    for _, r in df.iterrows():
        chunks = _chunk_text(str(r["text"]))
        for j, ch in enumerate(chunks):
            if len(ch) < min_chars:
                continue
            rows.append(
                {
                    "doc_id": r["doc_id"],
                    "company": r["company"],
                    "page_num": int(r["page_num"]),
                    "chunk_idx": j,
                    "source_path": r["source_path"],
                    "text": ch,
                }
            )
            texts.append(ch)

    if not texts:
        raise RuntimeError("No eligible chunks to embed.")

    # 3) Embed
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    mat = _embed_texts(client, texts, model=embed_model)  # (N, D)

    # 4) Build FAISS IP index on normalized vectors (cosine)
    dim = mat.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(mat)  # vector order aligns with 'rows'

    # 5) Save files
    faiss.write_index(index, str(paths.faiss_index))
    pd.DataFrame(rows).to_parquet(paths.metadata_parquet, index=False)
    paths.manifest_json.write_text(
        json.dumps(
            {
                "db_path": str(Path(db_path).resolve()),
                "count": int(mat.shape[0]),
                "dim": int(dim),
                "embed_model": embed_model,
                "built_at": pd.Timestamp.utcnow().isoformat(),
            },
            indent=2,
        )
    )

    return {
        "index": str(paths.faiss_index),
        "meta": str(paths.metadata_parquet),
        "manifest": str(paths.manifest_json),
    }


# ----------------------------- Query index --------------------------------

@dataclass
class Hit:
    score: float
    company: str
    doc_id: str
    page_num: int
    text: str
    source_path: str
    chunk_idx: int


def _load_index(out_dir: str | Path) -> Tuple[faiss.IndexFlatIP, pd.DataFrame, Dict[str, str]]:
    paths = IndexPaths.at(out_dir)
    if not (paths.faiss_index.exists() and paths.metadata_parquet.exists()):
        raise FileNotFoundError(f"Vector index not found under {paths.root}. Build it first.")
    index = faiss.read_index(str(paths.faiss_index))
    meta = pd.read_parquet(paths.metadata_parquet)
    manifest = json.loads(paths.manifest_json.read_text()) if paths.manifest_json.exists() else {}
    return index, meta, manifest


def search(
    query: str,
    *,
    out_dir: str | Path = "./data/vec_index",
    top_k: int = 8,
    embed_model: str = DEFAULT_EMBED_MODEL,
    company_filter: Optional[str] = None,
) -> List[Hit]:
    index, meta, manifest = _load_index(out_dir)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # 1) Embed query
    q_emb = _embed_texts(client, [query], model=embed_model)  # (1, D)

    # 2) Search
    scores, idx = index.search(q_emb, top_k)  # cosine similarities
    ids = idx[0]
    scs = scores[0]

    hits: List[Hit] = []
    for i, score in zip(ids, scs):
        if i < 0:
            continue
        row = meta.iloc[int(i)]
        if company_filter and company_filter.lower() not in str(row["company"]).lower():
            continue
        hits.append(
            Hit(
                score=float(score),
                company=str(row["company"]),
                doc_id=str(row["doc_id"]),
                page_num=int(row["page_num"]),
                text=str(row["text"]),
                source_path=str(row["source_path"]),
                chunk_idx=int(row["chunk_idx"]),
            )
        )
    return hits


# ----------------------------- Answer synthesis (optional) --------------------

SYSTEM_ANS = """You are a precise assistant. Answer the user’s question ONLY from the provided snippets.
Quote the company name if present and include page numbers when available as (p.X).
If the answer isn’t found in the snippets, say “No matching data found.” Be concise (1–2 sentences)."""

def answer_from_hits(question: str, hits: List[Hit], model: str = "gpt-4o-mini") -> str:
    if not hits:
        return "No matching data found."
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # Build a tight context
    ctx_lines = []
    for h in hits:
        ctx_lines.append(f"[{h.company}] (p.{h.page_num}) {h.text}")
    context = "\n\n".join(ctx_lines[:8])

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_ANS},
            {"role": "user", "content": f"Question: {question}\n\nSnippets:\n{context}"},
        ],
        max_output_tokens=250,
    )
    return getattr(resp, "output_text", "").strip()

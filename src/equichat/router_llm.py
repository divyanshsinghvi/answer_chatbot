from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple, Optional

import duckdb
from openai import OpenAI

# Optional deps for vector search (install: faiss-cpu, numpy, pandas)
try:
    import faiss  # type: ignore
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    faiss = None  # type: ignore
    np = None  # type: ignore
    pd = None  # type: ignore


# =========================== SCHEMAS & PROMPTS ================================

FACTS_VIEW_SCHEMA = """
facts(
  doc_id TEXT,
  company TEXT,
  period_type TEXT,     -- 'quarter' | 'year' | 'estimate' | 'snapshot'
  period_end TEXT,      -- ISO date when available, otherwise NULL
  metric_key TEXT,      -- e.g., Revenue, EBITDA, EV/EBITDA, ROE
  metric_variant TEXT,  -- e.g., 'FY24A', 'Q4FY24', 'FY25E'
  value DOUBLE,
  unit TEXT,            -- 'cr','%','x','days','tonne','Rs', etc.
  source_page INT,
  source_span TEXT,     -- breadcrumb like 'page_2.quarterly_financials_q4fy24_cr.revenue'
  confidence DOUBLE,
  rn BIGINT
)
"""

SYSTEM_CLASSIFY = """You are an intent classifier for financial PDF QA.
Given a user question, choose the SINGLE best target store to query:

- "facts": numeric/metric questions (Revenue, EBITDA, PAT, margins, ROE, EV/EBITDA, shareholding %, etc.)
- "vector": narrative, fuzzy, or semantic lookups (e.g., "Where do they mention copper downstream volumes?", risks, commentary, management notes)

Return ONLY JSON:
{"tool":"facts"|"vector"}
"""

SYSTEM_SQL_FACTS = f"""You are a SQL generator for DuckDB over the facts view.

Schema:
{FACTS_VIEW_SCHEMA}

Rules:
- SELECT exactly: company, metric_key, metric_variant, value, unit, source_page
- Use case-insensitive fuzzy matching:
  - company ILIKE '%...%'  (include only if the question mentions a company)
  - metric_key ILIKE '%...%' OR source_span ILIKE '%...%'
- For fiscal years/quarters ALWAYS fuzzy:
  metric_variant ILIKE '%FY24%' (not equals)
- If the user asks for top/compare/aggregate, you may add ORDER BY / GROUP BY,
  but keep the required columns present.
- LIMIT 20.
- Return ONLY the SQL (no prose, no fences).
"""

SYSTEM_ANSWER_FACTS = """You are a financial QA assistant.
Use ONLY the provided rows to answer concisely (1–3 sentences).
Each row is: (company, metric_key, metric_variant, value, unit, source_page).
If rows show different periods or segments, either choose the most relevant or list the top few clearly.
If no rows, say "No matching data found."
"""

SYSTEM_ANSWER_VECTOR = """You are a QA assistant using semantic search results.
You are given rows with: (company, page_num, text, score[, source_path]).
Use ONLY these snippets to answer. Be concise (1–3 sentences). Add page references if useful.
If no rows, say "No matching data found."
"""


# =============================== UTILITIES ====================================

def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*", "", s).strip()
        s = s.replace("```", "").strip()
    return s


def _fix_variant_equals(sql: str) -> str:
    # Replace strict equality on metric_variant to fuzzy ILIKE (FY/quarter labels)
    return re.sub(
        r"metric_variant\s*=\s*'([^']*FY[0-9]{2}[^']*)'",
        r"metric_variant ILIKE '%\1%'",
        sql,
        flags=re.IGNORECASE,
    )


def _rows_to_text_facts(rows: List[Tuple[Any]]) -> str:
    if not rows:
        return "No rows."
    out = []
    for r in rows:
        if len(r) >= 6:
            company, metric_key, metric_variant, value, unit, source_page = r[:6]
            out.append(f"{company} | {metric_key} | {metric_variant} | {value} {unit} | p.{source_page}")
        else:
            out.append(" | ".join(str(x) for x in r))
    return "\n".join(out)


def _rows_to_text_vector(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "No rows."
    lines = []
    for r in rows:
        company = r.get("company", "")
        page = r.get("page_num", "")
        text = r.get("text", r.get("snippet", ""))
        score = r.get("score", "")
        if company:
            lines.append(f"{company} | p.{page} | {text}  [score={score}]")
        else:
            lines.append(f"p.{page} | {text}  [score={score}]")
    return "\n".join(lines)


# =============================== FACTS PATH ===================================

def classify_intent(client: OpenAI, query: str, model: str = "gpt-4o-mini", context: str | None = None) -> str:
    user_payload = f"Conversation context:\n{context}\n\nUser question:\n{query}" if context else query
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_CLASSIFY},
            {"role": "user", "content": user_payload},
        ],
        max_output_tokens=80,
    )
    text = _strip_fences(getattr(resp, "output_text", "").strip())
    m = re.search(r'"tool"\s*:\s*"([^"]+)"', text)
    tool = (m.group(1).strip().lower() if m else "facts")
    if tool not in {"facts", "vector"}:
        # heuristic: default to facts for obvious numeric intents, else vector
        if re.search(r"\b(revenue|sales|ebitda|pat|eps|margin|roe|ev/ebitda|capex|guidance|price target|valuation|sotp)\b", query, re.I):
            tool = "facts"
        else:
            tool = "vector"
    return tool


def generate_sql_facts(client: OpenAI, query: str, model: str = "gpt-4o-mini", context: str | None = None) -> str:
    user_payload = f"Context:\n{context}\n\nQuestion:\n{query}" if context else query
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_SQL_FACTS},
            {"role": "user", "content": user_payload},
        ],
        max_output_tokens=400,
    )
    sql = _strip_fences(getattr(resp, "output_text", "").strip())
    sql = _fix_variant_equals(sql)

    # Ensure required columns are present; if not, wrap defensively
    must = ["company", "metric_key", "metric_variant", "value", "unit"]
    if "SELECT" in sql.upper() and not all(k in sql for k in must):
        sql = f"""
        WITH src AS ({sql})
        SELECT company, metric_key, metric_variant, value, unit, source_page
        FROM src
        LIMIT 20
        """.strip()
    return sql


def run_sql(conn: duckdb.DuckDBPyConnection, sql: str) -> List[Tuple[Any]]:
    try:
        return conn.execute(sql).fetchall()
    except Exception:
        try:
            return conn.execute(sql + "\n LIMIT 20").fetchall()
        except Exception:
            return []


def answer_from_facts(client: OpenAI, query: str, rows: List[Tuple[Any]], model: str = "gpt-4o-mini", context: str | None = None) -> str:
    rows_text = _rows_to_text_facts(rows)
    user_payload = f"Context:\n{context}\n\nQuestion:\n{query}\nRows:\n{rows_text}" if context else f"Question:\n{query}\nRows:\n{rows_text}"
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_ANSWER_FACTS},
            {"role": "user", "content": user_payload},
        ],
        max_output_tokens=300,
    )
    return getattr(resp, "output_text", "").strip()


# ============================== VECTOR PATH ===================================

class VectorIndex:
    """
    Minimal FAISS-backed retriever for your on-disk index at data/vec_index/.
    Requires files:
      - data/vec_index/index.faiss
      - data/vec_index/meta.parquet  (must include at least: text, page_num; optional: company, source_path)
    """
    def __init__(self, folder: str = "data/vec_index"):
        if faiss is None or np is None or pd is None:
            raise RuntimeError("Vector dependencies missing. Install faiss-cpu, numpy, pandas.")
        self.folder = folder
        self.index = faiss.read_index(f"{folder}/index.faiss")
        self.meta = pd.read_parquet(f"{folder}/meta.parquet")
        if "page_num" not in self.meta.columns:
            # try to derive page id if your meta uses another name
            if "page" in self.meta.columns:
                self.meta = self.meta.rename(columns={"page": "page_num"})
            else:
                self.meta["page_num"] = -1
        if "text" not in self.meta.columns:
            # support 'snippet' or 'chunk'
            cand = [c for c in ["snippet", "chunk", "content"] if c in self.meta.columns]
            if cand:
                self.meta = self.meta.rename(columns={cand[0]: "text"})
            else:
                self.meta["text"] = ""

    def search(self, query_emb: "np.ndarray", k: int = 5) -> List[Dict[str, Any]]:
        D, I = self.index.search(query_emb.astype("float32"), k)
        rows: List[Dict[str, Any]] = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            md = self.meta.iloc[int(idx)].to_dict()
            md["score"] = float(dist)
            rows.append(md)
        return rows


def embed_query(client: OpenAI, text: str, model: str = "text-embedding-3-large") -> "np.ndarray":
    if np is None:
        raise RuntimeError("Vector dependencies missing. Install numpy.")
    resp = client.embeddings.create(model=model, input=[text])
    emb = np.array(resp.data[0].embedding, dtype="float32").reshape(1, -1)
    return emb


def answer_from_vector(client: OpenAI, query: str, rows: List[Dict[str, Any]], model: str = "gpt-4o-mini", context: str | None = None) -> str:
    rows_text = _rows_to_text_vector(rows)
    user_payload = f"Context:\n{context}\n\nQuestion:\n{query}\nRows:\n{rows_text}" if context else f"Question:\n{query}\nRows:\n{rows_text}"
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_ANSWER_VECTOR},
            {"role": "user", "content": user_payload},
        ],
        max_output_tokens=350,
    )
    return getattr(resp, "output_text", "").strip()

# =============================== PUBLIC API ===================================

def query_with_llm_router(
    client: OpenAI,
    query: str,
    conn: duckdb.DuckDBPyConnection,
    model: str = "gpt-4o-mini",
    vec_folder: str = "data/vec_index",
    k: int = 5,
    fallback_to_vector_if_empty: bool = True,
    context: str | None = None,
) -> Dict[str, Any]:
    """
    End-to-end:
      1) classify to one of: facts | vector
      2) generate SQL (facts) or embed+search (vector)
      3) run
      4) synthesize short answer from rows

    Returns: {"tool": str, "sql": Optional[str], "rows": list, "answer": str}
    """
    tool = classify_intent(client, query, model=model, context=context)

    if tool == "facts":
        sql = generate_sql_facts(client, query, model=model, context=context)
        rows = run_sql(conn, sql)
        if not rows and fallback_to_vector_if_empty:
            tool = "vector"
        else:
            answer = answer_from_facts(client, query, rows, model=model, context=context)
            return {"tool": "facts", "sql": sql, "rows": rows, "answer": answer}

    # vector route
    if faiss is None:
        raise RuntimeError("faiss-cpu not installed; cannot use vector search.")
    q_emb = embed_query(client, query)
    retriever = VectorIndex(vec_folder)
    vrows = retriever.search(q_emb, k=k)
    answer = answer_from_vector(client, query, vrows, model=model, context=context)
    return {"tool": "vector", "sql": None, "rows": vrows, "answer": answer}

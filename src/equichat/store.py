from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, List

import duckdb
import pandas as pd

from .config import CONFIG
from .schemas import Document, Fact


DDL = """
PRAGMA threads=4;

CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT PRIMARY KEY,
    company     TEXT NOT NULL,
    ticker      TEXT,
    industry    TEXT,
    report_date TEXT,
    source_path TEXT NOT NULL,
    page_count  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS facts (
    doc_id        TEXT NOT NULL,
    company       TEXT NOT NULL,
    period_type   TEXT NOT NULL,
    period_end    TEXT,
    metric_key    TEXT NOT NULL,
    metric_variant TEXT,
    value         DOUBLE NOT NULL,
    unit          TEXT NOT NULL,
    source_page   INTEGER NOT NULL,
    source_span   TEXT,
    confidence    DOUBLE NOT NULL,
    CONSTRAINT facts_basic CHECK (confidence >= 0.0 AND confidence <= 1.0)
);

CREATE TABLE IF NOT EXISTS documents_text (
    doc_id       TEXT NOT NULL,
    company      TEXT NOT NULL,
    page_number  INTEGER NOT NULL,
    text         TEXT NOT NULL
);

-- Helpful indexes
CREATE INDEX IF NOT EXISTS idx_facts_company_metric
    ON facts (company, metric_key);

CREATE INDEX IF NOT EXISTS idx_facts_period
    ON facts (period_type, period_end);

CREATE INDEX IF NOT EXISTS idx_facts_doc
    ON facts (doc_id);

CREATE INDEX IF NOT EXISTS idx_text_company_page
    ON documents_text (company, page_number);
""".strip()


class Store:
    """
    Thin wrapper around DuckDB for storing documents and extracted facts.

    Notes
    -----
    - Pass a file path (e.g., './data/equichat.duckdb') to persist the DB.
      Defaults to CONFIG.db_path which is ':memory:' for tests unless overridden.
    - Use `register_entities_df` to provide a (company -> industry) mapping
      for industry-level top-k queries.
    """

    def __init__(self, db_path: Optional[str] = None, read_only: bool = False) -> None:
        self.db_path = db_path or CONFIG.db_path
        self.read_only = read_only
        cfg = {"threads": "2"}
        if self.read_only and self.db_path != ":memory:":
            cfg["access_mode"] = "READ_ONLY"
        self.conn = duckdb.connect(database=self.db_path, read_only=self.read_only, config=cfg)
        if not self.read_only:
            self.conn.execute(DDL)

    # --- Context manager sugar -------------------------------------------------
    def __enter__(self) -> "Store":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    # --- Session helpers -------------------------------------------------------
    def register_entities_df(self, df: pd.DataFrame, view_name: str = "entities") -> None:
        """
        Register a DataFrame with at least columns: ['company', 'industry'] as a view.
        Required by industry-level aggregation queries (top-k).
        """
        if not {"company", "industry"}.issubset(set(df.columns.str.lower())):
            # normalize columns if needed
            cols = {c: c.lower() for c in df.columns}
            df = df.rename(columns=cols)

        self.conn.unregister(view_name) if view_name in self.conn.list_views() else None
        self.conn.register(view_name, df)

    # --- Document ops ----------------------------------------------------------
    def upsert_document(self, doc: Document) -> None:
        self.conn.execute(
            """
            INSERT INTO documents AS d (doc_id, company, ticker, industry, report_date, source_path, page_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (doc_id) DO UPDATE SET
                company=excluded.company,
                ticker=excluded.ticker,
                industry=excluded.industry,
                report_date=excluded.report_date,
                source_path=excluded.source_path,
                page_count=excluded.page_count
            """,
            [
                doc.doc_id,
                doc.company,
                doc.ticker,
                doc.industry,
                doc.report_date,
                doc.source_path,
                doc.page_count,
            ],
        )

    # --- Facts ops -------------------------------------------------------------
    def insert_facts(self, facts: Iterable[Fact]) -> int:
        """
        Bulk insert facts. Returns number of inserted rows.
        """
        rows: List[Tuple] = []
        for f in facts:
            rows.append(
                (
                    f.doc_id,
                    f.company,
                    f.period_type,
                    f.period_end,
                    f.metric_key,
                    f.metric_variant,
                    float(f.value),
                    f.unit,
                    int(f.source_page),
                    f.source_span,
                    float(f.confidence),
                )
            )
        if not rows:
            return 0

        self.conn.executemany(
            """
            INSERT INTO facts (
                doc_id, company, period_type, period_end,
                metric_key, metric_variant, value, unit,
                source_page, source_span, confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return len(rows)

    # --- Lookups & aggregations -----------------------------------------------
    def latest_metric(
        self,
        company: str,
        metric_key: str,
        prefer_period: Optional[str] = None,  # 'quarter' | 'year' | 'asof'
    ) -> Optional[Tuple[str, str, float, str, str, Optional[str], int, float]]:
        """
        Get the latest metric for a company.

        Returns tuple:
            (company, metric_key, value, unit, period_type, period_end, source_page, confidence)
        or None if not found.
        """
        if prefer_period:
            q = """
            WITH candidate AS (
              SELECT *
              FROM facts
              WHERE company = ? AND metric_key = ? AND period_type = ?
            ),
            ranked AS (
              SELECT *, ROW_NUMBER() OVER (ORDER BY COALESCE(period_end, '9999-12-31') DESC) AS rn
              FROM candidate
            )
            SELECT company, metric_key, value, unit, period_type, period_end, source_page, confidence
            FROM ranked
            WHERE rn = 1
            """
            row = self.conn.execute(q, [company, metric_key, prefer_period]).fetchone()
            if row:
                return row

        # Fallback: any period, pick most recent by period_end
        q2 = """
        WITH candidate AS (
          SELECT *
          FROM facts
          WHERE company = ? AND metric_key = ?
        ),
        ranked AS (
          SELECT *, ROW_NUMBER() OVER (ORDER BY COALESCE(period_end, '9999-12-31') DESC) AS rn
          FROM candidate
        )
        SELECT company, metric_key, value, unit, period_type, period_end, source_page, confidence
        FROM ranked
        WHERE rn = 1
        """
        return self.conn.execute(q2, [company, metric_key]).fetchone()

    def topk(
        self,
        industry: str,
        metric_key: str,
        k: int = 3,
        prefer_period: Optional[str] = "quarter",
    ) -> List[Tuple[str, float, str, Optional[str]]]:
        """
        Top-K companies in an industry by the metric (latest per company).

        Requires a registered 'entities(company, industry)' view in the session.

        Returns list of tuples: (company, value, unit, period_end)
        """
        # Pick latest row per company for the metric (optionally filter to prefer_period)
        where_period = "AND f.period_type = ?" if prefer_period else ""
        params: List = [industry, metric_key]
        if prefer_period:
            params.append(prefer_period)
        params.append(k)

        q = f"""
        WITH metric_latest AS (
          SELECT
            f.company,
            f.value,
            f.unit,
            f.period_end,
            ROW_NUMBER() OVER (
              PARTITION BY f.company
              ORDER BY COALESCE(f.period_end, '9999-12-31') DESC
            ) AS rn
          FROM facts f
          JOIN entities e ON lower(e.company) = lower(f.company)
          WHERE lower(e.industry) = lower(?) AND f.metric_key = ?
          {where_period}
        )
        SELECT company, value, unit, period_end
        FROM metric_latest
        WHERE rn = 1
        ORDER BY value DESC
        LIMIT ?
        """
        return self.conn.execute(q, params).fetchall()


    # --- Text ops ---------------------------------------------------------
    def insert_page_text(self, doc_id: str, company: str, page_number: int, text: str) -> None:
        """
        Insert raw page text for unstructured Q&A.
        """
        self.conn.execute(
            """
            INSERT INTO documents_text (doc_id, company, page_number, text)
            VALUES (?, ?, ?, ?)
            """,
            [doc_id, company, page_number, text],
        )

    def bulk_insert_text(self, rows: List[Tuple[str, str, int, str]]) -> int:
        """
        Bulk insert many (doc_id, company, page_number, text).
        Returns number of rows inserted.
        """
        if not rows:
            return 0
        self.conn.executemany(
            "INSERT INTO documents_text (doc_id, company, page_number, text) VALUES (?, ?, ?, ?)",
            rows,
        )
        return len(rows)

    def search_text(self, company: str, keyword: str, limit: int = 5) -> List[Tuple[int, str]]:
        """
        Simple keyword search inside a company's text pages.
        Returns list of (page_number, snippet).
        """
        q = """
        SELECT page_number, text
        FROM documents_text
        WHERE company ILIKE ? AND text ILIKE ?
        LIMIT ?
        """
        return self.conn.execute(q, [f"%{company}%", f"%{keyword}%", limit]).fetchall()
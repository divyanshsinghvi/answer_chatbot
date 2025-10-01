#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
from typing import Optional

import pandas as pd

from equichat.store import Store


def main() -> int:
    """
    Dump all extracted facts from the DuckDB store in READ-ONLY mode.
    This avoids file-lock conflicts with the running API server.
    Set EQUICHAT_DB_PATH to your .duckdb file if not already set.
    """
    db_path = os.getenv("EQUICHAT_DB_PATH", ":memory:")
    # Open read-only to avoid locking
    with Store(db_path, read_only=True) as store:
        try:
            df = store.conn.execute(
                """
                SELECT
                  company,
                  metric_key,
                  value,
                  unit,
                  period_type,
                  period_end,
                  source_page,
                  substr(source_span, 1, 160) AS snippet
                FROM facts
                ORDER BY company, metric_key, COALESCE(period_end, '9999-12-31')
                """
            ).df()
        except Exception as e:
            print(f"Query failed: {e}", file=sys.stderr)
            return 2

    if df.empty:
        print("No facts in DB. Ingest PDFs first (POST /ingest or your ingest script).")
        return 0

    # Pretty print
    pd.set_option("display.max_colwidth", 160)
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
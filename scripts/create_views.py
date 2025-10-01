#!/usr/bin/env python3
from equichat.store import Store
from equichat.config import CONFIG

with Store(CONFIG.db_path, read_only=False) as store:
    store.conn.execute("""
    CREATE OR REPLACE VIEW v_facts_best AS
    SELECT *
    FROM (
      SELECT
        f.*,
        ROW_NUMBER() OVER (
          PARTITION BY company, metric_key, COALESCE(metric_variant, '')
          ORDER BY confidence DESC, source_page ASC
        ) AS rn
      FROM facts f
    )
    WHERE rn = 1;
    """)
    print("✅ Created view v_facts_best")


with Store(CONFIG.db_path, read_only=False) as store:
    store.conn.execute("""
    CREATE OR REPLACE VIEW v_rows_flat AS
    SELECT
      source_page AS "page number",
      metric_key  AS metric,
      metric_variant AS time_period,
      value,
      unit
    FROM v_facts_best;
    """)
    print("✅ View v_rows_flat created")
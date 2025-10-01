from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import duckdb

from .config import CONFIG
from .schemas import QueryResult
from .store import Store


# -----------------------------
# Metric lookup (single company)
# -----------------------------
def metric_lookup(
    store: Store,
    company: str,
    metric_key: str,
    prefer_period: Optional[str] = None,  # "quarter" | "year" | "asof"
) -> QueryResult:
    """
    Look up the latest value of `metric_key` for `company`.
    Returns a QueryResult with provenance fields.
    """
    row = store.latest_metric(company=company, metric_key=metric_key, prefer_period=prefer_period)
    if not row:
        return QueryResult(
            status="not_available",
            company=company,
            metric_key=metric_key,
            message=f"Metric '{metric_key}' not reported for '{company}'.",
        )

    comp, key, value, unit, period_type, period_end, source_page, confidence = row
    if confidence < CONFIG.confidence_threshold:
        return QueryResult(
            status="uncertain",
            company=comp,
            metric_key=key,
            value=value,
            unit=unit,
            period_type=period_type,
            period_end=period_end,
            source_page=source_page,
            confidence=confidence,
            message="Low-confidence extraction. Consider verifying the source page.",
        )

    return QueryResult(
        status="ok",
        company=comp,
        metric_key=key,
        value=value,
        unit=unit,
        period_type=period_type,
        period_end=period_end,
        source_page=source_page,
        confidence=confidence,
    )


# -----------------------------------
# Industry top-K (latest per company)
# -----------------------------------
def industry_topk(
    store: Store,
    industry: str,
    metric_key: str,
    k: int = 3,
    prefer_period: Optional[str] = "quarter",
) -> QueryResult:
    """
    Top-K companies by metric within `industry`.
    Requires a registered DuckDB view named 'entities(company, industry)'.
    """
    # Ensure the 'entities' view exists
    try:
        _ = store.conn.execute("DESCRIBE entities").fetchall()
    except duckdb.CatalogException:
        return QueryResult(
            status="not_available",
            message="Industry map not loaded. Register a DataFrame as 'entities(company, industry)'.",
        )

    rows = store.topk(industry=industry, metric_key=metric_key, k=k, prefer_period=prefer_period)
    if not rows:
        return QueryResult(
            status="not_available",
            message=f"No rows found for industry='{industry}', metric='{metric_key}'.",
        )

    payload = [{"company": r[0], "value": r[1], "unit": r[2], "period_end": r[3]} for r in rows]
    return QueryResult(status="ok", rows=payload, metric_key=metric_key)


# -------------------------------------------------------------
# Filtered aggregation (e.g., "top 3 revenue where mcap >= 1000")
# -------------------------------------------------------------
def filtered_agg_topk(
    store: Store,
    industry: str,
    metric_key: str,
    filters: Dict[str, Tuple[str, float]],  # e.g., {"market_cap": (">=", 1000.0)}
    k: int = 3,
    prefer_period: Optional[str] = "quarter",
) -> QueryResult:
    """
    Top-K by `metric_key` in `industry`, after applying metric-based filters like:
      filters={"market_cap": (">=", 1000.0)}
    Assumes metrics in ₹ crore and entities view registered.
    """
    # Verify entities view
    try:
        _ = store.conn.execute("DESCRIBE entities").fetchall()
    except duckdb.CatalogException:
        return QueryResult(
            status="not_available",
            message="Industry map not loaded. Register a DataFrame as 'entities(company, industry)'.",
        )

    # Build dynamic WHERE for filter metrics by joining each as its latest value
    joins_sql = []
    where_sql = []
    params: List[Any] = []

    # Base CTE: latest metric per company (subject metric)
    where_period_subject = "AND f.period_type = ?" if prefer_period else ""
    if prefer_period:
        params.append(prefer_period)

    # For each filter metric, select latest value per company (any period unless a better policy is needed)
    i = 0
    for filt_metric, (op, threshold) in filters.items():
        alias = f"filt{i}"
        joins_sql.append(
            f"""
            LEFT JOIN (
              SELECT company, value AS v
              FROM (
                SELECT company, value,
                       ROW_NUMBER() OVER (PARTITION BY company ORDER BY COALESCE(period_end, '9999-12-31') DESC) AS rn
                FROM facts
                WHERE metric_key = ?
              ) t WHERE rn = 1
            ) {alias} ON lower({alias}.company) = lower(f.company)
            """
        )
        if op not in {">", ">=", "<", "<=", "=", "==", "!="}:
            return QueryResult(status="unknown", message=f"Unsupported operator '{op}' in filters.")
        # normalize "==" to "=" for SQL
        if op == "==":
            op = "="
        where_sql.append(f"COALESCE({alias}.v, NULL) {op} ?")
        params.insert(0, filt_metric)  # metric key param should go before thresholds so we append thresholds later
        params.append(threshold)
        i += 1

    sql = f"""
    WITH subject AS (
      SELECT f.company, f.value, f.unit, f.period_end,
             ROW_NUMBER() OVER (PARTITION BY f.company ORDER BY COALESCE(f.period_end, '9999-12-31') DESC) AS rn
      FROM facts f
      JOIN entities e ON lower(e.company) = lower(f.company)
      WHERE lower(e.industry) = lower(?) AND f.metric_key = ?
      {where_period_subject}
    )
    SELECT s.company, s.value, s.unit, s.period_end
    FROM subject s
    JOIN entities e ON lower(e.company) = lower(s.company)
    {"".join(joins_sql)}
    WHERE s.rn = 1
      AND {" AND ".join(where_sql) if where_sql else "1=1"}
    ORDER BY s.value DESC
    LIMIT ?
    """

    # params order: industry, metric_key, [prefer_period?], [filter metric keys...], [filter thresholds...], k
    call_params: List[Any] = [industry, metric_key]
    if prefer_period:
        call_params.append(prefer_period)
    # Insert metric_key params for each filter in the same order we created joins_sql
    # (we put them at the front of `params` earlier to keep threshold pairs after).
    # But since we appended both metric names and thresholds into `params`, we only need to extend here:
    call_params.extend(params)
    call_params.append(k)

    try:
        rows = store.conn.execute(sql, call_params).fetchall()
    except Exception as e:
        return QueryResult(status="unknown", message=f"Aggregation failed: {e}")

    if not rows:
        return QueryResult(
            status="not_available",
            message="No companies matched the filters for this industry/metric.",
        )

    payload = [{"company": r[0], "value": r[1], "unit": r[2], "period_end": r[3]} for r in rows]
    return QueryResult(status="ok", rows=payload, metric_key=metric_key)

# src/equichat/facts_query.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .store import Store
from .config import CONFIG

def _metric_predicates(metric_keys: Sequence[str]) -> Tuple[str, List[str]]:
    expanded = []
    for k in metric_keys:
        if k.lower() == "revenue":
            expanded += ["Revenue", "Sales"]
        else:
            expanded.append(k)
    conds, params = [], []
    for k in expanded:
        conds.append("lower(metric_key)=lower(?)")
        params.append(k)
    return "(" + " OR ".join(conds) + ")", params

def _period_predicate(period_label: Optional[str]) -> Tuple[str, List[str]]:
    if not period_label:
        return ("", [])
    like = f"%{period_label}%"
    return ("(metric_variant ILIKE ? OR source_span ILIKE ?)", [like, like])

def query_facts_with_store(
    store: Store,
    text: str,
    metric_keys: Sequence[str],
    company_hint: Optional[str],
    period_label: Optional[str],
    limit: int = 5,
) -> List[Dict[str, Any]]:
    where, params = [], []

    if company_hint:
        where.append("company ILIKE ?")
        params.append(f"%{company_hint}%")

    mcond, mparams = _metric_predicates(metric_keys)
    where.append(mcond)
    params += mparams

    pcond, pparams = _period_predicate(period_label)
    if pcond:
        where.append(pcond)
        params += pparams

    where_sql = " AND ".join(where) if where else "1=1"

    sql = f"""
    SELECT
        company,
        metric_key,
        metric_variant,
        value,
        unit,
        period_type,
        period_end,
        source_page,
        source_span,
        confidence
    FROM facts
    WHERE {where_sql}
    ORDER BY confidence DESC, source_page ASC
    LIMIT {int(limit)}
    """

    rows = store.conn.execute(sql, params).fetchall()
    cols = [c[0] for c in store.conn.description]
    return [dict(zip(cols, r)) for r in rows]

# Optional convenience (DON'T use inside FastAPI since it opens a second connection):
def query_facts(
    text: str,
    metric_keys: Sequence[str],
    company_hint: Optional[str],
    period_label: Optional[str],
    limit: int = 5,
) -> List[Dict[str, Any]]:
    from .store import Store  # local import to avoid cycles
    with Store(CONFIG.db_path, read_only=True) as tmp_store:
        return query_facts_with_store(
            tmp_store, text, metric_keys, company_hint, period_label, limit
        )

def answer_text_for_cli(rows: List[Dict[str, Any]], question: str) -> str:
    if not rows:
        return "No matching facts found. Try re-ingesting or broaden the query."
    if len(rows) == 1:
        r = rows[0]
        tp = r.get("metric_variant") or r.get("period_type") or ""
        tp = f" ({tp})" if tp else ""
        return f"{r['company']}: {r['metric_key']}{tp} = {r['value']} {r['unit']} (p.{r['source_page']})"
    lead = rows[0]
    msg = [f"{lead['company']}: {lead['metric_key']} ({lead.get('metric_variant','')}) = {lead['value']} {lead['unit']} (p.{lead['source_page']})"]
    for r in rows[1:]:
        msg.append(f"- {r['metric_key']} ({r.get('metric_variant','')}) = {r['value']} {r['unit']} (p.{r['source_page']})")
    return "\n".join(msg)

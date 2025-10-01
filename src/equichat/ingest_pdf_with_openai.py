from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import OpenAI

from .config import CONFIG
from .store import Store


# ----------------------------- Public API -------------------------------------


def ingest_pdf_with_openai(
    pdf_path: str,
    company_hint: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Parse a financial PDF using the OpenAI Responses + Files API and persist normalized 'facts' to DuckDB.

    Returns a small dict (for API responses) like:
      {
        "doc_id": "abcd1234...",
        "company": "Hindalco Industries",
        "page_count": <best-effort>,
        "source_path": "/abs/path/to/file.pdf",
        "facts_inserted": 42
      }
    """
    pdf = Path(pdf_path).expanduser().resolve()
    if not pdf.exists():
        raise FileNotFoundError(f"File not found: {pdf}")

    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    # 1) Call OpenAI to get structured JSON for ALL metrics (your working schema)
    client = OpenAI(api_key=api_key)
    result = _extract_all_metrics_from_pdf(client, pdf, model=model)

    # 2) Normalize to facts rows
    doc_id = file_doc_id(pdf.as_posix())
    company = _resolve_company(result, company_hint=company_hint)
    unit_hint = (result.get("document_meta") or {}).get("unit_hint")

    facts = _flatten_schema_to_facts(
        result_json=result,
        doc_id=doc_id,
        company=company,
        unit_hint=unit_hint,
    )

    # 3) Persist to DuckDB
    with Store(CONFIG.db_path, read_only=False) as store:
        # clean old rows for this document (idempotent re-ingest)
        store.conn.execute("DELETE FROM facts WHERE doc_id = ?", [doc_id])
        if facts:
            _bulk_insert_facts(store, facts)

    return {
        "doc_id": doc_id[:16],
        "company": company,
        "page_count": _best_effort_page_count(result),
        "source_path": str(pdf),
        "facts_inserted": len(facts),
    }


# --------------------------- OpenAI extraction --------------------------------


SYSTEM = (
    "You are a meticulous financial data extractor. Read the attached PDF (tables FIRST, then text). "
    "Return ONLY a single JSON object conforming to the user's schema. Do NOT invent values. "
    "If a field is not present, omit it and add the field name to 'missing'. Always include 'provenance' "
    "with page numbers and a brief quote. Use negatives for bracketed numbers (e.g., (158) => -158). "
    "Keep units consistent with the table (₹ crore)."
)

def _schema_text() -> str:
    # Your working schema exactly as provided
    return r"""
Return a single JSON object with the following top-level keys.
If a subsection is not present in the PDF, omit that subsection and list its name in "missing".

{
  "document_meta": {
    "company": string,
    "report_date": string,
    "report_period": string,
    "currency": "INR" | "USD" | "Other",
    "unit_hint": string
  },

  "page_1": {
    "general_market_data": {
      "cmp_as_of": string,
      "cmp_rs": number,
      "target_price_rs": number,
      "target_return_pct": number,
      "market_cap_cr": number,
      "enterprise_value_cr": number,
      "outstanding_shares_cr": number,
      "free_float_pct": number,
      "dividend_yield_pct": number,
      "fifty_two_week_high_rs": number,
      "fifty_two_week_low_rs": number,
      "avg_6m_volume_lacs": number,
      "beta": number,
      "face_value_rs": number
    },
    "shareholding_data_pct_q4fy24": {
      "promoters": number,
      "fiis": number,
      "mfs_institutions": number,
      "public": number,
      "others": number,
      "total": number,
      "promoters_pledge": number
    },
    "price_performance_pct": {
      "absolute_1y": number,
      "sensex_1y": number,
      "relative_1y": number,
      "absolute_6m": number?,
      "relative_6m": number?,
      "absolute_3m": number?,
      "relative_3m": number?
    },
    "consolidated_financials_annual_fy24a": {
      "sales_cr": number,
      "sales_growth_pct": number,
      "ebitda_cr": number,
      "ebitda_margin_pct": number,
      "adj_pat_cr": number,
      "adj_pat_growth_pct": number,
      "adj_eps_rs": number,
      "pe_x": number,
      "pb_x": number,
      "ev_ebitda_x": number,
      "roe_pct": number,
      "de_ratio_x": number,
      "projections_available_for": [string]
    },
    "narrative_highlights_q4fy24": {
      "consolidated_revenue_cr": number,
      "ebitda_cr": number,
      "ebitda_growth_yoy_pct": number,
      "ebitda_margin_pct": number,
      "ebitda_margin_improvement_bps": number,
      "adjusted_pat_cr": number,
      "adjusted_pat_growth_yoy_pct": number,
      "novelis_revenue_cr": number,
      "novelis_revenue_growth_yoy_pct": number,
      "copper_business_revenue_cr": number,
      "copper_business_revenue_growth_yoy_pct": number,
      "copper_sales_volume_tonne": number,
      "copper_sales_volume_growth_yoy_pct": number,
      "aluminium_business_revenue_cr": number,
      "aluminium_business_revenue_growth_yoy_pct": number,
      "aluminium_upstream_volume_tonne": number,
      "aluminium_upstream_volume_growth_yoy_pct": number,
      "aluminium_downstream_volume_tonne": number,
      "aluminium_downstream_volume_growth_yoy_pct": number,
      "cost_of_sales_cr": number,
      "cost_of_sales_decline_yoy_pct": number
    }
  },

  "page_2": {
    "quarterly_financials_q4fy24_cr": {
      "revenue": number,
      "ebitda": number,
      "ebitda_margin_pct": number,
      "depreciation": number,
      "ebit": number,
      "interest": number,
      "other_income_cr": number,
      "pbt": number,
      "tax": number,
      "reported_pat": number,
      "pat_attrib_shareholders": number,
      "adj_pat": number,
      "num_shares_cr": number,
      "adj_eps_rs": number
    },
    "changes_in_estimates_pct": {
      "fy25e_revenue_change_pct": number,
      "fy25e_ebitda_change_pct": number,
      "fy25e_margins_change_bps": number,
      "fy25e_adj_pat_change_pct": number,
      "fy25e_adj_eps_change_pct": number,
      "fy26e_revenue_change_pct": number?,
      "fy26e_ebitda_change_pct": number?,
      "fy26e_margins_change_bps": number?,
      "fy26e_adj_pat_change_pct": number?,
      "fy26e_adj_eps_change_pct": number?
    },
    "sotp_valuation_rs_per_share": {
      "aluminium": number,
      "copper": number,
      "novelis": number,
      "net_debt": number,
      "quoted_investments": number,
      "target_sotp_value": number
    }
  },

  "page_3": {
    "financial_ratios_fy24a": {
      "ebitda_margin_pct": number,
      "net_profit_margin_pct": number,
      "roe_pct": number,
      "roce_pct": number,
      "receivables_days": number,
      "inventory_days": number,
      "payables_days": number,
      "current_ratio_x": number,
      "quick_ratio_x": number,
      "gross_asset_turnover_x": number,
      "total_asset_turnover_x": number,
      "interest_coverage_x": number,
      "adj_debt_equity_x": number,
      "ev_sales_x": number,
      "ev_ebitda_x": number,
      "pe_x": number,
      "pbv_x": number
    },
    "cash_flow_fy24a_cr": {
      "net_income_plus_depn": number,
      "changes_in_working_capital": number,
      "cash_flow_operations": number,
      "capital_expenditure": number,
      "cash_flow_investment": number,
      "cash_flow_finance": number,
      "change_in_cash": number,
      "closing_cash": number
    }
  },

  "provenance": [
    {"metric": string, "page": integer, "quote": string}
  ],

  "missing": [string],
  "warnings": [string]
}
""".strip()


def _user_instructions() -> str:
    return (
        "Extract ALL metrics present in the PDF into the schema below. Prefer tabular values. "
        "If a subsection does not exist, omit it and add the subsection key to 'missing'. "
        "Return ONLY the JSON.\n\nSchema:\n" + _schema_text()
    )


def _extract_all_metrics_from_pdf(client: OpenAI, pdf: Path, model: str) -> Dict[str, Any]:
    uploaded = client.files.create(file=open(pdf, "rb"), purpose="user_data")
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": SYSTEM}]},
            {
                "role": "user",
                "content": [
                    {"type": "input_file", "file_id": uploaded.id},
                    {"type": "input_text", "text": _user_instructions()},
                ],
            },
        ],
        max_output_tokens=2500,
    )
    text = getattr(resp, "output_text", None)
    if not text:
        # fallback gather
        parts: List[str] = []
        out = getattr(resp, "output", None)
        if isinstance(out, list):
            for it in out:
                if getattr(it, "type", None) == "output_text":
                    parts.append(getattr(it, "text", ""))
        text = "\n".join(parts).strip()
    if not text:
        raise RuntimeError("Model returned no text")

    # robust JSON parse (strip fences, then parse first {...})
    s = text.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, re.S | re.I)
    if m:
        s = m.group(1).strip()
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.S)
        if not m:
            raise
        return json.loads(m.group(0))


# ---------------------------- Flatten to facts --------------------------------


@dataclass
class FactRow:
    doc_id: str
    company: str
    metric_key: str
    metric_variant: Optional[str]  # we'll place the time_period label here (e.g., FY24A, Q4FY24)
    value: float
    unit: str
    period_type: str  # unknown | quarter | year | estimate | snapshot
    period_end: Optional[str]  # keep None; can compute later if needed
    source_page: int
    source_span: str
    confidence: float


_METRIC_UNIT_HINTS = {
    "_pct": "%",
    "_x": "x",
    "_rs": "Rs",
    "_days": "days",
    "_tonne": "tonne",
}

_TABLEY_SECTIONS = (
    "quarterly_financials",
    "financial_ratios",
    "cash_flow",
    "consolidated_financials_annual",
    "sotp_valuation",
)

def _resolve_company(result: Dict[str, Any], company_hint: Optional[str]) -> str:
    meta = result.get("document_meta") or {}
    if meta.get("company"):
        return str(meta["company"]).strip()
    if company_hint:
        return company_hint
    # fallback from file name if nothing else
    return "Unknown"


def _normalize_unit(unit_hint: Optional[str], key: str) -> str:
    # Explicit unit by suffix
    for suf, u in _METRIC_UNIT_HINTS.items():
        if key.endswith(suf):
            return u
    # Default from unit_hint (prefer crore)
    if unit_hint:
        u = unit_hint.lower()
        if "crore" in u or "cr" in u:
            return "cr"
        if "million" in u or "mn" in u:
            return "mn"
        if "billion" in u or "bn" in u:
            return "bn"
        return unit_hint
    return "cr"


_PERIOD_PAT = re.compile(r"(q\d+fy\d{2}|fy\d{2}[a-z]?)", re.I)

def _guess_period_type_and_label(container_key: str, field_key: str) -> Tuple[str, Optional[str]]:
    """
    Heuristics:
      - look for tokens like q4fy24, fy24a, fy25e in either the section or the field key
      - map to period_type roughly: quarter, year, estimate, or snapshot
    """
    text = f"{container_key}_{field_key}".lower()
    m = _PERIOD_PAT.search(text)
    if not m:
        # snapshot / point-in-time
        return "snapshot", None
    label = m.group(1).upper()
    if label.startswith("Q"):
        return "quarter", label
    if label.endswith("E"):
        return "estimate", label
    return "year", label


def _humanize_metric_name(field_key: str) -> str:
    # convert snake-like keys to human labels (lightly)
    k = field_key
    k = re.sub(r"_(cr|pct|x|rs|days|tonne)$", "", k, flags=re.I)
    k = k.replace("_yoy", " YoY")
    k = k.replace("_bps", " bps")
    k = k.replace("_per_share", " per share")
    return re.sub(r"[_\-]+", " ", k).strip().title()


def _container_confidence(container_name: str) -> float:
    # Tables score higher than narrative sections
    for t in _TABLEY_SECTIONS:
        if container_name.startswith(t):
            return 0.88
    if "narrative" in container_name:
        return 0.72
    if "market_data" in container_name or "shareholding" in container_name:
        return 0.78
    return 0.75


def _flatten_schema_to_facts(
    result_json: Dict[str, Any],
    doc_id: str,
    company: str,
    unit_hint: Optional[str],
) -> List[FactRow]:
    facts: List[FactRow] = []

    # Page containers -> numeric page number
    page_map = {"page_1": 1, "page_2": 2, "page_3": 3}

    for page_key, page_num in page_map.items():
        page_obj = result_json.get(page_key)
        if not isinstance(page_obj, dict):
            continue

        for container_key, container in page_obj.items():
            if not isinstance(container, dict):
                continue

            conf = _container_confidence(container_key)

            for field_key, val in container.items():
                if val is None:
                    continue
                # Skip non-numeric fields (e.g., projections_available_for arrays)
                if isinstance(val, (list, dict, str)):
                    # Allow strings only if numeric-looking (rare). Mostly skip.
                    try:
                        fval = float(val)  # might raise
                    except Exception:
                        continue
                else:
                    try:
                        fval = float(val)
                    except Exception:
                        continue

                unit = _normalize_unit(unit_hint, field_key)
                period_type, period_label = _guess_period_type_and_label(container_key, field_key)
                metric_name = _humanize_metric_name(field_key)

                facts.append(
                    FactRow(
                        doc_id=doc_id,
                        company=company,
                        metric_key=metric_name,
                        metric_variant=period_label,  # carry time label here
                        value=float(fval),
                        unit=unit,
                        period_type=period_type,
                        period_end=None,
                        source_page=page_num,
                        source_span=f"{page_key}.{container_key}.{field_key}",
                        confidence=conf,
                    )
                )

    # Also try to put a few provenance-only metrics if present (e.g., unit sanity)
    # Not needed for DB facts.

    return facts


# ------------------------------- Persistence ----------------------------------


def _bulk_insert_facts(store: Store, facts: List[FactRow]) -> None:
    # Ensure the facts table exists (Store.__init__ should install DDL already)
    data = [
        (
            f.doc_id,
            f.company,
            f.metric_key,
            f.metric_variant,
            f.value,
            f.unit,
            f.period_type,
            f.period_end,
            f.source_page,
            f.source_span,
            f.confidence,
        )
        for f in facts
    ]
    store.conn.executemany(
        """
        INSERT INTO facts (
            doc_id, company, metric_key, metric_variant,
            value, unit, period_type, period_end,
            source_page, source_span, confidence
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        data,
    )


# ------------------------------- Utilities ------------------------------------


def file_doc_id(path: str) -> str:
    h = hashlib.sha1()
    h.update(path.encode("utf-8"))
    return h.hexdigest()


def _best_effort_page_count(result_json: Dict[str, Any]) -> int:
    # We don't have true count from API; infer from which page_* keys appeared
    count = 0
    for k in ("page_1", "page_2", "page_3", "page_4", "page_5"):
        if isinstance(result_json.get(k), dict):
            count += 1
    return count or 1

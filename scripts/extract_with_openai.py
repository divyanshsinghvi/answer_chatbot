#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

from openai import OpenAI

console = Console()


# --------- The schema we want the model to fill (as text in the prompt) ---------
def full_schema_text() -> str:
    return r"""
Return a single JSON object with the following top-level keys.
If a subsection is not present in the PDF, omit that subsection and list its name in "missing".

{
  "document_meta": {
    "company": string,
    "report_date": string,             // e.g. "2024-06-24" or "24 Jun 2024"
    "report_period": string,           // e.g. "Q4FY24"
    "currency": "INR" | "USD" | "Other",
    "unit_hint": string                // e.g. "₹ crore"
  },

  "page_1": {
    "general_market_data": {
      "cmp_as_of": string,             // e.g. "2024-06-21" if shown near CMP
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
      "absolute_6m": number?,          // include if shown
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
      "projections_available_for": [string]    // e.g. ["FY25E","FY26E"]
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
      "other_income_cr": number,       // use negative if shown in parentheses
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
      "net_debt": number,              // negative if represented in brackets
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
      "capital_expenditure": number,   // negative if shown in parentheses
      "cash_flow_investment": number,  // negative if shown in parentheses
      "cash_flow_finance": number,     # negative if shown in parentheses
      "change_in_cash": number,        # negative if shown in parentheses
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


SYSTEM = """You are a meticulous financial data extractor. Read the attached PDF (tables FIRST, then text).
Return ONLY a single JSON object conforming to the user's schema. Do NOT invent values.
If a field is not present, omit it and add the field name to 'missing'. Always include
'provenance' for metrics you extract (page number + a brief source phrase). Use negatives
for bracketed numbers (e.g., (158) => -158). Keep units consistent with the table (₹ crore)."""


USER_INSTRUCTIONS = """Extract ALL metrics present in the PDF into the schema below. Prefer tabular values.
If a subsection does not exist, omit it and add the subsection key to 'missing'. Return ONLY the JSON.

Schema:
""" + full_schema_text()


# ---------------- JSON parsing helpers (robust to code fences) ----------------
def parse_model_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    # Strip code fences if present
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S | re.I)
    if m:
        text = m.group(1).strip()
    # Try parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Last-resort: find first {...}
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        return json.loads(m.group(0))
    raise ValueError("Could not parse JSON from model output")


# ---------------------------- OpenAI call -------------------------------------
def extract_all_from_pdf(client: OpenAI, pdf_path: Path, model: str) -> Dict[str, Any]:
    uploaded = client.files.create(file=open(pdf_path, "rb"), purpose="user_data")

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": SYSTEM}]},
            {
                "role": "user",
                "content": [
                    {"type": "input_file", "file_id": uploaded.id},
                    {"type": "input_text", "text": USER_INSTRUCTIONS},
                ],
            },
        ],
        max_output_tokens=2500,
    )

    text = getattr(resp, "output_text", None)
    if not text:
        # fallback: collect any text parts
        pieces: List[str] = []
        out = getattr(resp, "output", None)
        if isinstance(out, list):
            for item in out:
                if getattr(item, "type", None) == "output_text":
                    pieces.append(getattr(item, "text", ""))
        text = "\n".join(pieces).strip()

    if not text:
        raise RuntimeError("Model returned no text.")

    return parse_model_json(text)


# --------------------------- Pretty print & save ------------------------------
def pretty_tables(result: Dict[str, Any]) -> None:
    def section_table(title: str, obj: Optional[Dict[str, Any]]):
        if not obj:
            return
        tbl = Table(title=title, box=box.MINIMAL_DOUBLE_HEAD)
        tbl.add_column("Field", style="bold")
        tbl.add_column("Value")
        for k, v in obj.items():
            tbl.add_row(k, str(v))
        console.print(tbl)

    console.rule("[bold]Document Meta[/bold]")
    section_table("document_meta", result.get("document_meta"))

    for page_key in ("page_1", "page_2", "page_3"):
        page = result.get(page_key)
        if not page:
            continue
        console.rule(f"[bold]{page_key}[/bold]")
        for subkey, subobj in page.items():
            section_table(f"{page_key}.{subkey}", subobj)

    if result.get("provenance"):
        tbl = Table(title="Provenance", box=box.SIMPLE_HEAVY)
        tbl.add_column("metric")
        tbl.add_column("page")
        tbl.add_column("quote")
        for item in result["provenance"]:
            tbl.add_row(str(item.get("metric")), str(item.get("page")), str(item.get("quote"))[:140])
        console.print(tbl)

    if result.get("warnings"):
        console.print(f"[yellow]Warnings:[/yellow] {result['warnings']}")
    if result.get("missing"):
        console.print(f"[magenta]Missing:[/magenta] {result['missing']}")


def flatten_for_csv(result: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    meta = result.get("document_meta") or {}
    for page_key in ("page_1", "page_2", "page_3"):
        page = result.get(page_key) or {}
        for subkey, subobj in page.items():
            if isinstance(subobj, dict):
                for k, v in subobj.items():
                    rows.append({
                        "section": f"{page_key}.{subkey}",
                        "field": k,
                        "value": v,
                        "company": meta.get("company"),
                        "report_period": meta.get("report_period"),
                    })
    return pd.DataFrame(rows)


# ---------------------------------- CLI --------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Extract ALL metrics from a financial PDF via OpenAI (tables + text) into strict JSON.")
    ap.add_argument("pdf", help="Path to PDF")
    ap.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), help="Vision-capable model, e.g., gpt-4o-mini")
    ap.add_argument("--save-json", type=str, default=None, help="Optional path to save JSON")
    ap.add_argument("--save-csv", type=str, default=None, help="Optional path to save a flat CSV")
    args = ap.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]OPENAI_API_KEY not set[/red]")
        return 2

    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.exists():
        console.print(f"[red]File not found:[/red] {pdf_path}")
        return 2

    client = OpenAI(api_key=api_key)
    result = extract_all_from_pdf(client, pdf_path, model=args.model)

    pretty_tables(result)

    if args.save_json:
        outp = Path(args.save_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(result, indent=2))
        console.print(f"[green]Saved JSON:[/green] {outp}")

    if args.save_csv:
        df = flatten_for_csv(result)
        if not df.empty:
            outc = Path(args.save_csv)
            outc.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(outc, index=False)
            console.print(f"[green]Saved CSV:[/green] {outc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

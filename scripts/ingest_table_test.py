#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Dict

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

# Pull internals from our ingest module (ok for a test script)
from equichat.config import CONFIG
from equichat.schemas import Fact
from equichat.ingest import _extract_table_facts_camelot, file_doc_id, guess_company_from_filename

console = Console()


def find_pdfs(paths: List[str]) -> List[Path]:
    out: List[Path] = []
    for p in paths:
        pp = Path(p)
        if pp.is_file() and pp.suffix.lower() == ".pdf":
            out.append(pp)
        elif pp.is_dir():
            out.extend([q for q in pp.rglob("*.pdf") if q.is_file()])
        else:
            console.print(f"[yellow]Skipping (not a PDF or directory):[/yellow] {pp}")
    return sorted(set(out))


def facts_to_df(pdf_path: Path, facts: List[Fact]) -> pd.DataFrame:
    rows: List[Dict] = []
    for f in facts:
        rows.append(
            dict(
                pdf=str(pdf_path),
                company=f.company,
                metric_key=f.metric_key,
                value=f.value,
                unit=f.unit,
                period_type=f.period_type,
                period_end=f.period_end,
                page=f.source_page,
                snippet=(f.source_span or "")[:140],
                confidence=f.confidence,
            )
        )
    return pd.DataFrame(rows)


def print_df(df: pd.DataFrame, title: str = "Table-extracted facts") -> None:
    if df.empty:
        console.print("[bold yellow]No table facts extracted.[/bold yellow]")
        return
    tbl = Table(title=title, box=box.MINIMAL_DOUBLE_HEAD, show_lines=False)
    # Order a few important columns first, then the rest
    cols_order = ["pdf", "company", "metric_key", "value", "unit", "page", "period_type", "period_end", "confidence", "snippet"]
    cols = [c for c in cols_order if c in df.columns] + [c for c in df.columns if c not in cols_order]
    for c in cols:
        tbl.add_column(c, overflow="fold")
    for _, r in df.iterrows():
        tbl.add_row(*[str(r.get(c, "")) for c in cols])
    console.print(tbl)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract metrics from PDF tables (Camelot only) without starting the API."
    )
    parser.add_argument("paths", nargs="+", help="PDF file(s) and/or directories to scan for PDFs")
    parser.add_argument("--max-pages", type=int, default=None, help="Limit pages per PDF (debug)")
    parser.add_argument("--save-csv", type=str, default=None, help="Optional path to save aggregated CSV")
    parser.add_argument("--ignore-last-page", action="store_true", help="Ignore last page of each PDF")
    parser.add_argument("--no-ignore-last-page", dest="ignore_last_page", action="store_false", help="Include last page")
    parser.set_defaults(ignore_last_page=True)

    args = parser.parse_args()

    # Force table backend to camelot (table-only run)
    CONFIG.table_backend = "camelot"
    CONFIG.max_pages = args.max_pages
    CONFIG.ignore_last_page = args.ignore_last_page

    # Sanity: check Camelot availability before we start
    try:
        import camelot  # noqa: F401
    except Exception:
        console.print(
            "[red]Camelot is not available. Install requirements and ensure Ghostscript is present."
            "\n  Ubuntu/Debian: sudo apt-get install -y ghostscript[/red]"
        )
        return 2

    pdfs = find_pdfs(args.paths)
    if not pdfs:
        console.print("[yellow]No PDFs found in given paths.[/yellow]")
        return 0

    all_df = []
    for pdf in pdfs:
        # Minimal metadata like ingest does
        doc_id = file_doc_id(pdf.as_posix())
        company = guess_company_from_filename(pdf.name) or "Unknown"

        # Run table-only extraction (no DB writes)
        try:
            facts = _extract_table_facts_camelot(pdf.as_posix(), company=company, doc_id=doc_id)
        except Exception as e:
            console.print(f"[red]Table extraction failed for {pdf}:[/red] {e}")
            continue

        df = facts_to_df(pdf, facts)
        print_df(df, title=f"Facts from {pdf.name}")
        all_df.append(df)

    if all_df:
        ag = pd.concat(all_df, ignore_index=True)
        # Sort for readability
        ag = ag.sort_values(by=["company", "metric_key", "page", "value"], ascending=[True, True, True, False])
        console.rule("[bold]Aggregated[/bold]")
        print_df(ag, title="All PDFs (table-only)")

        if args.save_csv:
            out = Path(args.save_csv)
            out.parent.mkdir(parents=True, exist_ok=True)
            ag.to_csv(out, index=False)
            console.print(f"[green]Saved CSV:[/green] {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
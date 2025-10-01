#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, sys, re
import duckdb
from typing import List, Tuple, Any
from rich.console import Console
from openai import OpenAI

console = Console()

# --------------------------- Prompts --------------------------------
SYSTEM_SQL = """You are a SQL query generator.
Translate natural language financial questions into SQL queries for DuckDB
over the view:

v_facts_best(
  doc_id TEXT,
  company TEXT,
  period_type TEXT,     -- e.g. 'quarter', 'year', 'estimate'
  period_end TEXT,
  metric_key TEXT,      -- metric name (Revenue, EBITDA, etc.)
  metric_variant TEXT,  -- e.g. 'FY24A', 'Q4FY24', 'FY25E'
  value DOUBLE,
  unit TEXT,
  source_page INT,
  source_span TEXT,
  confidence DOUBLE,
  rn BIGINT
)

Rules:
- Always SELECT company, metric_key, metric_variant, value, unit.
- Use ILIKE for fuzzy matches on company and metric_key.
- For fiscal years/quarters ALWAYS use fuzzy matching:
  metric_variant ILIKE '%FY24%' (not = 'FY24').
- Limit results with LIMIT 20.
- Return only the SQL query, nothing else.
"""

SYSTEM_ANSWER = """You are a financial QA assistant.
You are given:
- The original natural language query
- A set of rows retrieved from DuckDB

Each row has: (company, metric_key, metric_variant, value, unit).

Rules:
- Use ONLY the provided rows to answer.
- If multiple rows are relevant, aggregate or list them.
- Be precise and concise (1–3 sentences).
- If no rows are relevant, say "No matching data found."
"""

# --------------------------- Helpers --------------------------------
def clean_sql(sql: str) -> str:
    sql = sql.strip()
    if sql.startswith("```"):
        sql = re.sub(r"^```[a-zA-Z]*", "", sql).strip()
        sql = sql.replace("```", "").strip()
    return sql

def fix_sql_variants(sql: str) -> str:
    # Replace = 'FY24' → ILIKE '%FY24%'
    sql = re.sub(
        r"metric_variant\s*=\s*'([^']*FY[0-9]{2}[^']*)'",
        r"metric_variant ILIKE '%\1%'",
        sql,
        flags=re.IGNORECASE,
    )
    return sql

def generate_sql(client: OpenAI, query: str, model: str) -> str:
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_SQL},
            {"role": "user", "content": query},
        ],
        max_output_tokens=400,
    )
    sql = getattr(resp, "output_text", "").strip()
    sql = clean_sql(sql)
    sql = fix_sql_variants(sql)
    console.print(f"[cyan]LLM SQL:[/cyan] {sql}")
    return sql

def answer_from_rows(client: OpenAI, query: str, rows: List[Tuple[Any]], model: str) -> str:
    # Format rows into a simple table-like string
    rows_text = "\n".join([str(r) for r in rows]) if rows else "No rows."
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_ANSWER},
            {"role": "user", "content": f"Query: {query}\nRows:\n{rows_text}"},
        ],
        max_output_tokens=400,
    )
    return getattr(resp, "output_text", "").strip()

# --------------------------- Main -----------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="LLM Query Router with post-answer generation")
    ap.add_argument("query", help="Natural language financial query")
    ap.add_argument("--db", default="./data/equichat.duckdb")
    ap.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    args = ap.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]OPENAI_API_KEY not set[/red]")
        return 2

    conn = duckdb.connect(args.db, read_only=True)
    client = OpenAI(api_key=api_key)

    sql = generate_sql(client, args.query, args.model)

    try:
        rows = conn.execute(sql).fetchall()
    except Exception as e:
        console.print(f"[red]SQL execution failed:[/red] {e}")
        rows = []

    console.print(f"[green]Raw rows:[/green] {rows}")

    answer = answer_from_rows(client, args.query, rows, args.model)
    console.rule("[bold yellow]Final Answer[/bold yellow]")
    console.print(answer)

    return 0

if __name__ == "__main__":
    sys.exit(main())

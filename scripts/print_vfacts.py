#!/usr/bin/env python3
import duckdb
from rich.console import Console
from rich.table import Table

console = Console()

def main():
    db_path = "./data/equichat.duckdb"
    conn = duckdb.connect(db_path, read_only=True)

    # Check schema
    console.rule("[bold cyan]Schema of v_facts_best[/bold cyan]")
    schema = conn.execute("DESCRIBE v_facts_best").fetchall()
    tbl = Table(title="v_facts_best schema")
    tbl.add_column("column")
    tbl.add_column("type")
    for col, typ, *_ in schema:
        tbl.add_row(col, typ)
    console.print(tbl)

    # Show first 20 rows
    console.rule("[bold green]Sample rows from v_facts_best[/bold green]")
    rows = conn.execute("""
        SELECT company, metric_key, metric_variant, value, unit
FROM v_facts_best
WHERE company ILIKE '%Hindalco%' AND metric_key ILIKE '%Revenue%' AND metric_variant ILIKE '%FY24%'
        LIMIT 20
    """).fetchdf()
    console.print(rows)

if __name__ == "__main__":
    main()
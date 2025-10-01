#!/usr/bin/env python3
import argparse, os, sys, duckdb
from rich.console import Console
from openai import OpenAI
from equichat.router_llm import query_with_llm_router

console = Console()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--db", default="./data/equichat.duckdb")
    ap.add_argument("--model", default=os.getenv("OPENAI_MODEL","gpt-4o-mini"))
    args = ap.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]OPENAI_API_KEY not set[/red]"); return 2

    client = OpenAI(api_key=api_key)
    conn = duckdb.connect(args.db, read_only=True)

    res = query_with_llm_router(client, args.query, conn, model=args.model)
    console.print(f"[cyan]Tool:[/cyan] {res['tool']}")
    console.print(f"[cyan]SQL:[/cyan] {res['sql']}")
    console.print(f"[green]Rows:[/green] {res['rows']}")
    console.rule("[bold yellow]Final Answer[/bold yellow]")
    console.print(res["answer"])
    return 0

if __name__ == "__main__":
    sys.exit(main())

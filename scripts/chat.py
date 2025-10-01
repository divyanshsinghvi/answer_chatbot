# scripts/chat.py — patch the renderer to handle new LLM router schema

from __future__ import annotations
import os, sys, json
import requests
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

API_URL = os.getenv("EQUICHAT_API", "http://127.0.0.1:8000")
console = Console()

def render_llm_router_response(payload: dict) -> None:
    # New schema from router_llm.query_with_llm_router
    tool = payload.get("tool")
    sql = payload.get("sql")
    answer = payload.get("answer")
    rows = payload.get("rows") or []

    # header
    hdr = f"Tool: {tool or 'n/a'}"
    console.print(Panel.fit(hdr, style="bold cyan"))

    # optional SQL
    if sql:
        console.print(Panel.fit(sql, title="SQL", box=box.SIMPLE, style="dim"))

    # answer
    if answer:
        console.print(Panel.fit(answer, title="Answer", box=box.MINIMAL_DOUBLE_HEAD, style="bold green"))
    else:
        console.print(Panel.fit("No answer text.", style="yellow"))

    # rows table (generic)
    if rows:
        # rows may be list[tuple] (facts) OR list[dict] (vector)
        tbl = Table(title="Rows", box=box.SIMPLE_HEAVY)
        # discover columns
        if isinstance(rows[0], dict):
            cols = list(rows[0].keys())
            for c in cols:
                tbl.add_column(str(c))
            for r in rows:
                tbl.add_row(*[str(r.get(c, "")) for c in cols])
        else:
            # assume tuple-like
            max_cols = max(len(r) for r in rows)
            for i in range(max_cols):
                tbl.add_column(f"col{i}")
            for r in rows:
                vals = [str(v) for v in r] + [""] * (max_cols - len(r))
                tbl.add_row(*vals)
        console.print(tbl)
    else:
        console.print("[dim]No rows returned.[/dim]")

def ask(query: str) -> None:
    try:
        resp = requests.post(f"{API_URL}/query", json={"query": query}, timeout=60)
    except Exception as e:
        console.print(f"[red]HTTP error:[/red] {e}")
        return

    if resp.status_code != 200:
        console.print(f"[red]Server responded {resp.status_code}[/red]: {resp.text[:400]}")
        return

    # Try new schema first
    try:
        payload = resp.json()
    except Exception:
        console.print(f"[red]Non-JSON response:[/red] {resp.text[:400]}")
        return

    if isinstance(payload, dict) and "tool" in payload and "answer" in payload:
        render_llm_router_response(payload)
        return

    # Fallback: legacy schema rendering
    status = payload.get("status", "unknown")
    console.print(Panel.fit(f"Status: {status}", style="bold cyan"))
    if "answer" in payload:
        console.print(Panel.fit(payload["answer"], title="Answer", box=box.MINIMAL_DOUBLE_HEAD, style="bold green"))
    rows = payload.get("rows") or []
    if rows:
        tbl = Table(title="Rows", box=box.SIMPLE_HEAVY)
        if isinstance(rows[0], dict):
            cols = list(rows[0].keys())
            for c in cols: tbl.add_column(str(c))
            for r in rows: tbl.add_row(*[str(r.get(c, "")) for c in cols])
        else:
            max_cols = max(len(r) for r in rows)
            for i in range(max_cols): tbl.add_column(f"col{i}")
            for r in rows:
                vals = [str(v) for v in r] + [""] * (max_cols - len(r))
                tbl.add_row(*vals)
        console.print(tbl)
    else:
        console.print("[dim]No rows returned.[/dim]")

def main():
    console.print("EquiChat CLI — talks to FastAPI at", API_URL)
    console.print("Type /quit to exit.")
    while True:
        q = console.input("> ").strip()
        if not q: continue
        if q.lower() in {"/q", "/quit", "exit"}: break
        ask(q)

if __name__ == "__main__":
    sys.exit(main())

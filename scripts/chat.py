#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Optional, Dict, Any

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from urllib import request, error


# --- Config -------------------------------------------------------------------
API_URL = os.getenv("EQUICHAT_API_URL", "http://127.0.0.1:8000")
QUERY_ENDPOINT = f"{API_URL}/query"

HISTORY_FILE = os.path.expanduser("~/.equichat_history")


# --- Simple client (stdlib only: urllib) --------------------------------------
def http_post_json(url: str, payload: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            return json.loads(body.decode("utf-8"))
    except error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8")
        except Exception:
            detail = str(e)
        return {"status": "error", "message": f"HTTP {e.code}: {detail}"}
    except Exception as e:
        return {"status": "error", "message": f"Request failed: {e}"}


# --- REPL State ----------------------------------------------------------------
@dataclass
class ChatState:
    company: Optional[str] = None
    industry: Optional[str] = None

    def set_slot(self, key: str, value: Optional[str]) -> str:
        key = key.lower().strip()
        if key not in {"company", "industry"}:
            return "Unknown slot. Use: company | industry"
        setattr(self, key, value)
        return f"Set {key} = {value}"


# --- Rendering helpers ---------------------------------------------------------
console = Console()


def render_result(result: Dict[str, Any]) -> None:
    status = result.get("status", "unknown")
    if status == "ok" and "rows" in result and isinstance(result["rows"], list):
        rows = result["rows"]
        if not rows:
            console.print("[bold yellow]No rows returned.[/bold yellow]")
            return
        # Build a nice table
        cols = list(rows[0].keys())
        table = Table(title="Aggregation Result", box=box.MINIMAL_DOUBLE_HEAD, show_lines=False)
        for c in cols:
            table.add_column(c, overflow="fold")
        for r in rows:
            table.add_row(*[str(r.get(c, "")) for c in cols])
        console.print(table)
        return

    # Single metric or messages
    grid = Table.grid(expand=True)
    grid.add_column(ratio=1)
    title = f"[bold green]Status[/bold green]: {status}"
    if result.get("message"):
        title += f" • [yellow]{result['message']}[/yellow]"
    console.print(Panel.fit(title, border_style="cyan"))

    # Key fields
    fields = ["company", "metric_key", "value", "unit", "period_type", "period_end", "source_page", "confidence"]
    details = Table(box=box.SIMPLE_HEAVY)
    details.add_column("Field", style="bold")
    details.add_column("Value")
    for f in fields:
        v = result.get(f)
        if v is not None:
            details.add_row(f, str(v))
    console.print(details)


def render_state(state: ChatState) -> None:
    kv = Table.grid(padding=(0, 1))
    kv.add_column(style="bold magenta")
    kv.add_column()
    kv.add_row("company", state.company or "-")
    kv.add_row("industry", state.industry or "-")
    console.print(Panel(kv, title="Session Context", border_style="magenta"))


# --- Commands ------------------------------------------------------------------
HELP_TEXT = """\
Commands:
  /set company <NAME>      set the current company for "this company" questions
  /set industry <NAME>     set the current industry (e.g., Banking, Pharma)
  /state                   show current session context
  /clear                   clear company & industry
  /help                    show this help
  /exit or Ctrl-D          exit

Examples:
  /set company Hindalco Industries
  ok tell me about the revenue of this company?
  compare it with other similar companies in the same industry
  top 3 companies by revenue in banking where market cap >= 1000
"""


def handle_command(cmd: str, state: ChatState) -> Optional[str]:
    parts = cmd.strip().split()
    if not parts:
        return None
    head = parts[0].lower()
    if head in ("/exit", "/quit"):
        raise EOFError()
    if head == "/help":
        console.print(Panel(HELP_TEXT, title="Help", border_style="blue"))
        return None
    if head == "/state":
        render_state(state)
        return None
    if head == "/clear":
        state.company = None
        state.industry = None
        console.print("[green]Cleared session context.[/green]")
        return None
    if head == "/set" and len(parts) >= 3:
        slot = parts[1]
        value = " ".join(parts[2:])
        console.print(f"[green]{state.set_slot(slot, value)}[/green]")
        return None
    if head == "/set" and len(parts) <= 2:
        console.print("[yellow]Usage: /set <company|industry> <VALUE>[/yellow]")
        return None
    console.print("[yellow]Unknown command. Type /help[/yellow]")
    return None


# --- Main loop -----------------------------------------------------------------
def main() -> int:
    console.print(Panel("[bold]EquiChat CLI[/bold] — talks to FastAPI at "
                        f"[cyan]{API_URL}[/cyan]\nType /help for commands.",
                        border_style="green"))

    state = ChatState()
    session = PromptSession(history=FileHistory(HISTORY_FILE))
    with patch_stdout():
        while True:
            try:
                text = session.prompt("> ").strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]bye[/dim]")
                break

            if not text:
                continue

            if text.startswith("/"):
                try:
                    handle_command(text, state)
                except EOFError:
                    console.print("[dim]bye[/dim]")
                    break
                continue

            # Call API
            payload = {"query": text, "company": state.company, "industry": state.industry}
            result = http_post_json(QUERY_ENDPOINT, payload)
            if result.get("status") == "error":
                console.print(f"[red]Error:[/red] {result.get('message')}")
                continue

            # Render
            render_result(result)

    return 0


if __name__ == "__main__":
    sys.exit(main())

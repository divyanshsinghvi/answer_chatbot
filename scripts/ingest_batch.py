#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any

import duckdb
from concurrent.futures import ThreadPoolExecutor, as_completed

from equichat.ingest import ingest_pdf_with_openai
from equichat.store import Store
from equichat.ingest import persist_facts



def file_doc_id(path: str) -> str:
    """Same hashing strategy as ingest: sha1 of the absolute path."""
    h = hashlib.sha1()
    h.update(path.encode("utf-8"))
    return h.hexdigest()


def doc_exists(conn: duckdb.DuckDBPyConnection, doc_id: str) -> bool:
    try:
        row = conn.execute(
            "SELECT 1 FROM documents WHERE doc_id = ? LIMIT 1", [doc_id]
        ).fetchone()
        return bool(row)
    except Exception:
        # If documents table doesn't exist yet, treat as not existing.
        return False


def get_cache_file(pdf: Path, cache_dir: Path) -> Path:
    """Get the cache file path for a PDF."""
    pdf_name = pdf.stem  # filename without extension
    return cache_dir / f"{pdf_name}_extracted.json"


def ingest_one(pdf: Path, model: str, force: bool, cache_dir: Path) -> Dict[str, Any]:
    abs_path = str(pdf.resolve())
    doc_id = file_doc_id(abs_path)
    cache_file = get_cache_file(pdf, cache_dir)

    print(f"📄 Processing {pdf.name}...")

    try:
        # Check if cached extraction exists
        if cache_file.exists() and not force:
            print(f"  ♻️  Loading from cache: {cache_file.name}")
            with open(cache_file, 'r') as f:
                res = json.load(f)
        else:
            print(f"  🤖 Extracting with OpenAI ({model})...")
            # Use persist=False to get facts without writing to DB (we'll batch write later)
            res = ingest_pdf_with_openai(abs_path, model=model, persist=False)

            # Save to cache file
            cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump(res, f, indent=2)
            print(f"  💾 Cached to: {cache_file.name}")

        facts = res.get("facts", [])
        return {
            "status": "ok",
            "pdf": pdf.name,
            "doc_id": res.get("doc_id", doc_id),
            "company": res.get("company", "unknown"),
            "facts": facts,
            "facts_count": len(facts),  # Add count for display
        }
    except Exception as e:
        return {"status": "fail", "pdf": pdf.name, "error": str(e), "doc_id": doc_id}



def main() -> int:
    ap = argparse.ArgumentParser(
        description="Batch ingest PDFs into DuckDB via OpenAI extraction (parallel, deduplicated)"
    )
    ap.add_argument("--folder", default="data", help="Folder containing PDFs (default: ./data)")
    ap.add_argument("--limit", type=int, default=5, help="Max PDFs to ingest (default: 5)")
    ap.add_argument("--workers", type=int, default=3, help="Number of parallel workers (default: 3)")
    ap.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        help="Vision-capable model for extraction",
    )
    ap.add_argument(
        "--db",
        default=os.getenv("EQUICHAT_DB_PATH", "./data/equichat.duckdb"),
        help="Path to DuckDB database",
    )
    ap.add_argument(
        "--cache-dir",
        default="./cache/extractions",
        help="Directory to cache OpenAI extractions (default: ./cache/extractions)",
    )
    ap.add_argument("--force", action="store_true", help="Re-extract even if cache exists")
    args = ap.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()

    if not folder.exists():
        print(f"❌ Folder not found: {folder}")
        return 2

    pdfs = sorted(p for p in folder.glob("*.pdf"))[: args.limit]
    if not pdfs:
        print(f"⚠️ No PDF files found in {folder}")
        return 0

    print(f"📥 Ingesting {len(pdfs)} PDF(s) from {folder} with {args.workers} workers")
    print(f"   Cache: {cache_dir}")
    print(f"   Force re-extract: {args.force}")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(ingest_one, pdf, args.model, args.force, cache_dir): pdf for pdf in pdfs}
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            if res["status"] == "ok":
                print(f"✅ {res['pdf']} → {res['facts_count']} facts (doc_id={res['doc_id'][:12]}…)")
            elif res["status"] == "skip":
                print(f"⏭️  Skipping {res['pdf']} (already ingested)")
            else:
                print(f"❌ Failed {res['pdf']}: {res.get('error')}")

    with Store(args.db, read_only=False) as store:
        for r in results:
            if r.get("facts"):
                # convert dicts back to tuples for executemany OR call persist_facts with FactRows if you kept them
                # here we just reuse persist_facts by rebuilding FactRow objects if needed,
                # but since we returned dicts, do a direct executemany:

                facts_data = r["facts"]
                if facts_data:
                    store.conn.execute("DELETE FROM facts WHERE doc_id = ?", [r["doc_id"]])
                    store.conn.executemany(
                        """
                        INSERT INTO facts (
                            doc_id, company, metric_key, metric_variant, value, unit,
                            period_type, period_end, source_page, source_span, confidence
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            (
                                f["doc_id"], f["company"], f["metric_key"], f["metric_variant"],
                                f["value"], f["unit"], f["period_type"], f["period_end"],
                                f["source_page"], f["source_span"], f["confidence"]
                            )
                            for f in facts_data
                        ],
                    )
                    print(f"💾 Persisted {len(facts_data)} facts for {r['company']}")

    # Summary
    print("\n📊 Summary:")
    total_facts = 0
    for r in results:
        if r["status"] == "ok":
            facts_count = r['facts_count']
            total_facts += facts_count
            print(f"- {r['company']} → {facts_count} facts from {r['pdf']} (doc_id={r['doc_id']})")
        elif r["status"] == "skip":
            print(f"- skipped {r['pdf']}")
        else:
            print(f"- failed {r['pdf']} ({r.get('error')})")

    print(f"\n🎉 Total: {total_facts} facts ingested from {len([r for r in results if r['status'] == 'ok'])} PDFs")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

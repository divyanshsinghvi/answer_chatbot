#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

import fitz  # PyMuPDF
from equichat.store import Store
from equichat.config import CONFIG


DDL = """
CREATE TABLE IF NOT EXISTS documents (
  doc_id TEXT PRIMARY KEY,
  company TEXT,
  source_path TEXT,
  page_count INTEGER,
  ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS raw_pages (
  doc_id TEXT,
  page_num INTEGER,
  text TEXT,
  company TEXT,
  source_path TEXT
);

-- helpful view to quickly see first lines
CREATE VIEW IF NOT EXISTS v_raw_pages_preview AS
SELECT doc_id, page_num, substr(text, 1, 200) AS preview
FROM raw_pages;
"""


def file_doc_id(path: str) -> str:
    h = hashlib.sha1()
    h.update(path.encode("utf-8"))
    return h.hexdigest()


def discover_pdfs(inputs: List[str], glob_pattern: Optional[str], recursive: bool) -> List[Path]:
    found: List[Path] = []
    for item in inputs:
        p = Path(item).expanduser().resolve()
        if p.is_file() and p.suffix.lower() == ".pdf":
            found.append(p)
        elif p.is_dir():
            if glob_pattern:
                pat = f"**/{glob_pattern}" if recursive else glob_pattern
                for fp in p.glob(pat):
                    if fp.is_file() and fp.suffix.lower() == ".pdf":
                        found.append(fp)
            else:
                it = p.rglob("*.pdf") if recursive else p.glob("*.pdf")
                found.extend([fp for fp in it if fp.is_file()])
        else:
            # treat item as a glob from CWD
            for fp in Path(".").glob(item):
                if fp.is_file() and fp.suffix.lower() == ".pdf":
                    found.append(fp.resolve())
    # de-dup
    uniq = []
    seen = set()
    for f in found:
        if f not in seen:
            uniq.append(f); seen.add(f)
    return uniq


def extract_page_texts(pdf_path: Path, ignore_last_page: bool) -> List[Tuple[int, str]]:
    """
    Returns [(page_num_1based, text), ...].
    Uses PyMuPDF 'text' extraction (ignores images/charts by default).
    """
    pages: List[Tuple[int, str]] = []
    with fitz.open(pdf_path) as doc:
        total = len(doc)
        last_idx = total - 1 if ignore_last_page and total > 0 else total
        for i in range(0, last_idx):
            page = doc[i]
            txt = page.get_text("text") or ""
            # normalize newlines a bit
            txt = "\n".join(line.rstrip() for line in txt.splitlines())
            pages.append((i + 1, txt))
    return pages


def upsert_document(store: Store, doc_id: str, company: str, source_path: str, page_count: int):
    # remove any existing document + pages if re-ingesting
    store.conn.execute("DELETE FROM raw_pages WHERE doc_id = ?", [doc_id])
    store.conn.execute("DELETE FROM documents WHERE doc_id = ?", [doc_id])
    store.conn.execute(
        "INSERT INTO documents (doc_id, company, source_path, page_count) VALUES (?, ?, ?, ?)",
        [doc_id, company, source_path, page_count],
    )


def insert_pages(store: Store, doc_id: str, company: str, source_path: str, pages: List[Tuple[int, str]]):
    if not pages:
        return
    store.conn.executemany(
        "INSERT INTO raw_pages (doc_id, page_num, text, company, source_path) VALUES (?, ?, ?, ?, ?)",
        [(doc_id, pnum, txt, company, source_path) for (pnum, txt) in pages],
    )


def already_ingested(store: Store, doc_id: str) -> bool:
    q = store.conn.execute("SELECT 1 FROM documents WHERE doc_id = ? LIMIT 1", [doc_id]).fetchone()
    return q is not None


def guess_company_from_name(path: Path) -> str:
    # crude but handy; you can replace with a smarter map later
    name = path.stem
    # strip date-like tails
    name = name.replace("_", " ").replace("-", " ")
    return name.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch-ingest ONLY full text (per page) from PDFs into DuckDB.")
    ap.add_argument("inputs", nargs="+", help="Files/dirs/globs. Directories will be scanned for PDFs.")
    ap.add_argument("--glob", dest="glob_pattern", default=None, help="Optional glob to filter PDFs inside dirs, e.g. '*.pdf'")
    ap.add_argument("--recursive", action="store_true", help="Recurse into directories")
    ap.add_argument("--ignore-last-page", action="store_true", help="Skip the last page (often disclaimers/charts)")
    ap.add_argument("--company", default=None, help="Company name override for all files (otherwise inferred from filename)")
    ap.add_argument("--skip-existing", action="store_true", help="Skip if doc_id already present in documents")
    args = ap.parse_args()

    pdfs = discover_pdfs(args.inputs, args.glob_pattern, args.recursive)
    if not pdfs:
        print("No PDFs found.")
        return 1

    with Store(CONFIG.db_path, read_only=False) as store:
        # ensure tables
        for stmt in DDL.strip().split(";"):
            s = stmt.strip()
            if s:
                store.conn.execute(s)

        total_docs, total_pages = 0, 0

        for pdf in pdfs:
            doc_id = file_doc_id(str(pdf))
            if args.skip_existing and already_ingested(store, doc_id):
                print(f"[skip] {pdf} (already ingested)")
                continue

            company = args.company or guess_company_from_name(pdf)

            try:
                pages = extract_page_texts(pdf, ignore_last_page=args.ignore_last_page)
            except Exception as e:
                print(f"[error] {pdf}: {e}")
                continue

            try:
                upsert_document(store, doc_id, company, str(pdf), page_count=len(pages))
                insert_pages(store, doc_id, company, str(pdf), pages)
                total_docs += 1
                total_pages += len(pages)
                print(f"[ok] {pdf} • pages={len(pages)} • doc_id={doc_id[:12]}…")
            except Exception as e:
                print(f"[error-db] {pdf}: {e}")

        print(f"\nDone. Ingested {total_docs} document(s), {total_pages} page(s) into {CONFIG.db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
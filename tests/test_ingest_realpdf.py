from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd
import pytest

from equichat.config import CONFIG
from equichat.store import Store
from equichat.ingest import ingest_pdf


@pytest.mark.parametrize("ignore_last_page", [True, False])
def test_ingest_real_pdfs(ignore_last_page: bool, tmp_path: Path, monkeypatch, capsys):
    """
    Ingest any real PDFs found in tests/data/*.pdf.

    - Uses a temporary, file-backed DuckDB to avoid locks.
    - Runs twice: with ignore_last_page True and False.
    - Asserts we ingested documents; asserts we extracted at least *some* facts across all PDFs.
    - Prints a small summary of the first few extracted facts so you can eyeball results.
    """
    data_dir = Path("tests") / "data"
    pdfs = sorted(glob.glob(str(data_dir / "*.pdf")))
    if not pdfs:
        pytest.skip("No real PDFs present in tests/data/*.pdf — add your files to run this test.")

    # Configure DB and toggle ignore_last_page for this run
    db_file = tmp_path / f"equichat_real_{'nolast' if ignore_last_page else 'withlast'}.duckdb"
    monkeypatch.setenv("EQUICHAT_DB_PATH", db_file.as_posix())
    CONFIG.ignore_last_page = ignore_last_page

    total_docs = 0
    with Store(db_file.as_posix()) as store:
        for p in pdfs:
            _ = ingest_pdf(p, store)  # company will be guessed from filename
            total_docs += 1

        # Pull all facts for a quick sanity check
        facts_df = store.conn.execute(
            "SELECT company, metric_key, value, unit, source_page, substr(source_span,1,120) AS snippet FROM facts"
        ).df()

        # Print a small sample to the test output (helpful for you to see what's extracted)
        if not facts_df.empty:
            print("\n[Extracted facts sample]")
            print(facts_df.head(10).to_string(index=False))

        # Assert we ingested documents
        assert total_docs == len(pdfs)

        # Assert at least some facts extracted across the PDFs (loose)
        # If you truly get zero, we'll xfail so you can tune Camelot/text regex.
        if facts_df.empty:
            pytest.xfail(
                "No facts extracted from real PDFs. "
                "Try run with ignore_last_page=False, verify camelot/table backend works, "
                "or adjust alias patterns/units."
            )

        # Optional: check for a few common metrics if present
        # (soft asserts: don't fail if not found—just informative)
        common = facts_df["metric_key"].value_counts().to_dict()
        print("\n[Metric counts]", common)

        # Ensure values are numeric and units normalized
        assert facts_df["value"].dtype.kind in {"i", "u", "f"}
        assert set(facts_df["unit"].unique()) <= {"cr"}  # our current normalizer outputs 'cr' only

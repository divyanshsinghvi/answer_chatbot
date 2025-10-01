from __future__ import annotations
from typing import Optional, Literal
from pydantic import BaseModel, Field


class Document(BaseModel):
    """Metadata about an ingested financial PDF."""

    doc_id: str = Field(..., description="Stable hash-based identifier for the PDF")
    company: str = Field(..., description="Company name (normalized)")
    ticker: Optional[str] = Field(default=None, description="Optional ticker symbol")
    industry: Optional[str] = Field(default=None, description="Industry sector")
    report_date: Optional[str] = Field(
        default=None, description="Date of report (ISO 8601 if available)"
    )
    source_path: str = Field(..., description="Filesystem path to the PDF")
    page_count: int = Field(..., description="Total number of pages in the PDF")


class Fact(BaseModel):
    """A single extracted metric (numeric fact) from a PDF."""

    doc_id: str
    company: str
    period_type: Literal["quarter", "year", "asof", "unknown"] = "unknown"
    period_end: Optional[str] = None  # e.g., "2024-03-31"
    metric_key: str  # canonical metric name, e.g., "revenue"
    metric_variant: Optional[str] = None  # raw header/alias matched, e.g., "Sales"
    value: float
    unit: str = "cr"
    source_page: int
    source_span: Optional[str] = None  # snippet of line or table cell
    confidence: float = 0.5


class QueryResult(BaseModel):
    """Unified result returned by tools or router."""

    status: Literal["ok", "not_available", "uncertain", "need_company", "unknown"]
    company: Optional[str] = None
    metric_key: Optional[str] = None
    value: Optional[float] = None
    unit: Optional[str] = None
    period_type: Optional[str] = None
    period_end: Optional[str] = None
    source_page: Optional[int] = None
    confidence: Optional[float] = None
    message: Optional[str] = None
    rows: Optional[list[dict]] = None  # for top-k/aggregation queries

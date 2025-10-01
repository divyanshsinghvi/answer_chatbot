# src/equichat/router.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

# Simple keyword → canonical metric mapping
_METRIC_ALIASES: Dict[str, List[str]] = {
    "Revenue": ["revenue", "sales", "turnover", "top line"],
    "EBITDA": ["ebitda"],
    "EBIT": ["ebit", "operating profit"],
    "PAT": ["pat", "net profit", "profit after tax"],
    "Adj. EPS": ["adj eps", "adjusted eps"],
    "EPS": ["eps"],
    "EBITDA margin": ["ebitda margin", "operating margin"],
    "ROE": ["roe", "return on equity"],
    "ROCE": ["roce", "return on capital employed"],
    "P/E": ["pe", "p e", "price to earnings"],
    "EV/EBITDA": ["ev ebitda", "ev/ebitda"],
}

# Match FY/Q labels like FY24, FY24A, FY25E, Q4FY24, Q3FY23, etc.
PERIOD_RE = re.compile(r"\b(q[1-4]fy\d{2}|fy\d{2}[a-z]?)\b", re.I)

COMPANY_TOKEN_RE = re.compile(r"[A-Z][A-Za-z.&\-\s]{1,60}")


@dataclass
class Route:
    tool: str                # 'facts_sql' | 'unknown'
    metric_keys: List[str]   # canonical keys
    company_hint: Optional[str]
    period_label: Optional[str]


def _find_metric_keys(text: str) -> List[str]:
    t = text.lower()
    hits: List[str] = []
    for canon, variants in _METRIC_ALIASES.items():
        for v in variants:
            if v in t:
                hits.append(canon)
                break
    # de-dup preserving order
    seen = set()
    out = []
    for k in hits:
        if k not in seen:
            out.append(k); seen.add(k)
    return out


def _find_period_label(text: str) -> Optional[str]:
    m = PERIOD_RE.search(text)
    if not m:
        return None
    return m.group(1).upper()


def _find_company_hint(text: str) -> Optional[str]:
    """
    Very light heuristic:
    - take the longest capitalized token sequence that isn't just the first word
    - users often type: "What is Hindalco revenue in FY24?"
    """
    # Prefer quoted company names if present
    qm = re.search(r'"([^"]+)"|\'([^\']+)\'', text)
    if qm:
        name = qm.group(1) or qm.group(2)
        return name.strip()

    # Otherwise, pick a mid-sentence Capitalized sequence
    candidates = COMPANY_TOKEN_RE.findall(text)
    # remove leading word if question starts with "What/How/Tell"
    if candidates:
        # choose the one with most spaces (longest phrase)
        candidates = sorted(candidates, key=lambda s: len(s.split()), reverse=True)
        name = candidates[0].strip()
        # filter obvious non-company words
        if len(name.split()) <= 1 and name.lower() in {"what", "how", "tell"}:
            return None
        return name
    return None


def route(user_text: str) -> Route:
    metric_keys = _find_metric_keys(user_text)
    period_label = _find_period_label(user_text)
    company_hint = _find_company_hint(user_text)

    if metric_keys:
        return Route(
            tool="facts_sql",
            metric_keys=metric_keys,
            company_hint=company_hint,
            period_label=period_label,
        )
    return Route(tool="unknown", metric_keys=[], company_hint=company_hint, period_label=period_label)

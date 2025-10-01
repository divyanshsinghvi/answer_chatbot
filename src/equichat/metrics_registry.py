from __future__ import annotations
from typing import Dict, List, Pattern, Tuple
import regex as re


# Canonical metric keys and their alias regexes
METRIC_ALIASES: Dict[str, List[str]] = {
    "revenue": ["revenue", "sales", "top line"],
    "ebitda": ["ebitda", "operating profit"],
    "pat": ["pat", "net profit", "profit after tax"],
    "eps": ["eps", "earnings per share"],
    "market cap": ["market cap", "mcap", "capitalization"],
}


def alias_patterns() -> Dict[str, List[Pattern]]:
    """Compile regex patterns for each metric alias."""
    return {k: [re.compile(pat, re.I) for pat in pats] for k, pats in METRIC_ALIASES.items()}


ALIAS_PATTERNS: Dict[str, List[Pattern]] = alias_patterns()


def canonicalize_metric(name: str) -> Tuple[str, float]:
    """
    Try to resolve a raw header/alias to a canonical metric key.
    Returns (metric_key, score).
    Score ~ match confidence (1.0 = exact alias, 0.0 = unknown).
    """
    name_l = name.lower()
    for key, pats in ALIAS_PATTERNS.items():
        for pat in pats:
            if pat.search(name_l):
                return key, 1.0
    return name, 0.0

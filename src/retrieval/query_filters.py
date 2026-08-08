"""
Extract structured retrieval filters (company ticker, year, quarter) from a
natural-language question.

Two extraction strategies are available, selectable via the ``mode`` parameter:

  "regex"  — Fast, zero-cost rule-based extraction using word-boundary patterns
             and a curated company alias dictionary.  Covers common phrasings
             ("Q2 2023", "Apple", "second quarter") but misses informal aliases
             ("the iPhone maker") and ambiguous phrasing.
             Use when latency/cost matters more than recall on edge cases.

  "llm"    — LLM call with JSON structured output.  Understands arbitrary
             phrasing, company aliases, ordinals, and fiscal vs calendar year.
             One cheap gpt-4o-mini call per question (~$0.00002).
             Use in production or when question phrasing is unpredictable.

Both strategies return the same dict shape and both return None for any field
they cannot confidently determine — it is always safer to over-retrieve than to
filter incorrectly and silently miss relevant chunks.

Usage
-----
    from src.retrieval.query_filters import extract_query_filters

    # LLM mode (default)
    filters = extract_query_filters(
        question="How did the iPhone maker perform in Q2 2023?",
        api_key=config.generation.api_key,
    )
    # → {'company': 'AAPL', 'year': 2023, 'quarter': 'Q2'}

    # Regex mode (no LLM call)
    filters = extract_query_filters(
        question="Apple revenues Q2 2023",
        mode="regex",
    )
    # → {'company': 'AAPL', 'year': 2023, 'quarter': 'Q2'}
"""
import json
import logging
import re
from typing import Literal, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared type
# ---------------------------------------------------------------------------

QueryFilters = dict[str, Optional[str | int]]
_EMPTY: QueryFilters = {"company": None, "year": None, "quarter": None}

# ---------------------------------------------------------------------------
# Regex strategy — curated alias table + compiled patterns
# ---------------------------------------------------------------------------

# Map lowercase substrings → canonical ticker.
# Word-boundary matching prevents partial hits ("meta" inside "metamorphosis").
_ALIAS_MAP: dict[str, str] = {
    # Apple
    "apple": "AAPL", "aapl": "AAPL",
    # Microsoft
    "microsoft": "MSFT", "msft": "MSFT",
    # Alphabet / Google
    "alphabet": "GOOGL", "google": "GOOGL", "googl": "GOOGL", "goog": "GOOGL",
    # Amazon
    "amazon": "AMZN", "amzn": "AMZN",
    # Meta / Facebook
    "meta": "META", "facebook": "META",
    # Netflix
    "netflix": "NFLX", "nflx": "NFLX",
    # Tesla
    "tesla": "TSLA", "tsla": "TSLA",
    # NVIDIA
    "nvidia": "NVDA", "nvda": "NVDA",
    # Intel
    "intel": "INTC", "intc": "INTC",
    # IBM
    "ibm": "IBM",
    # Salesforce
    "salesforce": "CRM", "crm": "CRM",
    # Oracle
    "oracle": "ORCL", "orcl": "ORCL",
    # Uber
    "uber": "UBER",
    # Lyft
    "lyft": "LYFT",
    # Airbnb
    "airbnb": "ABNB", "abnb": "ABNB",
    # Snap
    "snap": "SNAP", "snapchat": "SNAP",
}

_ORDINAL_MAP: dict[str, str] = {
    "first": "Q1", "second": "Q2", "third": "Q3", "fourth": "Q4",
}

_MONTH_QUARTER_MAP: dict[str, str] = {
    # Jan–Mar → Q1, Apr–Jun → Q2, Jul–Sep → Q3, Oct–Dec → Q4
    "january": "Q1", "february": "Q1", "march": "Q1",
    "april":   "Q2", "may":      "Q2", "june":  "Q2",
    "july":    "Q3", "august":   "Q3", "september": "Q3",
    "october": "Q4", "november": "Q4", "december":  "Q4",
    # Abbreviated
    "jan": "Q1", "feb": "Q1", "mar": "Q1",
    "apr": "Q2",              "jun": "Q2",
    "jul": "Q3", "aug": "Q3", "sep": "Q3",
    "oct": "Q4", "nov": "Q4", "dec": "Q4",
}

_YEAR_RE    = re.compile(r"\b(20[12]\d)\b")
_QUARTER_RE = re.compile(r"\b(Q[1-4])\b", re.IGNORECASE)
_ORDINAL_RE = re.compile(r"\b(first|second|third|fourth)\s+quarter\b", re.IGNORECASE)
_MONTH_RE   = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september"
    r"|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b",
    re.IGNORECASE,
)


def _extract_regex(question: str) -> QueryFilters:
    q_lower = question.lower()

    # Year
    year: Optional[int] = None
    m = _YEAR_RE.search(question)
    if m:
        year = int(m.group(1))

    # Quarter — priority: explicit (Q2) > ordinal (second quarter) > month (April)
    quarter: Optional[str] = None
    m = _QUARTER_RE.search(question)
    if m:
        quarter = m.group(1).upper()
    else:
        m = _ORDINAL_RE.search(question)
        if m:
            quarter = _ORDINAL_MAP[m.group(1).lower()]
        else:
            m = _MONTH_RE.search(question)
            if m:
                quarter = _MONTH_QUARTER_MAP[m.group(1).lower()]

    # Company — longest alias wins to avoid partial matches
    company: Optional[str] = None
    matched_len = 0
    for alias, ticker in _ALIAS_MAP.items():
        if re.search(r"\b" + re.escape(alias) + r"\b", q_lower) and len(alias) > matched_len:
            company = ticker
            matched_len = len(alias)

    return {"company": company, "year": year, "quarter": quarter}


# ---------------------------------------------------------------------------
# LLM strategy — structured JSON output via OpenAI
# ---------------------------------------------------------------------------

_LLM_SYSTEM_PROMPT = """\
You are a metadata extraction assistant for SEC 10-Q financial filings.

Extract three pieces of structured information from the user's question:

1. company  – stock ticker (AAPL, MSFT, GOOGL, AMZN, META, TSLA, NVDA, NFLX,
               UBER, LYFT, ABNB, SNAP, CRM, ORCL, IBM, INTC …).
               Return the primary company being asked about.
               If you cannot determine it confidently, return null.

2. year     – fiscal/calendar year as a 4-digit integer (e.g. 2023).
               Return null if not mentioned or unclear.

3. quarter  – one of "Q1", "Q2", "Q3", "Q4".
               Interpret ordinals: first→Q1, second→Q2, third→Q3, fourth→Q4.
               Interpret month ranges: Jan-Mar→Q1, Apr-Jun→Q2, Jul-Sep→Q3, Oct-Dec→Q4.
               Return null if not mentioned or unclear.

Return ONLY a JSON object with exactly these keys: company, year, quarter.
No markdown, no explanation. When uncertain about a field, return null — a
missing filter is safe; a wrong filter silently drops valid results.

Examples:
  {"company": "AAPL", "year": 2023, "quarter": "Q2"}
  {"company": "MSFT", "year": 2022, "quarter": null}
  {"company": null,   "year": null,  "quarter": null}
"""


def _extract_llm(question: str, api_key: str, model: str) -> QueryFilters:
    import openai

    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                {"role": "user",   "content": question},
            ],
            temperature=0.0,
            max_tokens=60,
            response_format={"type": "json_object"},
        )
        raw = (response.choices[0].message.content or "").strip()
        parsed = json.loads(raw)

        company = str(parsed["company"]).upper().strip() if parsed.get("company") else None
        year: Optional[int] = None
        if parsed.get("year") is not None:
            try:
                year = int(parsed["year"])
            except (TypeError, ValueError):
                pass
        quarter_raw = parsed.get("quarter")
        quarter = str(quarter_raw).upper().strip() if quarter_raw else None
        if quarter and quarter not in ("Q1", "Q2", "Q3", "Q4"):
            logger.warning("LLM returned unexpected quarter %r — ignoring", quarter)
            quarter = None

        return {"company": company, "year": year, "quarter": quarter}

    except Exception as exc:
        logger.warning("LLM filter extraction failed (%s) — returning empty filters.", exc)
        return _EMPTY.copy()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_query_filters(
    question: str,
    mode: Literal["llm", "regex"] = "llm",
    api_key: str = "",
    model: str = "gpt-4o-mini",
) -> QueryFilters:
    """
    Extract company, year, and quarter filters from a natural-language question.

    Args:
        question: User's question string.
        mode:     ``"llm"``   — LLM structured-output extraction (default, handles
                                arbitrary phrasing and company aliases).
                  ``"regex"`` — Rule-based extraction (zero LLM cost, covers
                                common explicit patterns).
        api_key:  OpenAI API key. Required only when ``mode="llm"``.
        model:    Model for LLM extraction (default: gpt-4o-mini — cheap/fast).

    Returns:
        dict: ``{'company': str|None, 'year': int|None, 'quarter': str|None}``
        Safe to unpack directly as kwargs to ``retrieve()`` / ``retrieve_llamaindex()``.
    """
    if mode == "regex":
        filters = _extract_regex(question)
    else:
        if not api_key:
            logger.warning(
                "extract_query_filters called with mode='llm' but no api_key — "
                "falling back to regex."
            )
            filters = _extract_regex(question)
        else:
            filters = _extract_llm(question, api_key=api_key, model=model)

    logger.debug("extract_query_filters(mode=%s) %r → %s", mode, question[:80], filters)
    return filters

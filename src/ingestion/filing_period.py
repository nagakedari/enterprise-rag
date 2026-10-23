"""
Fiscal-period extraction for SEC 10-Q filings.

Filing labels in this corpus ("2023 Q3 AAPL.pdf") are not calendar quarters
and are not consistent across companies — the quarter ended July 1, 2023 is
"2023 Q3" for Apple but "2023 Q2" for Intel. So the period a filing covers
is read from the filing itself and stored on every chunk as
period_start / period_end, letting retrieval filter by an actual date.

period_end comes from the cover-page line every 10-Q must carry:
    "For the quarterly period ended July 1, 2023"
(Income-statement column headers are NOT used for it: the first
"Three Months Ended" date can be a prior-year comparison column.)

period_start is derived, since filings rarely state it. Statement headers
tell the two fiscal-calendar styles apart:
  - calendar-quarter filers (Amazon, Microsoft) compare against the same
    day a year earlier ("September 30, 2023 / 2022")
        → the calendar quarter ending on period_end
  - 52/53-week filers (Apple, Intel, NVIDIA) compare against a different
    day ("July 1, 2023 / June 25, 2022"), and are also the default when no
    comparison date can be read
        → period_end minus 13 weeks + 1 day
The 52/53-week rule approximates a 14-week quarter (e.g. Apple's quarter
ended Dec 31, 2022) as 13 weeks, leaving its first week uncovered; it never
makes two filings' ranges overlap.
"""
import logging
import re
from datetime import date, datetime, timedelta
from typing import Iterable, Optional, Set, Tuple

logger = logging.getLogger(__name__)

_DATE = r"[A-Z][a-zA-Z]+\s*\d{1,2}\s*,\s*20\d\d"
_COVER_RE = re.compile(r"quarterly\s+period\s+ended\s*:?\s*(" + _DATE + ")", re.IGNORECASE)
# "Three Months Ended  September 30, 2023  September 30, 2022" and the
# shared-month/day form "Three Months Ended March 31, 2022 2023"
_MONTHS_ENDED_RE = re.compile(r"Months\s+Ended((?:\s*(?:" + _DATE + r"|20\d\d))+)", re.IGNORECASE)
_SHARED_DAY_RE = re.compile(r"([A-Z][a-zA-Z]+\s*\d{1,2})\s*,\s*20\d\d((?:\s+20\d\d)+)")

_COVER_PAGES = 3
_STATEMENT_PAGES = 12


def _parse_date(text: str) -> Optional[date]:
    cleaned = re.sub(r"\s+", " ", text).replace(" ,", ",").strip().title()
    try:
        return datetime.strptime(cleaned, "%B %d, %Y").date()
    except ValueError:
        return None


def _statement_dates(text: str) -> Set[date]:
    """Every date in a 'Months Ended' column header."""
    found: Set[date] = set()
    for match in _MONTHS_ENDED_RE.finditer(text):
        header = match.group(1)
        for raw in re.findall(_DATE, header):
            parsed = _parse_date(raw)
            if parsed:
                found.add(parsed)
        for month_day, years in _SHARED_DAY_RE.findall(header):
            for year in years.split():
                parsed = _parse_date(f"{month_day}, {year}")
                if parsed:
                    found.add(parsed)
    return found


def _period_start(period_end: date, statement_text: str) -> date:
    try:
        same_day_last_year = period_end.replace(year=period_end.year - 1)
    except ValueError:  # Feb 29
        same_day_last_year = None
    is_month_end = (period_end + timedelta(days=1)).day == 1
    if is_month_end and same_day_last_year and same_day_last_year in _statement_dates(statement_text):
        # The three calendar months ending on period_end
        year, month = period_end.year, period_end.month - 2
        if month <= 0:
            year, month = year - 1, month + 12
        return date(year, month, 1)
    return period_end - timedelta(weeks=13) + timedelta(days=1)


def extract_filing_period(page_texts: Iterable[str]) -> Tuple[Optional[date], Optional[date]]:
    """Return (period_start, period_end) from a filing's pages, or (None, None)."""
    pages = list(page_texts)
    cover = " ".join(pages[:_COVER_PAGES])
    match = _COVER_RE.search(cover)
    period_end = _parse_date(match.group(1)) if match else None
    if period_end is None:
        return None, None
    statements = re.sub(r"\s+", " ", " ".join(pages[:_STATEMENT_PAGES]))
    return _period_start(period_end, statements), period_end


def to_rfc3339(value: Optional[date]) -> Optional[str]:
    """Weaviate DATE properties take RFC 3339 timestamps."""
    return f"{value.isoformat()}T00:00:00Z" if value else None

"""
PDF parser for SEC 10-Q filings.

Extracts text page-by-page using PyMuPDF (fitz).
Metadata (company, quarter, year) is inferred from the filename convention:
    "2023 Q2 AAPL.pdf"  →  year=2023, quarter="Q2", company="AAPL"

Phase-1 note (demo limitation):
    Tables are extracted as plain text alongside prose.
    Phase 2 will add structure-aware table extraction.
"""
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class PageText:
    page_num: int   # 1-indexed
    text: str


@dataclass
class ParsedDocument:
    source_file: str
    company: str
    quarter: str        # e.g. "Q2"
    year: int           # e.g. 2023
    pages: List[PageText] = field(default_factory=list)
    total_pages: int = 0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_filename(filename: str) -> tuple[str, str, int]:
    """
    Extract (company, quarter, year) from filenames like '2023 Q2 AAPL.pdf'.
    Returns ('UNKNOWN', 'UNKNOWN', 0) on parse failure.
    """
    stem = Path(filename).stem   # e.g. '2023 Q2 AAPL'
    parts = stem.split()
    if len(parts) == 3:
        try:
            year = int(parts[0])
            quarter = parts[1]
            company = parts[2]
            return company, quarter, year
        except ValueError:
            pass
    logger.warning("Cannot parse metadata from filename '%s'; using defaults.", filename)
    return "UNKNOWN", "UNKNOWN", 0


def _clean_text(raw: str) -> str:
    """Collapse excessive blank lines while preserving paragraph breaks."""
    text = re.sub(r"[ \t]+", " ", raw)          # normalise horizontal whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)       # max two consecutive newlines
    return text.strip()


# ── Public API ────────────────────────────────────────────────────────────────

def parse_pdf(pdf_path: Path) -> ParsedDocument:
    """Parse a single PDF and return a ParsedDocument."""
    company, quarter, year = _parse_filename(pdf_path.name)
    pages: List[PageText] = []

    with fitz.open(str(pdf_path)) as doc:
        total_pages = len(doc)
        for page_num, page in enumerate(doc, start=1):
            raw_text = page.get_text("text")
            cleaned = _clean_text(raw_text)
            if cleaned:
                pages.append(PageText(page_num=page_num, text=cleaned))

    logger.info(
        "Parsed '%s'  company=%s  quarter=%s %s  pages=%d  non-empty=%d",
        pdf_path.name, company, quarter, year, total_pages, len(pages),
    )
    return ParsedDocument(
        source_file=pdf_path.name,
        company=company,
        quarter=quarter,
        year=year,
        pages=pages,
        total_pages=total_pages,
    )


def parse_all_pdfs(docs_path: Path) -> List[ParsedDocument]:
    """Parse every PDF in `docs_path` and return them sorted by filename."""
    pdf_files = sorted(docs_path.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {docs_path}")

    logger.info("Found %d PDF files in %s", len(pdf_files), docs_path)
    return [parse_pdf(p) for p in pdf_files]

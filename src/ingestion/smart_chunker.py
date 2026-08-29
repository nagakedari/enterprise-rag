"""
Parent-Child document chunker for SEC 10-Q filings (Phase 2).

Strategy:
    1.  Detect SEC section boundaries (ITEM / PART / NOTE headers) in the full
        document text and split there FIRST — no token limit at this stage.
        This guarantees that no parent chunk ever crosses a section boundary.
    2.  Within each section, apply a token-count-limited split to produce
        parent chunks (~1000 tokens).  Paragraph and sentence breaks are
        preferred over mid-sentence cuts.
    3.  Sub-split each parent into child chunks (~300 tokens) for embedding.
    4.  Each child carries:
            section_title  - the SEC section it belongs to (e.g. "ITEM 1A. RISK FACTORS")
            parent_text    - the parent block text returned to the LLM
            context_window - "[COMPANY QUARTER YEAR] Section: TITLE\\n\\nchild_text"
                             Embedded (not stored in Weaviate) so the vector
                             encodes company + section context.

Why section-first beats token-first:
    A token-count-first splitter can (a) merge two short adjacent sections into
    one parent (mixing Risk Factors with Properties) or (b) split a long section
    mid-paragraph, leaving the section header in one parent and the substance in
    the next.  Splitting at section boundaries first ensures every parent belongs
    to exactly one logical section.
"""
import bisect
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import SmartChunkConfig
from src.ingestion.pdf_parser import ParsedDocument

logger = logging.getLogger(__name__)


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class SmartTextChunk:
    """A child chunk produced by the parent-child chunker."""
    text: str               # child chunk text (~300 tokens) — used for embedding
    parent_text: str        # parent chunk text (~1000 tokens) — returned to the LLM
    parent_id: str          # deterministic UUID for deduplication at retrieval time
    chunk_index: int        # global child chunk index within the document
    token_count: int        # child chunk token count
    page_num: int           # page where this chunk's section starts
    file_name: str          # PDF filename  e.g. "2023 Q2 AAPL.pdf"
    source_file: str        # same as file_name; kept for pipeline compatibility
    company: str
    quarter: str
    year: int
    section_title: str      # SEC section header (e.g. "ITEM 1A. RISK FACTORS")
    last_updated_date: str  # ISO-8601 UTC timestamp of ingestion run
    context_window: str     # enriched text embedded (not stored in Weaviate)


# ── Section boundary detection ─────────────────────────────────────────────────

# Matches ITEM / PART / NOTE headers at the start of a line.
# Captures the full first line (up to 100 chars) as the section title.
_SECTION_HEADER_RE = re.compile(
    r"^((?:ITEM|Item)\s+\d+[A-Z]?\.?\s*[^\n]{0,100}"
    r"|(?:PART|Part)\s+[IVX]+\s*[^\n]{0,100}"
    r"|(?:NOTE|Note)\s+\d+[^\n]{0,100})",
    re.MULTILINE,
)


def _find_section_offsets(text: str) -> List[Tuple[int, str]]:
    """Return sorted (char_offset, title) pairs for each SEC section header."""
    results: List[Tuple[int, str]] = []
    for m in _SECTION_HEADER_RE.finditer(text):
        results.append((m.start(), m.group(1).strip()))
    return results


def _split_into_sections(text: str) -> List[Tuple[str, str, int]]:
    """
    Split *text* at SEC section boundaries.

    Returns a list of (section_text, section_title, start_char_offset) triples.
    Text before the first header is labelled "PREAMBLE".
    Falls back to a single block with an empty title if no headers are found.
    """
    offsets = _find_section_offsets(text)
    if not offsets:
        return [(text, "", 0)]

    sections: List[Tuple[str, str, int]] = []

    if offsets[0][0] > 0:
        preamble = text[: offsets[0][0]].strip()
        if preamble:
            sections.append((preamble, "PREAMBLE", 0))

    for i, (start, title) in enumerate(offsets):
        end = offsets[i + 1][0] if i + 1 < len(offsets) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            sections.append((section_text, title, start))

    return sections


# ── Page number lookup ─────────────────────────────────────────────────────────

def _build_page_index(
    doc: ParsedDocument,
) -> Tuple[str, List[int], List[int]]:
    """
    Concatenate all non-empty page texts separated by '\\n\\n' and record where
    each page begins in the resulting string.

    Returns:
        full_text   - complete document text as one string
        page_starts - sorted char offsets of each page's start
        page_nums   - page numbers corresponding to each entry in page_starts
    """
    parts: List[str] = []
    page_starts: List[int] = []
    page_nums: List[int] = []
    offset = 0
    for page in doc.pages:
        if not page.text.strip():
            continue
        page_starts.append(offset)
        page_nums.append(page.page_num)
        parts.append(page.text)
        offset += len(page.text) + 2   # '\n\n' separator added below

    full_text = "\n\n".join(parts)
    return full_text, page_starts, page_nums


def _page_at(char_offset: int, page_starts: List[int], page_nums: List[int]) -> int:
    """Return the page number for a given character offset (binary search)."""
    if not page_starts:
        return 1
    idx = bisect.bisect_right(page_starts, char_offset) - 1
    return page_nums[max(0, idx)]


# ── Context window builder ─────────────────────────────────────────────────────

def _build_context_window(
    child_text: str,
    company: str,
    quarter: str,
    year: int,
    section_title: str,
) -> str:
    """
    Prepend filing identity and section label to the child text.
    This string is what gets embedded — not the raw child text.
    """
    header = f"[{company} {quarter} {year}]"
    if section_title:
        header += f" Section: {section_title}"
    return f"{header}\n\n{child_text}"


# ── Token counter ──────────────────────────────────────────────────────────────

def _token_len(text: str, enc: tiktoken.Encoding) -> int:
    return len(enc.encode(text))


# ── Public API ────────────────────────────────────────────────────────────────

def chunk_document_smart(
    doc: ParsedDocument,
    config: SmartChunkConfig,
    last_updated_date: Optional[str] = None,
) -> List[SmartTextChunk]:
    """
    Produce SmartTextChunks for a single ParsedDocument.

    Flow:
        pages → full_text → section split (no token cap)
              → parent split within each section (token-capped)
              → child split within each parent
              → context_window enrichment per child
    """
    if last_updated_date is None:
        last_updated_date = datetime.now(timezone.utc).isoformat()
    enc = tiktoken.get_encoding(config.encoding)

    # ── 1. Build full text and page boundary index ────────────────────────────
    full_text, page_starts, page_nums = _build_page_index(doc)
    if not full_text.strip():
        return []

    # ── 2. Split at SEC section boundaries (no token limit) ──────────────────
    sections = _split_into_sections(full_text)

    logger.debug(
        "SectionSplit '%s' → %d sections: %s",
        doc.source_file,
        len(sections),
        [title for _, title, _ in sections],
    )

    # ── 3. Within each section, split into parent chunks ─────────────────────
    # Generic separators only — SEC headers were consumed in step 2.
    parent_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=config.encoding,
        chunk_size=config.parent_max_tokens,
        chunk_overlap=config.parent_overlap_tokens,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    child_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=config.encoding,
        chunk_size=config.child_max_tokens,
        chunk_overlap=config.child_overlap_tokens,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: List[SmartTextChunk] = []
    child_index = 0
    total_parents = 0

    for section_idx, (section_text, section_title, start_offset) in enumerate(sections):
        if not section_text.strip():
            continue

        section_page = _page_at(start_offset, page_starts, page_nums)
        parent_texts = parent_splitter.split_text(section_text)
        total_parents += len(parent_texts)

        # Track position within section_text so each parent gets its own page,
        # not the section-header page (which may be many pages before the parent's text).
        parent_cursor = 0

        for parent_idx, parent_text in enumerate(parent_texts):
            parent_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_DNS,
                    f"{doc.source_file}::section::{section_idx}::parent::{parent_idx}",
                )
            )

            # Locate this parent inside the section to find its actual page.
            found_at = section_text.find(parent_text, parent_cursor)
            if found_at >= 0:
                parent_page = _page_at(start_offset + found_at, page_starts, page_nums)
                parent_cursor = found_at + 1  # advance past start; overlap is fine
            else:
                parent_page = section_page  # fallback: section start page

            child_texts = child_splitter.split_text(parent_text)

            for child_text in child_texts:
                context_window = _build_context_window(
                    child_text, doc.company, doc.quarter, doc.year, section_title
                )
                chunks.append(SmartTextChunk(
                    text=child_text,
                    parent_text=parent_text,
                    parent_id=parent_id,
                    chunk_index=child_index,
                    token_count=_token_len(child_text, enc),
                    page_num=parent_page,
                    file_name=Path(doc.source_file).name,
                    source_file=doc.source_file,
                    company=doc.company,
                    quarter=doc.quarter,
                    year=doc.year,
                    section_title=section_title,
                    last_updated_date=last_updated_date,
                    context_window=context_window,
                ))
                child_index += 1

    logger.info(
        "SmartChunker '%s' → %d sections → %d parents → %d child chunks",
        doc.source_file, len(sections), total_parents, len(chunks),
    )
    return chunks


def chunk_documents_smart(
    docs: List[ParsedDocument],
    config: SmartChunkConfig,
    last_updated_date: Optional[str] = None,
) -> List[SmartTextChunk]:
    """
    Chunk all parsed documents and return a flat list of SmartTextChunks.
    All chunks share the same last_updated_date (defaults to current UTC time).
    """
    if last_updated_date is None:
        last_updated_date = datetime.now(timezone.utc).isoformat()
    all_chunks: List[SmartTextChunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document_smart(doc, config, last_updated_date))
    logger.info(
        "SmartChunker total across %d documents: %d child chunks",
        len(docs), len(all_chunks),
    )
    return all_chunks

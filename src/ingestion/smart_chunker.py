"""
Parent-Child document chunker for SEC 10-Q filings (Phase 2).

Strategy:
    1.  Split each document into parent chunks (~1000 tokens) using
        RecursiveCharacterTextSplitter with SEC-aware separators:
            "\n\nITEM "  →  "\n\nPART "  →  "\n\nNOTE "
            "\n\n"  →  "\n"  →  ". "  →  " "  →  ""
        Trying SEC section boundaries first keeps ITEM / NOTE sections
        intact as much as possible within the token budget.
    2.  Sub-split each parent into child chunks (~300 tokens) using a
        tighter RecursiveCharacterTextSplitter (no SEC separators needed
        since the parent is already section-scoped).
    3.  Each child chunk carries:
            text           - child text (small, ~300 tokens) → embedded
            parent_text    - parent text (large, ~1000 tokens) → LLM context
            parent_id      - deterministic UUID; deduplicated at retrieval
            section_title  - nearest section header found in the parent text
            context_window - "[COMPANY QUARTER YEAR] Section: TITLE\\n\\nchild_text"
                             This is what gets embedded (not raw child text),
                             so the vector encodes company + section context.

Why this helps retrieval:
    SEC filing body text rarely mentions the company name or the filing
    period inline.  Prepending "[AAPL Q2 2023] Section: RISK FACTORS"
    means a query like "Apple Q2 2023 liquidity risk" will score higher
    against the right chunks even when the body text never says "Apple".
"""
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import tiktoken
from langchain_core.documents import Document
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
    page_num: int           # page this chunk originates from
    file_name: str          # PDF filename  e.g. "2023 Q2 AAPL.pdf"
    source_file: str        # same as file_name; kept for pipeline compatibility
    company: str
    quarter: str
    year: int
    section_title: str      # nearest preceding section header ("" if none)
    last_updated_date: str  # ISO-8601 UTC timestamp of ingestion run
    context_window: str     # enriched text that gets embedded (not stored in Weaviate)


# ── Section title detection ────────────────────────────────────────────────────

# Matches: ITEM 1A. / PART II / NOTE 5 at the start of a line
_HEADER_RE = re.compile(
    r"^(?:ITEM\s+\d+[A-Z]?\.?|PART\s+[IVX]+|NOTE\s+\d+)",
    re.IGNORECASE | re.MULTILINE,
)
# Fallback: ALL-CAPS lines of 5–80 chars (e.g. "RISK FACTORS")
_ALLCAPS_RE = re.compile(r"^[A-Z][A-Z\s\-]{4,}$", re.MULTILINE)


def _extract_section_title(text: str) -> str:
    """
    Return the last section header found in *text*, or ''.
    Prefers explicit ITEM/PART/NOTE patterns; falls back to ALL-CAPS lines.
    """
    matches = list(_HEADER_RE.finditer(text))
    if matches:
        return matches[-1].group().strip()
    caps = [
        m.group().strip()
        for m in _ALLCAPS_RE.finditer(text)
        if 5 <= len(m.group().strip()) <= 80
    ]
    return caps[-1] if caps else ""


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

    This string is what gets embedded — not the raw child text.  The prefix
    anchors the vector to the correct company, period, and section so that
    queries naming the company or section retrieve better matches.
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

    Args:
        doc:               Parsed PDF document.
        config:            Smart chunking parameters.
        last_updated_date: ISO-8601 UTC timestamp to stamp on every chunk.
                           Defaults to the current UTC time if not provided.
                           Pass a single value from the pipeline so all chunks
                           in one ingestion run share the same timestamp.

    Flow:
        pages  →  parent split  →  child split per parent
        → section title detection per parent
        → context_window enrichment per child
    """
    if last_updated_date is None:
        last_updated_date = datetime.now(timezone.utc).isoformat()
    enc = tiktoken.get_encoding(config.encoding)

    # Parent splitter: SEC section boundaries tried first, then progressively
    # finer boundaries.  chunk_overlap keeps sentence context at boundaries.
    parent_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=config.encoding,
        chunk_size=config.parent_max_tokens,
        chunk_overlap=config.parent_overlap_tokens,
        # separators=[
        #     "\n\nITEM ", "\n\nPART ", "\n\nNOTE ",
        #     "\n\n", "\n", ". ", " ", "",
        # ],
        separators = [
            "\n\nPART ",      # Highest level (Part I, II)
            "\n\nItem ",      # Standard SEC Item headers
            "\n\nITEM ",      # Catch-all for uppercase versions
            "\n\nNote ",      # For the detailed Financial Notes
            "\n\nNOTE ",      # Catch-all for uppercase Notes
            "\n\nEXHIBIT ",   # For the legal/signature sections
            "\n\n",           # Paragraphs
            "\n",             # Line breaks
            ". ",             # Sentences
            " ",              # Words
            ""                # Characters
        ]
    )

    # Child splitter: finer splits within each parent; no need for SEC separators
    # since the parent is already section-scoped.
    child_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=config.encoding,
        chunk_size=config.child_max_tokens,
        chunk_overlap=config.child_overlap_tokens,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # One LangChain Document per page preserves page_num in metadata
    lc_docs = [
        Document(page_content=page.text, metadata={"page_num": page.page_num})
        for page in doc.pages
        if page.text.strip()
    ]

    parent_docs = parent_splitter.split_documents(lc_docs)

    chunks: List[SmartTextChunk] = []
    child_index = 0
    current_section_title = ""

    for parent_idx, parent_doc in enumerate(parent_docs):
        parent_text = parent_doc.page_content
        page_num = int(parent_doc.metadata.get("page_num", 1))

        # Update running section title from this parent's content
        detected = _extract_section_title(parent_text)
        if detected:
            current_section_title = detected

        parent_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_DNS,
                f"{doc.source_file}::parent::{parent_idx}",
            )
        )

        child_texts = child_splitter.split_text(parent_text)

        for child_text in child_texts:
            context_window = _build_context_window(
                child_text, doc.company, doc.quarter, doc.year, current_section_title
            )
            chunks.append(SmartTextChunk(
                text=child_text,
                parent_text=parent_text,
                parent_id=parent_id,
                chunk_index=child_index,
                token_count=_token_len(child_text, enc),
                page_num=page_num,
                file_name=Path(doc.source_file).name,
                source_file=doc.source_file,
                company=doc.company,
                quarter=doc.quarter,
                year=doc.year,
                section_title=current_section_title,
                last_updated_date=last_updated_date,
                context_window=context_window,
            ))
            child_index += 1

    logger.info(
        "SmartChunker '%s' → %d parent chunks → %d child chunks",
        doc.source_file, len(parent_docs), len(chunks),
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
